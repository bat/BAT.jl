# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# The transport lock owns these fields, except for the atomic fast-path flag.
# Renderer state has a separate owner; no stop/wakeup operation enters Makie.
Base.@kwdef mutable struct _MakieLiveControl
    phase::Symbol = :fresh
    outcome::Symbol = :inactive
    reason::Symbol = :none
    accepting::Threads.Atomic{Bool} = Threads.Atomic{Bool}(false)
    tick::Bool = false
    last_tick::UInt64 = time_ns()
    timer::Union{Nothing,Timer} = nothing
    task::Union{Nothing,Task} = nothing
    disposal_task::Union{Nothing,Task} = nothing
    shutdown_started::Union{Nothing,UInt64} = nothing
    error::Union{Nothing,CapturedException} = nothing
    error_phase::Symbol = :none
    error_reported::Bool = false
    cleanup_error::Union{Nothing,CapturedException} = nothing
    data_complete::Bool = false
    last_publication::Int = 0
    owner_thread::Int = 0
    owner_task::Union{Nothing,Task} = nothing
    initializing::Bool = false
    view::Union{Nothing,_LiveView} = nothing
end

_live_elapsed(start::UInt64) = (time_ns() - start) / 1.0e9
_live_can_publish_locked(vis) = vis.content.lifecycle.phase in (:running, :draining) &&
    vis.content.lifecycle.outcome === :running

function _claim_live_visualizer!(vis)
    isnothing(vis.content) && throw(ArgumentError("Create a fresh BATVisualizer for sampling"))
    lock(vis.content.buffer_lock) do
        ctl = vis.content.lifecycle
        ctl.phase === :fresh || throw(ArgumentError("Create a fresh BATVisualizer for sampling"))
        ctl.phase = :starting
        ctl.owner_thread = Threads.threadid()
        ctl.owner_task = current_task()
    end
    return nothing
end

function _activate_live_visualizer!(vis)
    lock(vis.content.buffer_lock) do
        ctl = vis.content.lifecycle
        ctl.phase === :starting || return false
        ctl.phase = :running
        ctl.outcome = :running
        ctl.accepting[] = true
        ctl.last_tick = time_ns()
        return true
    end
end

function _stop_live_locked!(vis, reason::Symbol, exception = nothing; phase::Symbol = reason)
    ctl = vis.content.lifecycle
    ctl.accepting[] = false
    # Closing a completed view does not undo the completed observation history.
    if ctl.outcome !== :complete
        ctl.phase === :quiesced || (ctl.phase = :stopping)
        if ctl.outcome in (:inactive, :running)
            ctl.outcome = reason === :aborted ? :aborted :
                reason in (:closed, :disposed, :disabled) ? :disabled : :failed
            ctl.reason = reason
        end
    end
    if !isnothing(exception) && isnothing(ctl.error)
        ctl.error = exception isa CapturedException ? exception : CapturedException(exception, Any[])
        ctl.error_phase = phase
    end
    ctl.tick = true
    notify(vis.content.buffer_cond; all=true)
    return nothing
end

function BAT.request_visualizer_stop!(vis::BATVisualizer{BATMakieVisualization}; reason::Symbol=:disabled, exception=nothing)
    isnothing(vis.content) && return nothing
    lock(vis.content.buffer_lock) do
        _stop_live_locked!(vis, reason, exception)
    end
    return nothing
end

function _fail_visualizer!(vis, err, bt, phase::Symbol)
    captured = CapturedException(err, bt)
    first_error, complete = lock(vis.content.buffer_lock) do
        first_error = !vis.content.lifecycle.error_reported
        _stop_live_locked!(vis, :failed, captured; phase=phase)
        vis.content.lifecycle.error_reported = true
        first_error, vis.content.lifecycle.data_complete
    end
    # Diagnostics must not replace a sampling exception or escape a producer.
    if first_error
        try
            message = phase in (:prepare, :startup) ? "Live visualization startup failed" :
                complete ? "Completed visualization failed; recorded observation history remains complete" :
                "Live visualization failed; observation is incomplete, sampling continues"
            @error message phase exception=(err, bt)
        catch
        end
    end
    return nothing
end

function _record_live_cleanup_error!(vis, err, bt)
    lock(vis.content.buffer_lock) do
        ctl = vis.content.lifecycle
        isnothing(ctl.cleanup_error) && (ctl.cleanup_error = CapturedException(err, bt))
    end
    return nothing
end

function _live_control_tick!(vis)
    try
        lock(vis.content.buffer_lock) do
            ctl = vis.content.lifecycle
            if ctl.phase === :running && _live_elapsed(ctl.last_tick) >= vis.backend.poll_interval
                ctl.tick = true
                ctl.last_tick = time_ns()
            end
            # Also wake capacity waiters to check their own monotonic deadline.
            notify(vis.content.buffer_cond; all=true)
        end
    catch err
        _fail_visualizer!(vis, err, catch_backtrace(), :wakeup)
    end
    return nothing
end

function _close_live_timer!(vis)
    timer = lock(() -> vis.content.lifecycle.timer, vis.content.buffer_lock)
    isnothing(timer) && return nothing
    close(timer)
    lock(vis.content.buffer_lock) do
        vis.content.lifecycle.timer === timer && (vis.content.lifecycle.timer = nothing)
    end
    return nothing
end

function _release_live_transport!(vis)
    lock(vis.content.buffer_lock) do
        for buffers in (vis.content.output_buffer, vis.content.pending_samples,
                        vis.content.output_timing, vis.content.pending_timing)
            empty!(buffers)
        end
        vis.content.n_buffer_steps[] = 0
        vis.content.buffer_dirty[] = false
    end
    return nothing
end

function _finalize_live_run!(vis)
    try
        _close_live_timer!(vis)
    catch err
        _record_live_cleanup_error!(vis, err, catch_backtrace())
    end
    lock(vis.content.buffer_lock) do
        ctl = vis.content.lifecycle
        ctl.outcome === :running && _stop_live_locked!(vis, :failed)
        ctl.accepting[] = false
        notify(vis.content.buffer_cond; all=true)
    end
    _release_live_transport!(vis)
    if vis.content.lifecycle.outcome !== :complete
        try
            _request_view_disposal!(vis)
            _cleanup_live_view!(vis)
        catch err
            _record_live_cleanup_error!(vis, err, catch_backtrace())
        end
    end
    return nothing
end

# Callables are the normal ingestion/final-publication operations. Keeping this
# task owner separate also permits deterministic tests of real wait/fault paths.
function _start_live_listener!(vis, tick!::F, finish!::G) where {F,G}
    _activate_live_visualizer!(vis) || return nothing
    ctl = vis.content.lifecycle
    period = max(0.001, min(vis.backend.poll_interval, vis.backend.observer_timeout / 4, 0.1))
    try
        ctl.timer = Timer(_ -> _live_control_tick!(vis), period; interval=period)
        task = @task begin
            phase = :selection
            try
                _apply_vsel!(vis, vis.backend.vsel)
                while true
                    mode = lock(vis.content.buffer_lock) do
                        while ctl.phase === :running && !ctl.tick
                            wait(vis.content.buffer_cond)
                        end
                        ctl.tick = false
                        ctl.phase
                    end
                    mode === :running || break
                    phase = :update
                    tick!()
                end
                if lock(() -> ctl.phase === :draining, vis.content.buffer_lock)
                    phase = :finalize
                    finish!()
                    lock(vis.content.buffer_lock) do
                        if ctl.phase === :draining
                            ctl.outcome = :complete
                        end
                    end
                end
            catch err
                _fail_visualizer!(vis, err, catch_backtrace(), phase)
            finally
                _finalize_live_run!(vis)
            end
        end
        # Publish the handle before scheduling; partial startup has an owner.
        ctl.task = task
        schedule(task)
    catch err
        _fail_visualizer!(vis, err, catch_backtrace(), :startup)
        if !isnothing(ctl.task) && !istaskstarted(ctl.task)
            ctl.task = nothing
        end
        _finalize_live_run!(vis)
        rethrow()
    end
    return nothing
end

function _live_error_summary(error, phase)
    isnothing(error) && return nothing
    message = try
        sprint(showerror, error.ex)
    catch
        string(typeof(error.ex))
    end
    return (type=string(typeof(error.ex)), message=message, phase=phase)
end

function _live_quiesced_locked!(vis)
    ctl = vis.content.lifecycle
    done = isnothing(ctl.task) || istaskdone(ctl.task)
    done && (ctl.task = nothing)
    if !isnothing(ctl.disposal_task) && istaskdone(ctl.disposal_task)
        ctl.disposal_task = nothing
    end
    quiesced = done && isnothing(ctl.timer) && !ctl.initializing && ctl.phase !== :starting
    if quiesced && ctl.phase in (:draining, :stopping, :quiesced)
        ctl.phase = :quiesced
        ctl.owner_task = nothing
    end
    return quiesced
end

function BAT.visualizer_status(vis::BATVisualizer{BATMakieVisualization})
    isnothing(vis.content) && return invoke(BAT.visualizer_status, Tuple{BATVisualizer}, vis)
    snapshot = lock(vis.content.buffer_lock) do
        ctl = vis.content.lifecycle
        # Release completed task closures, which otherwise retain sampler state.
        quiesced = _live_quiesced_locked!(vis)
        disposed = isnothing(ctl.view) ? false : ctl.view.disposed[]
        view_pending = !isnothing(ctl.view) && ctl.view.cleanup_pending[]
        valid = !isnothing(ctl.view) && ctl.view.presentation_valid[]
        phase = quiesced && ctl.phase in (:stopping, :quiesced) ? :quiesced : ctl.phase
        (phase=phase, outcome=ctl.outcome, reason=ctl.reason,
            data_complete=ctl.data_complete, last_publication=ctl.last_publication,
            quiesced=quiesced, cleanup_pending=!quiesced || view_pending || !isnothing(ctl.disposal_task),
            view_disposed=disposed, presentation_valid=valid,
            error=ctl.error, error_phase=ctl.error_phase, cleanup_error=ctl.cleanup_error)
    end
    return (; Base.structdiff(snapshot, NamedTuple{(:error_phase, :error, :cleanup_error)})...,
        error=_live_error_summary(snapshot.error, snapshot.error_phase),
        cleanup_error=_live_error_summary(snapshot.cleanup_error, :cleanup))
end

function _await_live_run!(vis, start::UInt64)
    # This independent wait owns its timer; the observation timer may be closed.
    remaining = max(0.0, vis.backend.observer_timeout - _live_elapsed(start))
    # Do not format arbitrary exceptions in a deadline predicate.
    settled = () -> lock(vis.content.buffer_lock) do
        quiesced = _live_quiesced_locked!(vis)
        ctl = vis.content.lifecycle
        quiesced && isnothing(ctl.disposal_task) &&
            (isnothing(ctl.view) || !ctl.view.cleanup_pending[])
    end
    timedwait(settled, remaining;
        pollint=max(0.001, min(0.01, vis.backend.observer_timeout / 10)))
end

function BAT.finish_visualizer!(vis::BATVisualizer{BATMakieVisualization}; success::Bool=true)
    isnothing(vis.content) && return nothing
    ctl = vis.content.lifecycle
    start = lock(vis.content.buffer_lock) do
        ctl.phase === :fresh && return nothing
        # Nested rollback/owner cleanup shares one allowance while resources
        # remain pending. A later disposal of a settled view gets a new budget.
        isnothing(ctl.shutdown_started) && (ctl.shutdown_started = time_ns())
        if success && ctl.phase === :running
            ctl.accepting[] = false
            ctl.phase = :draining
            ctl.tick = true
            notify(vis.content.buffer_cond; all=true)
        elseif !success && ctl.phase !== :fresh
            _stop_live_locked!(vis, :aborted)
        end
        ctl.shutdown_started
    end
    isnothing(start) && return nothing
    # Startup failures have no listener to roll back the partial view.
    if isnothing(ctl.task) && ctl.phase === :stopping
        ctl.task = @task _finalize_live_run!(vis)
        schedule(ctl.task)
    end
    ctl.task === current_task() && return nothing
    try
        if _await_live_run!(vis, start) === :timed_out
            _fail_visualizer!(vis, ErrorException("Visualization shutdown exceeded observer_timeout"), Any[], :shutdown_timeout)
            _close_live_timer!(vis)
        else
            lock(vis.content.buffer_lock) do
                ctl.shutdown_started = nothing
            end
        end
    catch err
        err isa InterruptException && begin
            BAT.request_visualizer_stop!(vis; reason=:aborted)
            rethrow()
        end
        _record_live_cleanup_error!(vis, err, catch_backtrace())
        BAT.request_visualizer_stop!(vis; reason=:failed)
    end
    return nothing
end

function _schedule_live_view_cleanup!(vis)
    ctl = vis.content.lifecycle
    # A wrong-thread callback may request cleanup but cannot perform GUI work.
    # Keep that request owned/pending until the view's supported context resumes.
    ctl.owner_thread in (0, Threads.threadid()) || return nothing
    task = lock(vis.content.buffer_lock) do
        view = ctl.view
        (isnothing(view) || !view.cleanup_pending[] || view.disposed[] || view.cleanup_running[]) && return nothing
        if isnothing(ctl.disposal_task) || istaskdone(ctl.disposal_task)
            task = @task begin
                try
                    _cleanup_live_view!(vis)
                catch err
                    _record_live_cleanup_error!(vis, err, catch_backtrace())
                end
            end
            ctl.disposal_task = task
            return task
        end
        return nothing
    end
    if !isnothing(task)
        try
            schedule(task)
        catch err
            _record_live_cleanup_error!(vis, err, catch_backtrace())
            lock(vis.content.buffer_lock) do
                ctl.disposal_task === task && !istaskstarted(task) && (ctl.disposal_task = nothing)
            end
        end
    end
    return nothing
end

function BAT.dispose_visualizer!(vis::BATVisualizer{BATMakieVisualization})
    isnothing(vis.content) && return nothing
    ctl = vis.content.lifecycle
    ctl.owner_thread in (0, Threads.threadid()) || throw(ArgumentError("Dispose a live view on its owning thread"))
    BAT.request_visualizer_stop!(vis; reason=:disposed)
    _request_view_disposal!(vis)
    _schedule_live_view_cleanup!(vis)
    return nothing
end
