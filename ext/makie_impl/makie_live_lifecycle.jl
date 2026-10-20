# This file is a part of BAT.jl, licensed under the MIT License (MIT).

const _LiveDiagnostic = NamedTuple{(:type, :message, :phase),Tuple{String,String,Symbol}}

# Diagnostics must not retain graphs, tasks or arbitrary exception payloads.
# Inspect only trusted concrete strings: even ErrorException.msg can be a
# user-defined AbstractString. No user formatting belongs in stop/status paths.
function _live_diagnostic_prefix(message::String, limit::Int)
    i, stop = 1, 0
    while i <= ncodeunits(message)
        next = nextind(message, i)
        next - 1 > limit && break
        stop = next - 1
        i = next
    end
    text = String(Vector{UInt8}(view(codeunits(message), 1:stop)))
    return stop < ncodeunits(message) ? text * " [truncated]" : text
end

function _live_diagnostic_type(name::Symbol)
    # String(Symbol) copies the entire name. Symbols are immutable and their
    # NUL-terminated storage is owned by Julia; inspect/copy only a bounded
    # prefix. Four lookahead bytes let the string helper preserve UTF-8 bounds.
    prefix = GC.@preserve name begin
        ptr = Base.unsafe_convert(Ptr{UInt8}, name)
        n = 0
        while n < 260 && unsafe_load(ptr, n + 1) != 0
            n += 1
        end
        unsafe_string(ptr, n)
    end
    return _live_diagnostic_prefix(prefix, 256)
end

function _live_error_summary(error, phase::Symbol)
    isnothing(error) && return nothing
    err = error isa CapturedException ? getfield(error, :ex) : error
    name = getfield(getfield(typeof(err), :name), :name)
    type = _live_diagnostic_type(name)
    raw = err isa Union{ErrorException,ArgumentError} ? getfield(err, :msg) : nothing
    message = raw isa String ? _live_diagnostic_prefix(raw, 4096) :
        err isa InterruptException ? "Interrupted" : "Exception details are not retained"
    return (type=type, message=message, phase=phase)
end

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
    # Private task-construction state, retained through cleanup and retries.
    # Fresh visualizers deliberately repeat this origin; it is not a unique seed.
    task_rng::Random.Xoshiro = Random.Xoshiro(0x4241547669733031)
    disposal_requested::Bool = false
    disposal_retry::Bool = false
    release_committed::Bool = false
    release_complete::Bool = false
    shutdown_started::Union{Nothing,UInt64} = nothing
    error::Union{Nothing,_LiveDiagnostic} = nothing
    error_reported::Bool = false
    cleanup_error::Union{Nothing,_LiveDiagnostic} = nothing
    data_complete::Bool = false
    last_publication::Int = 0
    owner_thread::Int = 0
    owner_task::Union{Nothing,Task} = nothing
    # One acquisition may continue without a view after early disposal. The
    # permit must not retain a numerical task after observation is released.
    startup_owner::Union{Nothing,WeakRef} = nothing
    initializing::Bool = false
    view::Union{Nothing,_LiveView} = nothing
end

# Caller holds buffer_lock and defers SIGINT. Only the trusted task/Timer
# enrollment below belongs here: no user work, explicit waits or recursive
# entry on this control. Preserve Base's existing constructor-internal locks.
function _with_live_task_rng(f::F, ctl::_MakieLiveControl) where {F}
    rng = Random.default_rng()
    saved = copy(rng)
    try
        copy!(rng, ctl.task_rng)
        try
            return f()
        finally
            # A constructed task consumes its split even if scheduling fails.
            copy!(ctl.task_rng, rng)
        end
    finally
        # Restore the actual initiating task, including its child-splitting
        # state; cleanup need not be initiated by the original sampling owner.
        copy!(rng, saved)
    end
end

_live_elapsed(start::UInt64) = (time_ns() - start) / 1.0e9
_live_can_publish_locked(vis) = vis.content.lifecycle.phase in (:running, :draining) &&
    vis.content.lifecycle.outcome === :running && !vis.content.lifecycle.disposal_requested

# Caller owns startup or an admitted view operation. Never recreate a released
# graph, and never update its inputs as a substitute for releasing its caches.
function _live_graph(vis)
    graph = vis.content.graph[]
    isnothing(graph) && error("Live observation history has been released")
    return graph
end

function _claim_live_visualizer!(vis, claimed::Base.RefValue{Bool}=Ref(false))
    isnothing(vis.content) && throw(ArgumentError("Create a fresh BATVisualizer for sampling"))
    owner = current_task()
    permit = WeakRef(owner)
    lock(vis.content.buffer_lock) do
        ctl = vis.content.lifecycle
        ctl.phase === :fresh || throw(ArgumentError("Create a fresh BATVisualizer for sampling"))
        # The caller's cleanup scope already exists. A pending interrupt must
        # see its acknowledgment even if delivered before this call returns.
        Base.disable_sigint() do
            ctl.phase = :starting
            ctl.owner_thread = Threads.threadid()
            ctl.owner_task = owner
            ctl.startup_owner = permit
            claimed[] = true
        end
    end
    return nothing
end

function _activate_live_visualizer!(vis)
    lock(vis.content.buffer_lock) do
        ctl = vis.content.lifecycle
        (ctl.phase === :starting && !ctl.disposal_requested) || return false
        ctl.phase = :running
        ctl.outcome = :running
        ctl.accepting[] = true
        ctl.last_tick = time_ns()
        return true
    end
end

function _stop_live_locked!(vis, reason::Symbol, exception = nothing; phase::Symbol = reason)
    # The caller owns buffer_lock; acquisition stays interruptible. Commit the
    # state and wakeup together: an interrupt inside notify could otherwise
    # remove a waiter before it has been scheduled. No user/renderer work here.
    Base.disable_sigint() do
        ctl = vis.content.lifecycle
        ctl.accepting[] = false
        # Outcome records the first stop. A later failure must still revoke
        # pending startup, even when that first outcome was benign disposal.
        if !(reason in (:closed, :disposed, :disabled)) || !isnothing(exception)
            ctl.startup_owner = nothing
        end
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
            ctl.error = _live_error_summary(exception, phase)
        end
        ctl.tick = true
        notify(vis.content.buffer_cond; all=true)
    end
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
        isnothing(ctl.cleanup_error) && (ctl.cleanup_error = _live_error_summary(err, :cleanup))
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
            _schedule_live_view_cleanup!(vis)
        catch err
            _record_live_cleanup_error!(vis, err, catch_backtrace())
        end
    end
    _live_release_event!(vis)
    return nothing
end

# The listener owns the normal ingestion/final-publication callables. Release
# waits for this task to terminate, not merely for its finally block to start.
function _run_live_listener!(vis, tick!::F, finish!::G) where {F,G}
    ctl = vis.content.lifecycle
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
                ctl.phase === :draining && (ctl.outcome = :complete)
            end
        end
    catch err
        _fail_visualizer!(vis, err, catch_backtrace(), phase)
    finally
        _finalize_live_run!(vis)
    end
    return nothing
end

function _start_live_listener!(vis, tick!::F, finish!::G) where {F,G}
    ctl = vis.content.lifecycle
    period = max(0.001, min(vis.backend.poll_interval, vis.backend.observer_timeout / 4, 0.1))
    try
        lock(vis.content.buffer_lock) do
            Base.disable_sigint() do
                _activate_live_visualizer!(vis) || return nothing
                _with_live_task_rng(ctl) do
                    ctl.timer = Timer(_ -> _live_control_tick!(vis), period; interval=period)
                    task = @task _run_live_listener!(vis, tick!, finish!)
                    # Store before scheduling; a queued task must remain owned even
                    # when it has not yet started. No user work in this enrollment.
                    ctl.task = task
                    schedule(task)
                end
            end
        end
    catch err
        _fail_visualizer!(vis, err, catch_backtrace(), :startup)
        _request_view_disposal!(vis)
        _schedule_live_view_cleanup!(vis)
        rethrow()
    end
    return nothing
end

# Retiring a failed release task must not certify a post-commit failure as
# success. Keep the request pending so an explicit disposal can retry it.
function _retire_live_disposal_locked!(vis)
    ctl = vis.content.lifecycle
    task = ctl.disposal_task
    if !isnothing(task) && istaskdone(task)
        failed = istaskfailed(task)
        ctl.release_complete = ctl.release_committed && !failed
        if failed && isnothing(ctl.cleanup_error)
            ctl.cleanup_error = _live_error_summary(TaskFailedException(task), :cleanup)
        end
        ctl.disposal_task = nothing
        ctl.release_complete && (ctl.disposal_retry = false)
    end
    return ctl.release_complete
end

function _live_quiesced_locked!(vis)
    ctl = vis.content.lifecycle
    done = isnothing(ctl.task) || istaskdone(ctl.task)
    if done && !isnothing(ctl.task)
        if istaskfailed(ctl.task) && isnothing(ctl.cleanup_error)
            ctl.cleanup_error = _live_error_summary(TaskFailedException(ctl.task), :cleanup)
        end
        ctl.task = nothing
    end
    _retire_live_disposal_locked!(vis)
    quiesced = done && isnothing(ctl.timer) && !ctl.initializing && ctl.phase !== :starting
    if quiesced && ctl.phase in (:draining, :stopping, :quiesced)
        ctl.phase = :quiesced
        ctl.owner_task = nothing
    end
    return quiesced
end

function BAT.visualizer_status(vis::BATVisualizer{BATMakieVisualization})
    isnothing(vis.content) && return invoke(BAT.visualizer_status, Tuple{BATVisualizer}, vis)
    return lock(vis.content.buffer_lock) do
        ctl = vis.content.lifecycle
        quiesced = _live_quiesced_locked!(vis)
        disposed = isnothing(ctl.view) ? false : ctl.view.disposed[]
        view_pending = !isnothing(ctl.view) && ctl.view.cleanup_pending[]
        valid = !isnothing(ctl.view) && ctl.view.presentation_valid[]
        phase = quiesced && ctl.phase in (:stopping, :quiesced) ? :quiesced : ctl.phase
        (phase=phase, outcome=ctl.outcome, reason=ctl.reason,
            data_complete=ctl.data_complete, last_publication=ctl.last_publication,
            quiesced=quiesced,
            cleanup_pending=!quiesced || view_pending ||
                (ctl.disposal_requested && !ctl.release_complete),
            view_disposed=disposed, presentation_valid=valid,
            history_released=ctl.release_complete,
            error=ctl.error, cleanup_error=ctl.cleanup_error)
    end
end

function _await_live_run!(vis, start::UInt64)
    # This independent wait owns its timer; the observation timer may be closed.
    remaining = max(0.0, vis.backend.observer_timeout - _live_elapsed(start))
    # Do not format arbitrary exceptions in a deadline predicate.
    settled = () -> lock(vis.content.buffer_lock) do
        quiesced = _live_quiesced_locked!(vis)
        ctl = vis.content.lifecycle
        quiesced && (!ctl.disposal_requested || ctl.release_complete) &&
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
        ctl.startup_owner = nothing
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
    # The release coordinator also handles startup rollback without a listener.
    # Do not enroll a fallback run task after release has started.
    needs_disposal = lock(vis.content.buffer_lock) do
        ctl.disposal_requested || (isnothing(ctl.task) && ctl.phase === :stopping)
    end
    if needs_disposal
        _request_view_disposal!(vis)
        _schedule_live_view_cleanup!(vis)
    end
    (ctl.task === current_task() || ctl.disposal_task === current_task()) && return nothing
    try
        if _await_live_run!(vis, start) === :timed_out
            _fail_visualizer!(vis, ErrorException("Visualization shutdown exceeded observer_timeout"), Any[], :shutdown_timeout)
            _close_live_timer!(vis)
            _request_view_disposal!(vis)
            _schedule_live_view_cleanup!(vis)
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

# Readiness events differ from repeated automatic disposal requests. Consume
# the latch before an attempt; never erase a request received during cleanup.
function _live_release_event!(vis)
    lock(vis.content.buffer_lock) do
        Base.disable_sigint() do
            ctl = vis.content.lifecycle
            if ctl.disposal_requested && !ctl.release_complete
                ctl.disposal_retry = true
                notify(vis.content.buffer_cond; all=true)
            end
        end
    end
    _schedule_live_view_cleanup!(vis)
    return nothing
end

function _finish_live_initialization!(vis)
    changed = lock(vis.content.buffer_lock) do
        Base.disable_sigint() do
            ctl = vis.content.lifecycle
            changed = ctl.initializing
            ctl.initializing = false
            if changed && ctl.disposal_requested
                ctl.disposal_retry = true
                notify(vis.content.buffer_cond; all=true)
            end
            changed
        end
    end
    changed && _schedule_live_view_cleanup!(vis)
    return nothing
end

# Isolate graph-bearing task locals from the coordinator's sleeping/returning
# frame. The listener may have failed before entering its own catch/finally.
function _join_live_listener_for_release!(vis)
    task = lock(() -> vis.content.lifecycle.task, vis.content.buffer_lock)
    isnothing(task) && return nothing
    task === current_task() && error("A live listener cannot join itself")
    try
        wait(task)
    catch err
        if err isa TaskFailedException && getfield(err, :task) === task && istaskdone(task)
            _record_live_cleanup_error!(vis, err, catch_backtrace())
        else
            rethrow()
        end
    end
    lock(vis.content.buffer_lock) do
        vis.content.lifecycle.task === task && (vis.content.lifecycle.task = nothing)
    end
    return nothing
end

function _try_release_live_history!(vis)
    try
        _join_live_listener_for_release!(vis)
        # Also owned here: no-listener startup and retry of failed finalization.
        _close_live_timer!(vis)
        _release_live_transport!(vis)
        _cleanup_live_view!(vis) || return false
        return lock(vis.content.buffer_lock) do
            Base.disable_sigint() do
                ctl = vis.content.lifecycle
                view = ctl.view
                (ctl.disposal_requested && !ctl.initializing && isnothing(ctl.task) &&
                    isnothing(ctl.timer)) || return false
                if !isnothing(view)
                    (view.disposed[] && !view.cleanup_pending[] && !view.cleanup_running[] &&
                        !view.cleanup_blocked[] && view.depth == 0 &&
                        !view.updating && isempty(view.pending) &&
                        isempty(view.subscriptions) && isnothing(view.figure)) || return false
                end
                vis.content.graph[] = nothing
                ctl.owner_task = nothing
                ctl.phase = :quiesced
                ctl.release_committed = true
                return true
            end
        end
    catch err
        _record_live_cleanup_error!(vis, err, catch_backtrace())
        return false
    end
end

function _coordinate_live_disposal!(vis)
    while true
        lock(vis.content.buffer_lock) do
            ctl = vis.content.lifecycle
            while !ctl.disposal_retry || ctl.initializing
                wait(vis.content.buffer_cond)
            end
            ctl.disposal_retry = false
        end
        _try_release_live_history!(vis) && return nothing
        # Stay owned on a failed/busy attempt. A one-shot task can lose the
        # last readiness event while unwinding; a timer would retry-spin faults.
    end
end

function _schedule_live_view_cleanup!(vis; retry::Bool=false)
    ctl = vis.content.lifecycle
    # A wrong-thread callback can latch a request and wake an existing owner,
    # but it cannot start a GUI cleanup task on the wrong thread.
    ctl.owner_thread in (0, Threads.threadid()) || return nothing
    try
        lock(vis.content.buffer_lock) do
            Base.disable_sigint() do
                _retire_live_disposal_locked!(vis)
                (ctl.disposal_requested && !ctl.release_complete) || return nothing
                if retry && ctl.disposal_task !== current_task()
                    ctl.disposal_retry = true
                    notify(vis.content.buffer_cond; all=true)
                end
                if isnothing(ctl.disposal_task) && ctl.disposal_retry
                    _with_live_task_rng(ctl) do
                        task = @task _coordinate_live_disposal!(vis)
                        ctl.disposal_task = task
                        # A scheduling exception cannot justify dropping a task
                        # which may already be queued but not yet started.
                        schedule(task)
                    end
                end
            end
        end
    catch err
        _record_live_cleanup_error!(vis, err, catch_backtrace())
    end
    return nothing
end

function BAT.dispose_visualizer!(vis::BATVisualizer{BATMakieVisualization})
    isnothing(vis.content) && return nothing
    ctl = vis.content.lifecycle
    ctl.owner_thread in (0, Threads.threadid()) || throw(ArgumentError("Dispose a live view on its owning thread"))
    BAT.request_visualizer_stop!(vis; reason=:disposed)
    _request_view_disposal!(vis)
    _schedule_live_view_cleanup!(vis; retry=true)
    return nothing
end
