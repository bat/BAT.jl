# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Rendering has its own guard. Never acquire it while holding buffer_lock:
# producers and the control timer must be able to stop an unavailable view.
mutable struct _LiveView
    lock::ReentrantLock
    figure::Union{Nothing,Figure}
    subscriptions::Vector{Any}
    disposed::Threads.Atomic{Bool}
    cleanup_pending::Threads.Atomic{Bool}
    cleanup_running::Threads.Atomic{Bool}
    cleanup_blocked::Threads.Atomic{Bool}
    presentation_valid::Threads.Atomic{Bool}
    depth::Int
    updating::Bool
    pending::Dict{Symbol,Any}
    seen_open::Threads.Atomic{Bool}
end
_LiveView() = _LiveView(ReentrantLock(), nothing, Any[],
    Threads.Atomic{Bool}(false), Threads.Atomic{Bool}(false), Threads.Atomic{Bool}(false),
    Threads.Atomic{Bool}(false), Threads.Atomic{Bool}(true), 0, false, Dict{Symbol,Any}(), Threads.Atomic{Bool}(false))

function _live_view(vis)
    lock(vis.content.buffer_lock) do
        life = vis.content.lifecycle
        isnothing(life.view) && life.disposal_requested && return nothing
        isnothing(life.view) && (life.view = _LiveView())
        return life.view
    end
end

# A completed figure is borrowed, never detached from its session. This query
# must not create a view or turn ordinary caller misuse into an observer failure.
function BAT.visualizer_figure(vis::BATVisualizer{BATMakieVisualization})
    isnothing(vis.content) && return nothing
    control_lock = vis.content.buffer_lock
    trylock(control_lock) || return nothing
    view = try
        life = vis.content.lifecycle
        life.owner_thread in (0, Threads.threadid()) ||
            throw(ArgumentError("Retrieve a live figure on its owning thread"))
        life.view
    finally
        unlock(control_lock)
    end
    isnothing(view) && return nothing

    # Never wait for a view while holding the transport guard. Revalidate state
    # after acquiring both guards in the same order as normal view operations.
    trylock(view.lock) || return nothing
    try
        trylock(control_lock) || return nothing
        try
            life = vis.content.lifecycle
            life.view === view || return nothing
            life.disposal_requested && return nothing
            (life.outcome === :complete && life.data_complete) || return nothing
            (view.disposed[] || view.cleanup_pending[] || view.cleanup_running[] ||
                !view.presentation_valid[]) && return nothing
            # Reentrant locks alone cannot exclude a query from a view callback.
            (view.depth != 0 || view.updating || !isempty(view.pending)) && return nothing
            _live_quiesced_locked!(vis) || return nothing
            isnothing(life.disposal_task) || return nothing
            return view.figure
        finally
            unlock(control_lock)
        end
    finally
        unlock(view.lock)
    end
end

function _live_view_admitted_locked(vis, view, publication)
    (view.disposed[] || view.cleanup_pending[] || !view.presentation_valid[]) && return false
    life = vis.content.lifecycle
    life.disposal_requested && return false
    life.phase === :starting && return true
    return publication ? _live_can_publish_locked(vis) :
        (_live_can_publish_locked(vis) || life.outcome === :complete)
end

# Bool reports admission and successful completion, not the callback's value.
# Derived synchronous widget callbacks can reenter; graph updates below are
# separately coalesced so they cannot recursively publish a new generation.
function _with_live_view(f, vis; publication::Bool = false)
    view = _live_view(vis)
    isnothing(view) && return false
    exited = false
    outermost_guard = false
    starting = lock(vis.content.buffer_lock) do
        vis.content.lifecycle.phase === :starting
    end
    try
        Threads.threadid() == vis.content.lifecycle.owner_thread ||
            throw(ArgumentError("Live Makie view operations must run on their owning thread"))
        lock(view.lock)
        try
            admitted = lock(vis.content.buffer_lock) do
                _live_view_admitted_locked(vis, view, publication)
            end
            admitted || return false
            view.depth += 1
            try
                f()
            finally
                view.depth -= 1
                exited = view.depth == 0
            end
            return view.presentation_valid[] && !view.cleanup_pending[]
        finally
            outermost_guard = view.depth == 0
            unlock(view.lock)
        end
    catch err
        view.presentation_valid[] = false
        starting && rethrow()
        if err isa InterruptException
            # Interactive callbacks can run outside both the sampling owner
            # and the listener. Stop observation before propagating interruption
            # so a refused final publication cannot be reported as complete.
            BAT.request_visualizer_stop!(vis; reason = :aborted,
                exception = CapturedException(err, catch_backtrace()))
            _request_view_disposal!(vis)
            rethrow()
        end
        _fail_visualizer!(vis, err, catch_backtrace(), :view)
        _request_view_disposal!(vis)
        return false
    finally
        # Backend cleanup may wait for its render loop. Schedule it outside
        # this callback; never join a rendering task from a BAT view callback.
        if view.cleanup_pending[]
            # A pre-admission holder can block cleanup while waiting for the
            # control guard, then reject. Its exit is readiness only if a real
            # cleanup attempt was blocked; other rejected callbacks cannot
            # manufacture retries, including callbacks emitted by cleanup.
            blocked = outermost_guard && Threads.atomic_xchg!(view.cleanup_blocked, false)
            (exited || blocked) ? _live_release_event!(vis) : _schedule_live_view_cleanup!(vis)
        end
    end
end

const _LIVE_MAX_PENDING_ATTRIBUTES = 32
const _LIVE_MAX_DEFERRED_UPDATES = 32

function _live_graph_update!(vis; publication::Bool = false, kwargs...)
    return _with_live_view(vis; publication = publication) do
        view = vis.content.lifecycle.view
        if view.updating
            publication && throw(ArgumentError("Recursive live data publication is unsupported"))
            for (key, value) in kwargs
                (!haskey(view.pending, key) && length(view.pending) >= _LIVE_MAX_PENDING_ATTRIBUTES) &&
                    throw(ArgumentError("Too many deferred live view attributes"))
                view.pending[key] = value
            end
            return nothing
        end
        view.updating = true
        try
            update!(_live_graph(vis); kwargs...)
            rounds = 0
            while !isempty(view.pending)
                allowed = lock(vis.content.buffer_lock) do
                    _live_view_admitted_locked(vis, view, false)
                end
                allowed || break
                rounds += 1
                rounds <= _LIVE_MAX_DEFERRED_UPDATES ||
                    throw(ArgumentError("Live view callbacks did not settle after deferred updates"))
                attributes = copy(view.pending)
                empty!(view.pending)
                update!(_live_graph(vis); attributes...)
            end
        finally
            empty!(view.pending)
            view.updating = false
        end
    end
end

function _track_live_subscription!(vis, handle)
    view = _live_view(vis)
    if handle isa AbstractVector
        append!(view.subscriptions, handle)
    else
        push!(view.subscriptions, handle)
    end
    return handle
end

# Static callbacks belong to the figure, so empty!(fig) releases control
# subscriptions and the sample graph captured by root input callbacks.
_view_on(f, owner::Scene, observable; kwargs...) = on(f, owner, observable; kwargs...)
_view_onany(f, owner::Scene, observables...; kwargs...) = onany(f, owner, observables...; kwargs...)

# Keep the static path's callback return values and registration semantics.
# Register before the optional initial call so failed startup still owns the
# handle. onany returns one handle for each input observable.
function _view_on(f, vis, observable; update::Bool = false, kwargs...)
    isnothing(vis) && return on(f, observable; update = update, kwargs...)
    guarded = function(args...)
        result = Ref{Any}(Consume(false))
        _with_live_view(vis) do
            result[] = f(args...)
        end
        return result[]
    end
    handle = _track_live_subscription!(vis, on(guarded, observable; kwargs...))
    update && guarded(observable[])
    return handle
end

function _view_onany(f, vis, observables...; update::Bool = false, kwargs...)
    isnothing(vis) && return onany(f, observables...; update = update, kwargs...)
    guarded = function(args...)
        _with_live_view(vis) do
            f(args...)
        end
        return nothing
    end
    handles = _track_live_subscription!(vis, onany(guarded, observables...; kwargs...))
    update && guarded(map(to_value, observables)...)
    return handles
end

function _attach_live_view!(vis, fig)
    view = _live_view(vis)
    view.figure = fig
    window_open = events(fig.scene).window_open
    view.seen_open[] = window_open[]
    # This control-only callback intentionally bypasses the rendering guard:
    # close must release producers even while another task is rendering.
    handle = on(window_open) do opened
        if opened
            view.seen_open[] = true
        elseif view.seen_open[]
            BAT.request_visualizer_stop!(vis; reason = :closed)
            _request_view_disposal!(vis)
            _schedule_live_view_cleanup!(vis)
        end
        return nothing
    end
    _track_live_subscription!(vis, handle)
    return view
end

function _request_view_disposal!(vis)
    lock(vis.content.buffer_lock) do
        Base.disable_sigint() do
            ctl = vis.content.lifecycle
            if !ctl.disposal_requested
                _stop_live_locked!(vis, :disposed)
                ctl.disposal_requested = true
                ctl.disposal_retry = true
            end
            view = ctl.view
            isnothing(view) || (view.cleanup_pending[] = !view.disposed[])
            notify(vis.content.buffer_cond; all = true)
        end
    end
    return nothing
end

# GLMakie exposes its currently displayed scene on the screen. A screen can be
# reused to show another figure, so attachment in current_screens alone is not
# proof of ownership. Other backends retain their own native-window policy.
function _close_owned_live_screens!(fig)
    for screen in copy(fig.scene.current_screens)
        if nameof(parentmodule(typeof(screen))) === :GLMakie &&
                hasproperty(screen, :scene) && screen.scene === fig.scene &&
                applicable(close, screen)
            _close_gl_screen!(screen, parentmodule(typeof(screen)))
        end
    end
    return nothing
end

# Admission is nonwaiting. Backend cleanup itself may yield/block, so callers
# use the owned cleanup task, never a rendering callback or listener finalizer.
function _cleanup_live_view!(vis)
    view = vis.content.lifecycle.view
    isnothing(view) && return true
    view.disposed[] && return true
    view.cleanup_pending[] || return true
    Threads.threadid() == vis.content.lifecycle.owner_thread || return false
    admitted = false
    # Install the reset before publishing cleanup_running, including the
    # handoff out of the view guard. A failed attempt must not clear another
    # executor's flag or leave its own flag permanently latched.
    try
        # Arm before trying the guard: a holder exiting after a failed trylock
        # must wake the coordinator even when its own admission was rejected.
        view.cleanup_blocked[] = true
        trylock(view.lock) || return false
        try
            (view.depth == 0 && !view.cleanup_running[]) || return false
            admitted = true
            view.cleanup_running[] = true
            view.cleanup_blocked[] = false
        finally
            unlock(view.lock)
        end
        # cleanup_pending stays latched, rejecting every new admission. Release
        # the guard before any backend operation which can wait for callbacks.
        failed_handles = Any[]
        for handle in view.subscriptions
            try
                Makie.Observables.off(handle)
            catch err
                push!(failed_handles, handle)
                _record_live_cleanup_error!(vis, err, catch_backtrace())
            end
        end
        view.subscriptions = failed_handles
        empty!(view.pending)
        if !isnothing(view.figure)
            _close_owned_live_screens!(view.figure)
            empty!(view.figure)
            view.figure = nothing
        end
        vis.content.lifecycle.renderer_owner[] = nothing
        if isempty(failed_handles)
            view.disposed[] = true
            view.cleanup_pending[] = false
            return true
        end
        view.presentation_valid[] = false
        return false
    catch err
        view.presentation_valid[] = false
        _record_live_cleanup_error!(vis, err, catch_backtrace())
        return false
    finally
        admitted && (view.cleanup_running[] = false)
    end
end
