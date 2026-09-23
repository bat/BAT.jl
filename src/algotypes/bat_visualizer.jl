# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    abstract type BATVisBackend

*Experimental feature, not yet part of stable public API.*

Abstract supertype of sampling-visualization backend configurations
(e.g. `BATMakieVisualization`).
"""
abstract type BATVisBackend end

struct BATNoVisBackend <: BATVisBackend end

"""
    struct BATVisualizer{B<:BATVisBackend}

*Experimental feature, not yet part of stable public API.*

Carries a visualization backend configuration and, once sampling starts, the
live visualization state. Pass via `BATContext(visualizer = ...)`. A used
visualizer is single-use -- create a fresh one per sampling run.

Constructors:

* ```BATVisualizer()```: no visualization
* ```BATVisualizer(backend::BATVisBackend)```: e.g. `BATVisualizer(BATMakieVisualization())`
"""
struct BATVisualizer{B<:BATVisBackend}
    backend::B
    content::Any
end
export BATVisualizer

function BATVisualizer()
    return BATVisualizer(BATNoVisBackend(), nothing)
end

# Package whose extension a backend needs, or `nothing` for backends that work
# in core alone. Extended by each extension-backed backend's own definition.
_vis_required_pkg(::BATVisBackend) = nothing

# The generic backend constructor promised by the docstring above. A backend
# whose implementation lives in a package extension is rejected here instead of
# yielding a visualizer that can never draw: without the extension loaded its
# `init_visualizer!` is the no-op fallback, so sampling would silently run with
# no visualization at all. Extensions define a more specific method that builds
# their live content, which takes precedence over this one.
function BATVisualizer(backend::BATVisBackend)
    pkg = _vis_required_pkg(backend)
    isnothing(pkg) || pkgext(Val(pkg))
    return BATVisualizer(backend, nothing)
end

# Copies the backend config but drops the live `content`: provenance deepcopies
# of a used visualizer would otherwise hit "deepcopy of Modules not supported"
# via Makie observable listeners. The copy keeps the concrete type (Base's
# generic struct deepcopy type-asserts it) with content stripped.
function Base.deepcopy_internal(v::BATVisualizer, stackdict::IdDict)
    haskey(stackdict, v) && return stackdict[v]
    v_copy = BATVisualizer(Base.deepcopy_internal(v.backend, stackdict), nothing)
    stackdict[v] = v_copy
    return v_copy
end

function init_visualizer!(vis::BATVisualizer; kwargs...) end

function register_state_for_vis!() end

function update_visualizer!(vis::BATVisualizer; kwargs...)
    update_visualizer_impl!(vis; kwargs...)
end

function update_visualizer_impl!(vis::BATVisualizer; kwargs...) end

# Internal lifecycle hooks. Backends claim ownership in prepare_visualizer_init!
# and roll back their own acquisition if preparation throws. A successful return
# transfers cleanup responsibility to the sampling owner.
request_visualizer_stop!(::BATVisualizer; reason = :disabled, exception = nothing) = nothing
finish_visualizer!(::BATVisualizer; success::Bool = true) = nothing

"""
    BAT.dispose_visualizer!(vis::BATVisualizer)

*Experimental, non-exported API.* Disable observation and dispose of resources
owned by this visualization session. Repeated disposal is harmless. Disposal
does not cancel sampling, and does not affect other figures. A backend may report
pending cleanup while an already admitted view operation is still running.
"""
dispose_visualizer!(::BATVisualizer) = nothing

"""
    BAT.visualizer_figure(vis::BATVisualizer)

*Experimental, non-exported API.* Return the retained figure of a successfully
completed visualization, or `nothing` when unavailable. Backends without a
retained figure, including the no-op backend, return `nothing`.

For Makie, observation must be complete and its tasks settled, with a valid idle
view and no pending disposal. Active, failed, disposed or busy sessions return
`nothing`, as do calls reentered from view callbacks. This query neither waits
for completion nor constructs, displays or renders a figure.

Call on the thread that created the live view. A different thread raises
`ArgumentError` when the session can be inspected; contention on its control
guard takes precedence and returns `nothing` without waiting.

The result is a borrowed mutable figure, not a copy or ownership transfer.
Coordinate subsequent Makie operations with callbacks and disposal on the owning
thread. Later closing or `dispose_visualizer!` can empty that same figure; the
query does not protect its later use. Inspect [`visualizer_status`](@ref) for
lifecycle state and diagnostics.
"""
visualizer_figure(::BATVisualizer) = nothing

"""
    BAT.visualizer_status(vis::BATVisualizer)

*Experimental, non-exported API.* Return a read-only lifecycle snapshot without
accessing the renderer. `phase`, `outcome` and `reason` describe the run;
`data_complete` and `last_publication` describe confirmed observation history.
`quiesced` records actual run-resource completion, independently of
`cleanup_pending`, `view_disposed` and `presentation_valid`. `error` records the
first failure and its phase; `cleanup_error` is a secondary cleanup diagnostic.
A no-op backend reports an inactive, quiesced session. Backends must not expose
mutable internal storage through this snapshot.
"""
visualizer_status(::BATVisualizer) = (
    phase = :inactive, outcome = :inactive, reason = :none,
    data_complete = false, last_publication = 0, quiesced = true,
    cleanup_pending = false, view_disposed = false, presentation_valid = false,
    error = nothing, cleanup_error = nothing,
)

# These guards contain backend cleanup faults, including broken diagnostics.
# They are deliberately outside computational operations and public callbacks.
function _visualizer_cleanup_diagnostic(operation, exception)
    try
        @warn "Visualization cleanup failed" operation exception
    catch
        # A logger failure must not replace a computational/startup exception or
        # prevent another backend in a producer group from being notified.
    end
    return nothing
end

function _safe_request_visualizer_stop!(vis::BATVisualizer; reason = :disabled, exception = nothing)
    try
        request_visualizer_stop!(vis; reason, exception)
    catch err
        _visualizer_cleanup_diagnostic(:request_stop, (err, catch_backtrace()))
    end
    return nothing
end

function _safe_finish_visualizer!(vis::BATVisualizer; success::Bool = true)
    try
        finish_visualizer!(vis; success)
    catch err
        # Intentional interruption of the owner is primary on a normal exit;
        # when unwinding, preserve the already active computational exception.
        err isa InterruptException && success && rethrow()
        _safe_request_visualizer_stop!(vis;
            reason=:cleanup_failed, exception=CapturedException(err, catch_backtrace()))
        _visualizer_cleanup_diagnostic(:finish, (err, catch_backtrace()))
    end
    return nothing
end

function _safe_clear_visualizer_init!(vis::BATVisualizer; suppress_interrupt::Bool = false)
    try
        clear_visualizer_init!(vis)
    catch err
        err isa InterruptException && !suppress_interrupt && rethrow()
        _visualizer_cleanup_diagnostic(:clear_initialization, (err, catch_backtrace()))
    end
    return nothing
end

# Optional initialization timing, implemented by visualization extensions. These
# hooks run only at iterator boundaries; they never emit public callback events.
prepare_visualizer_init!(::BATVisualizer, algorithm) = nothing
capture_visualizer_init(::BATVisualizer) = false
clear_visualizer_init!(::BATVisualizer) = nothing
reset_visualizer_init!(::BATVisualizer, chain, walkers) = nothing
begin_visualizer_init_segment(::BATVisualizer, chain, output) = nothing
end_visualizer_init_segment!(::BATVisualizer, before, chain, output) = nothing
