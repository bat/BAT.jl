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

# Optional initialization timing, implemented by visualization extensions. These
# hooks run only at iterator boundaries; they never emit public callback events.
prepare_visualizer_init!(::BATVisualizer, algorithm) = nothing
capture_visualizer_init(::BATVisualizer) = false
clear_visualizer_init!(::BATVisualizer) = nothing
reset_visualizer_init!(::BATVisualizer, chain, walkers) = nothing
begin_visualizer_init_segment(::BATVisualizer, chain, output) = nothing
end_visualizer_init_segment!(::BATVisualizer, before, chain, output) = nothing
