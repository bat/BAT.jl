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
