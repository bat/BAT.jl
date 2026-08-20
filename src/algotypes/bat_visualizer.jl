# This file is a part of BAT.jl, licensed under the MIT License (MIT).

abstract type BATVisBackend end

struct BATNoVisBackend <: BATVisBackend end

struct BATVisualizer{B<:BATVisBackend}
        backend::B
        content::Any
end
export BATVisualizer

function BATVisualizer()
        return BATVisualizer(BATNoVisBackend(), nothing)
end

# A deepcopy of a visualizer copies the backend CONFIG but deliberately drops
# the live `content`: content is the live handle of one specific run (locks,
# a listener task, and -- once a figure has been built -- a compute graph
# whose Makie observable listeners reach backend Modules, which Base.deepcopy
# outright refuses to copy). BATContext gets deepcopied in several places
# (bat_sample's result-provenance `orig_context`, MCMCChainPoolInit's dummy
# context, ...), and before this method existed, any such deepcopy CRASHED
# with the obscure "deepcopy of Modules not supported" as soon as the
# context's visualizer had completed a prior run. The copy keeps the same
# concrete type (Base's generic struct deepcopy type-asserts that) with
# content === nothing -- a Makie-backend init_visualizer! rejects such a
# stripped copy with a clear error if anyone tries to sample with one.
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
