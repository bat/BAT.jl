# Here are the definitions of the NestedSamplers proposals, which discribe the algorithm to choose new live-points
export TNS_Uniformly
export TNS_RandomWalk
export TNS_RStaggering
export TNS_Slice
export TNS_RSlice
export TNS_AutoProposal

abstract type TNS_Proposal end


struct TNS_Uniformly <: TNS_Proposal end

struct TNS_AutoProposal <: TNS_Proposal end


@with_kw struct TNS_RandomWalk <: TNS_Proposal
    ratio::Float64 = 0.5
    walks::Int64 = 25
    scale::Float64 = 1.0 # >= 0
end

@with_kw struct TNS_RStaggering <: TNS_Proposal
    ratio::Float64 = 0.5
    walks::Int64 = 25
    scale::Float64 = 1.0 # >= 0
end

@with_kw struct TNS_Slice <: TNS_Proposal
    slices::Int64 = 5
    scale::Float64 = 1.0 # >= 0
end

@with_kw struct TNS_RSlice <: TNS_Proposal
    slices::Int64 = 5
    scale::Float64 = 1.0 # >= 0
end

function TNS_prop(prop::TNS_Uniformly)
    return Proposals.Uniform()
end

function TNS_prop(prop::TNS_AutoProposal)
    return :auto     # :auto declaration: ndims < 10: Proposals.Uniform, 10 ≤ ndims ≤ 20: Proposals.RWalk, ndims > 20: Proposals.Slice
end

function TNS_prop(prop::TNS_RandomWalk)
    return Proposals.RWalk(;prop.ratio, prop.walks, prop.scale)
end

function TNS_prop(prop::TNS_RStaggering)
    Proposals.RStagger(;prop.ratio, prop.walks, prop.scale)
end

function TNS_prop(prop::TNS_Slice)
    return Proposals.Slice(;prop.slices, prop.scale)
end

function TNS_prop(prop::TNS_RSlice)
    return Proposals.RSlice(;prop.slices, prop.scale)
end

function TNS_prop(prop::TNS_Proposal) # if nothing is choosen
    return :auto
end
