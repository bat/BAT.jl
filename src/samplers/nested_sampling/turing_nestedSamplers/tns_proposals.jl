export Uniformly
export RandomWalk

abstract type TNS_Proposal end


struct Uniformly <: TNS_Proposal end

struct AutoProposal <: TNS_Proposal end

@with_kw struct RandomWalk <: TNS_Proposal
    ratio::Float64 = 0.5
    walks::Int64 = 25
    scale::Float64 = 1.0 # >= 0
end

@with_kw struct RStaggering <: TNS_Proposal
    ratio::Float64 = 0.5
    walks::Int64 = 25
    scale::Float64 = 1.0 # >= 0
end

@with_kw struct Slicing <: TNS_Proposal
    slices::Int64 = 5
    scale::Float64 = 1.0 # >= 0
end

@with_kw struct RSlicing <: TNS_Proposal
    slices::Int64 = 5
    scale::Float64 = 1.0 # >= 0
end

function TNS_prop(prop::Uniformly)
    return Proposals.Uniform()
end

function TNS_prop(prop::AutoProposal)
    # :auto declaration: ndims < 10: Proposals.Uniform, 10 ≤ ndims ≤ 20: Proposals.RWalk, ndims > 20: Proposals.Slice
    return :auto
end

function TNS_prop(prop::RandomWalk)
    return Proposals.RWalk(;prop.ratio, prop.walks, prop.scale)
end

function TNS_prop(prop::RStaggering)
    Proposals.RStagger(;prop.ratio, prop.walks, prop.scale)
end

function TNS_prop(prop::Slicing)
    return Proposals.Slice(;prop.slices, prop.scale)
end

function TNS_prop(prop::RSlicing)
    return Proposals.RSlice(;prop.slices, prop.scale)
end

function TNS_prop(prop::TNS_Proposal) # if nothing is choosen
    return :auto
end
