export ENS_Uniformly 
export ENS_RandomWalk
export ENS_Slice
export ENS_AutoProposal

"The subtypes of ENS_Proposal discribe the algorithm to propose a new live-points."
abstract type ENS_Proposal end

"New live point is found by uniformly sampling from the bounding volume."
struct ENS_Uniformly <: ENS_Proposal end

"Choose the proposal depending from the dimensions. Declaration: ndims < 10: Proposals.Uniform, 10 ≤ ndims ≤ 20: Proposals.RWalk, ndims > 20: Proposals.Slice"
struct ENS_AutoProposal <: ENS_Proposal end

"New live point is found by a random walk away from existing point."
@with_kw struct ENS_RandomWalk <: ENS_Proposal
    ratio::Float64 = 0.5
    walks::Int64 = 25
    scale::Float64 = 1.0 # >= 0
end

"New live point is found by a serie of random slices from an existing live-point."
@with_kw struct ENS_Slice <: ENS_Proposal
    slices::Int64 = 5
    scale::Float64 = 1.0 # >= 0
end

function ENS_prop(prop::ENS_Uniformly)
    return Proposals.Uniform()
end

function ENS_prop(prop::ENS_AutoProposal)
    return :auto     # :auto declaration: ndims < 10: Proposals.Uniform, 10 ≤ ndims ≤ 20: Proposals.RWalk, ndims > 20: Proposals.Slice
end

function ENS_prop(prop::ENS_RandomWalk)
    return Proposals.RWalk(;prop.ratio, prop.walks, prop.scale)
end

function ENS_prop(prop::ENS_Slice)
    return Proposals.Slice(;prop.slices, prop.scale)
end

function ENS_prop(prop::ENS_Proposal) # if nothing is choosen
    return :auto
end
