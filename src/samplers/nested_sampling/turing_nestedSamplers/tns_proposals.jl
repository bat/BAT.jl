export TNS_Uniformly 
export TNS_RandomWalk
export TNS_Slice
export TNS_AutoProposal

"The subtypes of TNS_Proposal discribe the algorithm to propose a new live-points."
abstract type TNS_Proposal end

"New live point is found by uniformly sampling from the bounding volume."
struct TNS_Uniformly <: TNS_Proposal end

"Choose the proposal depending from the dimensions. Declaration: ndims < 10: Proposals.Uniform, 10 ≤ ndims ≤ 20: Proposals.RWalk, ndims > 20: Proposals.Slice"
struct TNS_AutoProposal <: TNS_Proposal end

"New live point is found by a random walk away from existing point."
@with_kw struct TNS_RandomWalk <: TNS_Proposal
    ratio::Float64 = 0.5
    walks::Int64 = 25
    scale::Float64 = 1.0 # >= 0
end

"New live point is found by a serie of random slices from an existing live-point."
@with_kw struct TNS_Slice <: TNS_Proposal
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

function TNS_prop(prop::TNS_Slice)
    return Proposals.Slice(;prop.slices, prop.scale)
end

function TNS_prop(prop::TNS_Proposal) # if nothing is choosen
    return :auto
end
