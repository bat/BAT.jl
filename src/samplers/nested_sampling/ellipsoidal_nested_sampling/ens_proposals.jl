# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    abstract type ENS_Proposal

Abstract type for the algorithms to propose new live points used used by EllipsoidalNestedSampling.
"""
abstract type ENS_Proposal end

"""
    struct ENS_Uniformly <: ENS_Proposal

Each point in the bounding volume has an uniform chance to be proposed as a new live point.
"""
struct ENS_Uniformly <: ENS_Proposal end
export ENS_Uniformly 

"""
    struct ENS_AutoProposal <: ENS_Proposal

Choose the proposal depending from the number of dimensions:
     ndims < 10: Proposals.Uniform, 
     10 ≤ ndims ≤ 20: Proposals.RWalk, 
     ndims > 20: Proposals.Slice
"""
struct ENS_AutoProposal <: ENS_Proposal end
export ENS_AutoProposal

"""
    struct ENS_RandomWalk <: ENS_Proposal

New live point is proposed by using a random walk away from an existing live point.
"""
@with_kw struct ENS_RandomWalk <: ENS_Proposal
    ratio::Float64 = 0.5
    walks::Int64 = 25
    scale::Float64 = 1.0 # >= 0
end
export ENS_RandomWalk

"""
    struct ENS_Slice <: ENS_Proposal

New live point is proposed by a serie of random slices from an existing live-point.
"""
@with_kw struct ENS_Slice <: ENS_Proposal
    slices::Int64 = 5
    scale::Float64 = 1.0 # >= 0
end
export ENS_Slice

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
