# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    abstract type ENSProposal

*Experimental feature, not part of stable public API.*

Abstract type for the algorithms to propose new live points used used by EllipsoidalNestedSampling.
"""
abstract type ENSProposal end

"""
    struct BAT.ENSUniformly <: ENSProposal

*Experimental feature, not part of stable public API.*

Each point in the bounding volume has an uniform chance to be proposed as a new live point.
"""
struct ENSUniformly <: ENSProposal end

"""
    struct BAT.ENSAutoProposal <: ENSProposal

*Experimental feature, not part of stable public API.*

Choose the proposal depending from the number of dimensions:
     ndims < 10: Proposals.Uniform, 
     10 ≤ ndims ≤ 20: Proposals.RWalk, 
     ndims > 20: Proposals.Slice
"""
struct ENSAutoProposal <: ENSProposal end

"""
    struct BAT.ENSRandomWalk <: ENSProposal

*Experimental feature, not part of stable public API.*

New live point is proposed by using a random walk away from an existing live point.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:
$(TYPEDFIELDS)
"""
@with_kw struct ENSRandomWalk <: ENSProposal

    "Acceptance ratio for the random walk."
    ratio::Float64 = 0.5

    "Minimum number of random walk steps."
    walks::Int64 = 25

    "Scale of the proposal distribution."
    scale::Float64 = 1.0 # >= 0
end

"""
    struct BAT.ENSSlice <: ENSProposal

*Experimental feature, not part of stable public API.*

New live point is proposed by a serie of random slices from an existing live-point.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct ENSSlice <: ENSProposal

    "Minimum number of slices"
    slices::Int64 = 5

    "Scale of the proposal distribution."
    scale::Float64 = 1.0 # >= 0
end

function ENSprop(prop::ENSUniformly)
    return Proposals.Uniform()
end

function ENSprop(prop::ENSAutoProposal)
    return :auto     # :auto declaration: ndims < 10: Proposals.Uniform, 10 ≤ ndims ≤ 20: Proposals.RWalk, ndims > 20: Proposals.Slice
end

function ENSprop(prop::ENSRandomWalk)
    return Proposals.RWalk(;ratio=prop.ratio, walks=prop.walks, scale=prop.scale)
end

function ENSprop(prop::ENSSlice)
    return Proposals.Slice(;slices=prop.slices, scale=prop.scale)
end

function ENSprop(prop::ENSProposal) # if nothing is choosen
    return :auto
end
