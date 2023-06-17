# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    abstract type ENSBound 

*Experimental feature, not part of stable public API.*

Abstract type for the bounds of the sampling region used by EllipsoidalNestedSampling.
"""
abstract type ENSBound end

"""
    struct BAT.ENSNoBounds <: ENSBound

*Experimental feature, not part of stable public API.*

No bounds means that the whole volume from the unit Cube is used to find new points.
"""
struct ENSNoBounds <: ENSBound end

"""
    struct BAT.ENSEllipsoidBound <: ENSBound

*Experimental feature, not part of stable public API.*

Ellipsoid bound means that a n-dimensional ellipsoid limits the sampling volume.
"""
struct ENSEllipsoidBound <: ENSBound end

"""
    struct BAT.ENSMultiEllipsoidBound <: ENSBound

*Experimental feature, not part of stable public API.*

Multi ellipsoid bound means that there are multiple elliposid in an optimal clustering are used to limit the sampling volume.
"""
struct ENSMultiEllipsoidBound <: ENSBound end




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



"""
    struct EllipsoidalNestedSampling <: AbstractSamplingAlgorithm

*Experimental feature, not part of stable public API.*

Uses the julia package
[NestedSamplers.jl](https://github.com/TuringLang/NestedSamplers.jl) to use nested sampling algorithm.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)


!!! note

    This functionality is only available when the
    [NestedSamplers.jl](https://github.com/TuringLang/NestedSamplers.jl) package 
    is loaded (e.g. via
    `import`).
"""
@with_kw struct EllipsoidalNestedSampling{TR<:AbstractTransformTarget} <: AbstractSamplingAlgorithm
    trafo::TR = (pkgext(Val(:NestedSamplers)); PriorToUniform())

    "Number of live-points."
    num_live_points::Int = 1000

    "Volume around the live-points."
    bound::ENSBound = ENSEllipsoidBound()

    "Algorithm used to choose new live-points."
    proposal::ENSProposal = ENSAutoProposal()
    
    "Scale factor for the volume."
    enlarge::Float64 = 1.25
    
    # "Not sure about how this works yet."
    # update_interval::Float64 =
    
    "Number of iterations before the first bound will be fit."
    min_ncall::Int = 2*num_live_points
    
    "Efficiency before fitting the first bound."
    min_eff::Float64 = 0.1

    # "The following four are the possible convergence criteria to end the algorithm."
    dlogz::Float64 = 0.01
    max_iters = Inf
    max_ncalls = 10^7
    maxlogl = Inf
end
export EllipsoidalNestedSampling
