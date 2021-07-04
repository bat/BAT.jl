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

function ENSBounding(bound::ENSNoBounds)
    return Bounds.NoBounds
end

function ENSBounding(bound::ENSEllipsoidBound)
    return Bounds.Ellipsoid
end

function ENSBounding(bound::ENSMultiEllipsoidBound)
    return Bounds.MultiEllipsoid
end

function ENSBounding(bound::ENSBound) # If nothing ist choosen
    return Bounds.MultiEllipsoid        # the bounds are MultiEllipsoid
end
