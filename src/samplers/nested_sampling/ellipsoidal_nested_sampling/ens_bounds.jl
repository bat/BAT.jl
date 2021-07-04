# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    abstract type ENS_Bound 

Abstract type for the bounds of the sampling region used by EllipsoidalNestedSampling.
"""
abstract type ENS_Bound end

"""
    struct ENS_NoBounds <: ENS_Bound

No bounds means that the whole volume from the unit Cube is used to find new points.
"""
struct ENS_NoBounds <: ENS_Bound end
export ENS_NoBounds

"""
    struct ENS_EllipsoidBound <: ENS_Bound

Ellipsoid bound means that a n-dimensional ellipsoid limits the sampling volume.
"""
struct ENS_EllipsoidBound <: ENS_Bound end
export ENS_EllipsoidBound

"""
    struct ENS_MultiEllipsoidBound <: ENS_Bound

Multi ellipsoid bound means that there are multiple elliposid in an optimal clustering are used to limit the sampling volume.
"""
struct ENS_MultiEllipsoidBound <: ENS_Bound end
export ENS_MultiEllipsoidBound


function ENS_Bounding(bound::ENS_NoBounds)
    return Bounds.NoBounds
end

function ENS_Bounding(bound::ENS_EllipsoidBound)
    return Bounds.Ellipsoid
end

function ENS_Bounding(bound::ENS_MultiEllipsoidBound)
    return Bounds.MultiEllipsoid
end

function ENS_Bounding(bound::ENS_Bound) # If nothing ist choosen
    return Bounds.MultiEllipsoid        # the bounds are MultiEllipsoid
end
