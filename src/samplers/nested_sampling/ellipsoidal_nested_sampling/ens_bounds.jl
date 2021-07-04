export ENS_NoBounds
export ENS_EllipsoidBound
export ENS_MultiEllipsoidBound

"The subtypes of ENS_Bound discribe the volume represent by the live-points."
abstract type ENS_Bound end

"No bounds is equivalent to the volume from the unit Cube."
struct ENS_NoBounds <: ENS_Bound end

"Bounds are n-dimensional ellipsoids."
struct ENS_EllipsoidBound <: ENS_Bound end

"For the bounds multiple ellipsoids are used in an optimal clustering."
struct ENS_MultiEllipsoidBound <: ENS_Bound end


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
