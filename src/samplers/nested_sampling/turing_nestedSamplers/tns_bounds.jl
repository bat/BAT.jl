export TNS_NoBounds
export TNS_EllipsoidBound
export TNS_MultiEllipsoidBound

"The subtypes of TNS_Bound discribe the volume represent by the live-points."
abstract type TNS_Bound end

"No bounds is equivalent to the volume from the unit Cube."
struct TNS_NoBounds <: TNS_Bound end

"Bounds are n-dimensional ellipsoids."
struct TNS_EllipsoidBound <: TNS_Bound end

"For the bounds multiple ellipsoids are used in an optimal clustering."
struct TNS_MultiEllipsoidBound <: TNS_Bound end


function TNS_Bounding(bound::TNS_NoBounds)
    return Bounds.NoBounds
end

function TNS_Bounding(bound::TNS_EllipsoidBound)
    return Bounds.Ellipsoid
end

function TNS_Bounding(bound::TNS_MultiEllipsoidBound)
    return Bounds.MultiEllipsoid
end

function TNS_Bounding(bound::TNS_Bound) # If nothing ist choosen
    return Bounds.MultiEllipsoid        # the bounds are MultiEllipsoid
end
