# Here are the definitions of the NestedSamplers bounds, which discribe the volume represent by the live-points
export TNS_NoBounds
export TNS_EllipsoidBound
export TNS_MultiEllipsoidBound

abstract type TNS_Bound end


struct TNS_NoBounds <: TNS_Bound end

struct TNS_EllipsoidBound <: TNS_Bound end

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
