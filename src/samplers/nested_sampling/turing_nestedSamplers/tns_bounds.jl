export NoBounds
export EllipsoidBound
export MultiEllipsoidBound

abstract type TNS_Bound end

struct NoBounds <: TNS_Bound end

struct EllipsoidBound <: TNS_Bound end

struct MultiEllipsoidBound <: TNS_Bound end


function TNS_Bounding(bound::NoBounds)
    return Bounds.NoBounds
end

function TNS_Bounding(bound::EllipsoidBound)
    return Bounds.Ellipsoid
end

function TNS_Bounding(bound::MultiEllipsoidBound)
    return Bounds.MultiEllipsoid
end

function TNS_Bounding(bound::TNS_Bound) # If nothing ist choosen
    return Bounds.MultiEllipsoid        # the bounds are MultiEllipsoid
end