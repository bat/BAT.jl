export NoBounds
export EllipsoidBound
export MultiEllipsoidBound

abstract type NSBound end

struct NoBounds <: NSBound end

struct EllipsoidBound <: NSBound end

struct MultiEllipsoidBound <: NSBound end


function NSbounding(bound::NoBounds)
    return Bounds.NoBounds
end

function NSbounding(bound::EllipsoidBound)
    return Bounds.Ellipsoid
end

function NSbounding(bound::MultiEllipsoidBound)
    return Bounds.MultiEllipsoid
end

function NSbounding(bound::NSBound) # wenn nichts ausgewählt wird
    print("auto = Ellipse")
    return Bounds.Ellipsoid         # Ist ellipse Standard
end