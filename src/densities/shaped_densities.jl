# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct DensityWithShape{D<:AbstractMeasureOrDensity,S<:AbstractValueShape} <: AbstractMeasureOrDensity
    density::D
    shape::S
end 

Base.parent(density::DensityWithShape) = density.density

var_bounds(density::DensityWithShape) = var_bounds(parent(density))

ValueShapes.varshape(density::DensityWithShape) = density.shape

@inline DensityInterface.DensityKind(x::DensityWithShape) = DensityKind(x.density)

DensityInterface.logdensityof(density::DensityWithShape, v::Any) = logdensityof(parent(density), v)


struct ReshapedDensity{D<:AbstractMeasureOrDensity,S<:AbstractValueShape} <: AbstractMeasureOrDensity
    density::D
    shape::S
end 

Base.parent(density::ReshapedDensity) = density.density

var_bounds(density::ReshapedDensity) = var_bounds(density.density)

@inline DensityInterface.DensityKind(x::ReshapedDensity) = DensityKind(x.density)

ValueShapes.varshape(density::ReshapedDensity) = density.shape


function _reshapeddensity_variate(density::ReshapedDensity, v::Any)
    vs = varshape(density)
    orig_vs = varshape(parent(density))
    reshape_variate(orig_vs, vs, v)
end

function DensityInterface.logdensityof(density::ReshapedDensity, v::Any)
    v_reshaped = _reshapeddensity_variate(density, v)
    logdensityof(parent(density), v_reshaped)
end

function checked_logdensityof(density::ReshapedDensity, v::Any)
    v_reshaped = _reshapeddensity_variate(density, v)
    checked_logdensityof(parent(density), v_reshaped)
end


ValueShapes.unshaped(density::AbstractMeasureOrDensity) = _unshaped_density(density, varshape(density))

_unshaped_density(density::AbstractMeasureOrDensity, ::ArrayShape{<:Real,1}) = density
_unshaped_density(density::AbstractMeasureOrDensity, ::AbstractValueShape) = ReshapedDensity(density, ArrayShape{Real}(totalndof(density)))


(shape::AbstractValueShape)(density::AbstractMeasureOrDensity) = _reshaped_density(density, shape, varshape(density))

function _reshaped_density(density::AbstractMeasureOrDensity, new_shape::VS, orig_shape::VS) where {VS<:AbstractValueShape}
    if orig_shape == new_shape
        density
    else
        throw(ArgumentError("Value shapes are incompatible"))
    end
end

function _reshaped_density(density::AbstractMeasureOrDensity, new_shape::AbstractValueShape, orig_shape::AbstractValueShape)
    if totalndof(orig_shape) == totalndof(new_shape)
        ReshapedDensity(density, new_shape)
    else
        throw(ArgumentError("Value shapes are incompatible"))
    end
end
