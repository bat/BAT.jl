# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct DensityWithShape{D<:AbstractDensity,S<:AbstractValueShape} <: AbstractDensity
    density::D
    shape::S
end 

Base.parent(density::DensityWithShape) = density.density

var_bounds(density::DensityWithShape) = var_bounds(parent(density))

ValueShapes.varshape(density::DensityWithShape) = density.shape

DensityInterface.logdensityof(density::DensityWithShape, v::Any) = logdensityof(parent(density), v)


struct ReshapedDensity{D<:AbstractDensity,S<:AbstractValueShape} <: AbstractDensity
    density::D
    shape::S
end 

Base.parent(density::ReshapedDensity) = density.density

var_bounds(density::ReshapedDensity) = var_bounds(density.density)

ValueShapes.varshape(density::ReshapedDensity) = density.shape


function _reshapeddensity_variate(density::ReshapedDensity, v::Any)
    orig_varshape = varshape(parent(density))
    reshape_variate(orig_varshape, v)
end

function DensityInterface.logdensityof(density::ReshapedDensity, v::Any)
    v_reshaped = _reshapeddensity_variate(density, v)
    logdensityof(parent(density), v_reshaped)
end

function checked_logdensityof(density::ReshapedDensity, v::Any)
    v_reshaped = _reshapeddensity_variate(density, v)
    checked_logdensityof(parent(density), v_reshaped)
end


ValueShapes.unshaped(density::AbstractDensity) = _unshaped_density(density, varshape(density))

_unshaped_density(density::AbstractDensity, ::ArrayShape{<:Real,1}) = density
_unshaped_density(density::AbstractDensity, ::AbstractValueShape) = ReshapedDensity(density, ArrayShape{Real}(totalndof(density)))


(shape::AbstractValueShape)(density::AbstractDensity) = _reshaped_density(density, shape, varshape(density))

function _reshaped_density(density::AbstractDensity, new_shape::VS, orig_shape::VS) where {VS<:AbstractValueShape}
    if orig_shape == new_shape
        density
    else
        throw(ArgumentError("Value shapes are incompatible"))
    end
end

function _reshaped_density(density::AbstractDensity, new_shape::AbstractValueShape, orig_shape::AbstractValueShape)
    if totalndof(orig_shape) == totalndof(new_shape)
        ReshapedDensity(density, new_shape)
    else
        throw(ArgumentError("Value shapes are incompatible"))
    end
end
