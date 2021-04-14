# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct DensityWithShape{D<:AbstractDensity,S<:AbstractValueShape} <: AbstractDensity
    density::D
    shape::S
end 

Base.parent(density::DensityWithShape) = density.density

var_bounds(density::DensityWithShape) = var_bounds(parent(density))

ValueShapes.varshape(density::DensityWithShape) = density.shape

function eval_logval_unchecked(density::DensityWithShape, v::Any)
    eval_logval_unchecked(parent(density), v)
end



struct ReshapedDensity{D<:AbstractDensity,S<:AbstractValueShape} <: AbstractDensity
    density::D
    shape::S
end 

Base.parent(density::ReshapedDensity) = density.density

var_bounds(density::ReshapedDensity) = var_bounds(density.density)

ValueShapes.varshape(density::ReshapedDensity) = density.shape


function eval_logval(density::ReshapedDensity, v::Any, T::Type{<:Real})
    v_shaped = fixup_variate(varshape(density), v)
    orig_density = parent(density)
    orig_varshape = varshape(orig_density)
    v_reshaped = reshape_variate(orig_varshape, v_shaped)
    eval_logval(orig_density, v_reshaped, T)
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
