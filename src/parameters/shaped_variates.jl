# This file is a part of BAT.jl, licensed under the MIT License (MIT).


#ToDo: Move functionality of reshape_variate and reshape_variates to package ValueShapes.


reshape_variate(shape::Missing, v::Any) = v

function reshape_variate(shape::AbstractValueShape, v::Any)
    v_shape = valshape(v)
    if !(v_shape <= shape)
        throw(ArgumentError("Shape of variate doesn't match target variate shape, with variate of type $(typeof(v)) and expected shape $(shape)"))
    end
    v
end

function reshape_variate(shape::ArrayShape{<:Real,1}, v::Any)
    unshaped_v = unshaped(v)::AbstractVector{<:Real}
    reshape_variate(shape, unshaped_v)
end

function reshape_variate(shape::AbstractValueShape, v::AbstractVector{<:Real})
    _reshape_realvec(shape, v)
end

function reshape_variate(shape::ArrayShape{<:Real,1}, v::AbstractVector{<:Real})
    _reshape_realvec(shape, v)
end

function _reshape_realvec(shape::AbstractValueShape, v::AbstractVector{<:Real})
    ndof = length(eachindex(v))
    ndof_expected = totalndof(shape)
    if ndof != ndof_expected
        throw(ArgumentError("Invalid length ($ndof) of variate, target shape  $(shape) has $ndof_expected degrees of freedom"))
    end
    shape(v)
end


reshape_variates(shape::Missing, vs::AbstractVector{<:Any}) = v

function reshape_variates(shape::AbstractValueShape, vs::AbstractVector{<:Any})
    v_elshape = elshape(vs)
    if !(v_elshape <= shape)
        throw(ArgumentError("Shape of variates doesn't match target variate shape, with variates of type $(eltype(vs)) and expected element shape $(shape)"))
    end
    vs
end

function reshape_variates(shape::ArrayShape{<:Real,1}, vs::AbstractVector{<:Any})
    unshaped_vs = unshaped.(vs)::AbstractVector{<:AbstractVector{<:Real}}
    reshape_variates(shape, unshaped_vs)
end

function reshape_variates(shape::AbstractValueShape, vs::AbstractVector{<:AbstractVector{<:Real}})
    _reshape_realvecs(shape, vs)
end

function reshape_variates(shape::ArrayShape{<:Real,1}, vs::AbstractVector{<:AbstractVector{<:Real}})
    _reshape_realvecs(shape, vs)
end

function _reshape_realvecs(shape::AbstractValueShape, vs::AbstractVector{<:AbstractVector{<:Real}})
    ndof = first(innersize(vs))
    ndof_expected = totalndof(shape)
    if ndof != ndof_expected
        throw(ArgumentError("Invalid length ($ndof) of variates, target shape $(shape) has $ndof_expected degrees of freedom"))
    end
    shape.(vs)
end
