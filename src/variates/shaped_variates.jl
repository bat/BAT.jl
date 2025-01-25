# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function _check_reshape_varshape_compat(trgshape::AbstractValueShape, srcshape::AbstractValueShape)
    if !(srcshape <= trgshape)
        throw(ArgumentError("Shape of variate doesn't match target variate trgshape, with variate of type $(typeof(v)) and expected trgshape $(trgshape)"))
    end
end
function ChainRulesCore.rrule(::typeof(_check_reshape_varshape_compat), trgshape::AbstractValueShape, srcshape::AbstractValueShape)
    _check_reshape_varshape_compat_pullback(ΔΩ) = (NoTangent(), NoTangent(), ZeroTangent())
    return _check_reshape_varshape_compat(trgshape, srcshape), _check_reshape_varshape_compat_pullback
end

function _check_reshape_ndof_compat(trgshape::AbstractValueShape, srcshape::AbstractValueShape)
    ndof_trg = totalndof(trgshape)
    ndof_src = totalndof(srcshape)
    if ndof_trg != ndof_src
        throw(ArgumentError("Source varshape has ($ndof_src) degrees of freedom but target varshape has $ndof_trg"))
    end
end
function ChainRulesCore.rrule(::typeof(_check_reshape_ndof_compat), trgshape::AbstractValueShape, srcshape::AbstractValueShape)
    _check_reshape_ndof_compat_pullback(ΔΩ) = (NoTangent(), NoTangent(), ZeroTangent())
    return _check_reshape_ndof_compat(trgshape, srcshape), _check_reshape_ndof_compat_pullback
end

reshape_variate(trgshape::Missing, srcshape::Any, v::Any) = throw(ArgumentError("Can't reshape variate without trgshape information"))

function reshape_variate(trgshape::VS, srcshape::VS, v::Any) where {VS<:AbstractValueShape}
    _check_reshape_varshape_compat(trgshape, srcshape)
    v
end

function reshape_variate(trgshape::ScalarShape{Real}, srcshape::ScalarShape{Real}, v::Real)
    v
end

function reshape_variate(trgshape::ArrayShape{<:Real,1}, srcshape::ArrayShape{<:Real,1}, v::AbstractVector{<:Real})
    _check_reshape_varshape_compat(trgshape, srcshape)
    v
end

function reshape_variate(trgshape::ArrayShape{<:Real,1}, srcshape::AbstractValueShape, v::Any)
    _check_reshape_ndof_compat(trgshape, srcshape)
    inverse(srcshape)(v)
end

function reshape_variate(trgshape::AbstractValueShape, srcshape::ArrayShape{<:Real,1}, v::AbstractVector{<:Real})
    _check_reshape_ndof_compat(trgshape, srcshape)
    trgshape(v)
end



function reshape_variates(trgshape::VS, srcshape::VS, vs::AbstractVector{<:Any}) where {VS<:AbstractValueShape}
    _check_reshape_varshape_compat(trgshape, srcshape)
    vs
end

function reshape_variates(trgshape::ScalarShape{Real}, srcshape::ScalarShape{Real}, vs::AbstractVector{Real})
    vs
end

function reshape_variates(trgshape::ArrayShape{<:Real,1}, srcshape::ArrayShape{<:Real,1}, vs::AbstractVector{<:AbstractVector{<:Real}})
    _check_reshape_varshape_compat(trgshape, srcshape)
    vs
end

function reshape_variates(trgshape::ArrayShape{<:Real,1}, srcshape::AbstractValueShape, vs::AbstractVector{<:AbstractVector{<:Real}})
    _check_reshape_ndof_compat(trgshape, srcshape)
    inverse(srcshape).(vs)
end

function reshape_variates(trgshape::AbstractValueShape, srcshape::ArrayShape{<:Real,1}, vs::AbstractVector{<:AbstractVector{<:Real}})
    _check_reshape_ndof_compat(trgshape, srcshape)
    trgshape(vs)
end



function check_variate end

function ChainRulesCore.rrule(::typeof(check_variate), trgshape::Any, v::Any)
    result = check_variate(trgshape, v)
    _check_variate_pullback(ΔΩ) = (NoTangent(), NoTangent(), ZeroTangent())
    return result, _check_variate_pullback
end

check_variate(trgshape::ScalarShape{Real}, v::Real) = nothing

function check_variate(trgshape::ArrayShape{<:Real,N}, v::AbstractArray{<:Real,N}) where N
    ndof = length(eachindex(v))
    ndof_expected = totalndof(trgshape)
    if ndof != ndof_expected
        throw(ArgumentError("Invalid length ($ndof) of variate, target trgshape  $(trgshape) has $ndof_expected degrees of freedom"))
    end
    nothing
end

function check_variate(trgshape::NamedTupleShape, v::ShapedAsNT)
    if !(valshape(v) <= trgshape)
        throw(ArgumentError("Shape of variate incompatible with target variate trgshape, with variate of type $(typeof(v)) and expected trgshape $(trgshape)"))
    end
    nothing
end

function check_variate(trgshape::NamedTupleShape{names}, v::NamedTuple{names}) where names
    nothing
end

function check_variate(trgshape::Any, v::Any)
    throw(ArgumentError("Shape of variate incompatible with target variate trgshape, with variate of type $(typeof(v)) and expected trgshape $(trgshape)"))
end

check_variate(trgshape::Missing, v::Any) = nothing
