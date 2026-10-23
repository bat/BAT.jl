# This file is a part of BAT.jl, licensed under the MIT License (MIT).


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
