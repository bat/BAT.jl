# This file is a part of BAT.jl, licensed under the MIT License (MIT).


const OptionalLADJ = Union{Real,Missing}


"""
    abstract type AbstractVariateTransform <: Function

**Deprecated**
"""
abstract type AbstractVariateTransform <: Function end
export AbstractVariateTransform


InverseFunctions.inverse(trafo::AbstractVariateTransform) = inv(trafo)

function ChangesOfVariables.with_logabsdet_jacobian(trafo::AbstractVariateTransform, x)
    r = trafo(x, 0)
    return r.v, r.ladj
end


ValueShapes.unshaped(trafo::AbstractVariateTransform) =
    _generic_unshaped_impl(trafo, varshape(trafo), valshape(trafo))

_generic_unshaped_impl(trafo::AbstractVariateTransform, ::ArrayShape{<:Real,1}, ::ArrayShape{<:Real,1}) =
    trafo

# ToDo: Add `UnshapedVariateTransform` that shapes at input and unshaped at
# output, to return from
# `_generic_unshaped_impl(trafo::AbstractVariateTransform, ::AbstractValueShape, ::AbstractValueShape)`.


"""
    ladjof(r::NamedTuple{(...,:ladj,...)})::Union{Real,Missing}

**Deprecated**
"""
function ladjof end
export ladjof

ladjof(x::NamedTuple) = x.ladj



struct LADJOfVarTrafo{T<:AbstractVariateTransform} <: Function
    trafo::T
end

(ladjof_trafo::LADJOfVarTrafo)(v::Any) = ladjof(trafo(v, 0))
(ladjof_trafo::LADJOfVarTrafo)(v::Any, prev_ladj::OptionalLADJ) = ladjof(trafo(v, prev_ladj))


"""
    ladjof(trafo::AbstractVariateTransform)::Function

**Deprecated**
"""
ladjof(trafo::AbstractVariateTransform) = LADJOfVarTrafo(trafo)


function _transform_density_sample(trafo::AbstractVariateTransform, s::DensitySample)
    r = trafo(s.v, zero(Float32))
    v = r.v
    logd = s.logd - r.ladj
    DensitySample(v, logd, s.weight, s.info, s.aux)
end

(trafo::AbstractVariateTransform)(s::DensitySample) = _transform_density_sample(trafo, s)


function _combined_trafo_ladj(trafo_ladj::OptionalLADJ, prev_ladj::OptionalLADJ, trg_v_isinf::Bool)
    if ismissing(trafo_ladj) || ismissing(prev_ladj)
        missing
    else
        ladj_sum = trafo_ladj + prev_ladj
        R = typeof(ladj_sum)
        if !isnan(ladj_sum)
            ladj_sum
        else
            # Should be safe to assume that target dist goes to zero at infinity, should win out over infinite prev_ladj:
            ladjs_should_cancel = (trafo_ladj == R(-Inf) && prev_ladj == R(+Inf) && trg_v_isinf)
            ladjs_should_cancel ? zero(R) : ladj_sum
        end
    end
end



# ToDo: Remove intermediate type `VariateTransform`?

"""
    abstract type VariateTransform{VT<:AbstractValueShape,VF<:AbstractValueShape}

**Deprecated**
"""
abstract type VariateTransform{
    VT<:AbstractValueShape,VF<:AbstractValueShape
} <: AbstractVariateTransform end

function apply_vartrafo end

function apply_vartrafo_impl end


apply_vartrafo(trafo::VariateTransform{<:Any,<:ScalarShape{T}}, v::T, prev_ladj::OptionalLADJ) where {T<:Real} =
    apply_vartrafo_impl(trafo, v, prev_ladj)

function apply_vartrafo(trafo::VariateTransform{<:Any,<:ScalarShape{T}}, v::AbstractArray{<:T,0}, prev_ladj::OptionalLADJ) where {T<:Real}
    r = apply_vartrafo_impl(trafo, v[], prev_ladj)
    (v = fill(r.v), ladj = r.ladj)
end
    
apply_vartrafo(trafo::VariateTransform{<:Any,<:ArrayShape{T,N}}, v::AbstractArray{<:T,N}, prev_ladj::OptionalLADJ) where {T<:Real,N} =
    apply_vartrafo_impl(trafo, v, prev_ladj)

apply_vartrafo(trafo::VariateTransform{<:Any,<:ValueShapes.NamedTupleShape{names}}, v::NamedTuple{names}, prev_ladj::OptionalLADJ) where names =
    apply_vartrafo_impl(trafo, v, prev_ladj)

apply_vartrafo(trafo::VariateTransform{<:Any,<:ValueShapes.NamedTupleShape{names}}, v::ShapedAsNT{names}, prev_ladj::OptionalLADJ) where names =
    apply_vartrafo_impl(trafo, v, prev_ladj)


(trafo::VariateTransform)(v::Any) = apply_vartrafo(trafo, v, missing).v
(trafo::VariateTransform)(v::Any, prev_ladj::OptionalLADJ) = apply_vartrafo(trafo, v, prev_ladj)
(trafo::VariateTransform)(s::DensitySample) = _transform_density_sample(trafo, s)



struct IdentityVT{
    VTF <: AbstractValueShape
} <: VariateTransform{VTF,VTF}
    varshape::VTF
end

Base.inv(trafo::IdentityVT) = trafo

ValueShapes.varshape(trafo::IdentityVT) = trafo.varshape
ValueShapes.valshape(trafo::IdentityVT) = trafo.varshape

ValueShapes.unshaped(trafo::IdentityVT{<:ArrayShape{<:Any,1}}) = trafo

import Base.∘
@inline ∘(a::AbstractVariateTransform, b::IdentityVT) = a
@inline ∘(a::IdentityVT, b::IdentityVT) = a
@inline ∘(a::IdentityVT, b::AbstractVariateTransform) = b


@inline apply_vartrafo_impl(trafo::IdentityVT, v::Any, prev_ladj::OptionalLADJ) = (v = v, ladj = prev_ladj)


function broadcast_trafo(
    ::IdentityVT,
    v_src::Union{ArrayOfSimilarVectors{<:Real},ShapedAsNTArray}
)
    deepcopy(v_src)
end

function broadcast_trafo(
    ::IdentityVT,
    s_src::DensitySampleVector
)
    deepcopy(s_src)
end


(trafo::IdentityVT)(s::DensitySample) = s


function Base.Broadcast.broadcasted(
    ::IdentityVT,
    v_src::Union{ArrayOfSimilarVectors{<:Real},ShapedAsNTArray}
)
    broadcast_trafo(identity, v_src)
end

function Base.Broadcast.broadcasted(
    ::IdentityVT,
    s_src::DensitySampleVector
)
    broadcast_trafo(identity, s_src)
end
