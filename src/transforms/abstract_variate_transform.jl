# This file is a part of BAT.jl, licensed under the MIT License (MIT).


const OptionalLADJ = Union{Real,Missing}


"""
    abstract type AbstractVariateTransform <: Function

Abstract type for change-of-variables transformations.

Subtypes (e.g. `SomeTrafo <: AbstractVariateTransform`) must support (with
`trafo::SomeTrafo`):

```julia
    (trafo)(v_prev::SomeVariate) == v_new
    (trafo)(v_prev::SomeVariate, ladj_prev::Union{Real,Missing})) == (v = v_new, ladj = ladj_new)
    (trafo)(s_prev::DensitySample)::DensitySample
    ((trafo2 ∘ trafo1)(v)::AbstractVariateTransform)(v) == trafo2(trafo1(v))
    inv(trafo)(trafo(v)) == v
    inv(inv(trafo)) == trafo

    ValueShapes.varshape(trafo)::ValueShapes.AbstractValueShape
```

with `varshape(v_prev) == varshape(trafo)`.

`ladj` must be `logabsdet(jacobian(trafo, v))`.
"""
abstract type AbstractVariateTransform <: Function end
export AbstractVariateTransform



"""
    ladjof(r::NamedTuple{(...,:ladj,...)})::Union{Real,Missing}

Extract the `log(abs(det(jacobian)))` value that is part of a result `r`.

Examples:

```julia
ladjof((..., ladj = some_ladj, ...)) == some_ladj
ladjof(trafo)(v) = trafo(v, )
```
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

Returns a function that computes the `log(abs(det(jacobian)))` of `trafo` for
a given variate `v`:

```julia
    ladjof(trafo)(v) == ladjof(trafo(v, 0))
    ladjof(trafo)(v, prev_ladj) == ladjof(trafo(v, prev_ladj))
```
"""
ladjof(trafo::AbstractVariateTransform) = LADJOfVarTrafo(trafo)


function _transform_density_sample(trafo::AbstractVariateTransform, s::DensitySample)
    r = trafo(s.v, zero(Float32))
    v = stripscalar(r.v)  # ToDo: Do we want to use stripscalar here?
    logd = s.logd - r.ladj
    DensitySample(v, logd, s.weight, s.info, s.aux)
end

(trafo::AbstractVariateTransform)(s::DensitySample) = _transform_density_sample(trafo, s)


# Custom broadcast(::AbstractVariateTransform, DensitySampleVector), multithreaded:
function Base.copy(
    instance::Base.Broadcast.Broadcasted{
        <:Base.Broadcast.AbstractArrayStyle{1},
        <:Any,
        <:AbstractVariateTransform,
        <:Tuple{<:Union{ArrayOfSimilarVectors{<:Real},ShapedAsNTArray}}
    }
)
    trafo = instance.f
    v_src = instance.args[1]
    vs_trg = valshape(trafo(first(v_src)))
    v_src_us = unshaped.(v_src)

    n = length(eachindex(v_src_us))
    v_trg_unshaped = nestedview(similar(flatview(v_src_us), totalndof(vs_trg), n))
    @assert axes(v_trg_unshaped) == axes(v_src)
    @assert v_trg_unshaped isa ArrayOfSimilarArrays
    @threads for i in eachindex(v_trg_unshaped, v_src)
        r = trafo(v_src[i])
        v_trg_unshaped[i] .= unshaped(r)
    end
    vs_trg.(v_trg_unshaped)
end

function Base.copy(
    instance::Base.Broadcast.Broadcasted{
        <:Base.Broadcast.AbstractArrayStyle,
        <:Any,
        <:AbstractVariateTransform,
        <:Tuple{DensitySampleVector}
    }
)
    trafo = instance.f
    s_src = instance.args[1]
    vs_trg = valshape(trafo(first(s_src.v)))
    s_src_us = unshaped.(s_src)

    n = length(eachindex(s_src_us))
    s_trg_unshaped = DensitySampleVector((
        nestedview(similar(flatview(s_src_us.v), totalndof(vs_trg), n)),
        zero(s_src_us.logd),
        deepcopy(s_src_us.weight),
        deepcopy(s_src_us.info),
        deepcopy(s_src_us.aux),
    ))
    @assert axes(s_trg_unshaped) == axes(s_src)
    @assert s_trg_unshaped.v isa ArrayOfSimilarArrays
    @threads for i in eachindex(s_trg_unshaped, s_src)
        r = trafo(s_src.v[i], zero(Float32))
        s_trg_unshaped.v[i] .= unshaped(r.v)
        s_trg_unshaped.logd[i] = s_src_us.logd[i] - r.ladj
    end
    vs_trg.(s_trg_unshaped)
end


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

function var_trafo_result(trg_v::Any, src_v::Any, trafo_ladj::OptionalLADJ, prev_ladj::OptionalLADJ)
    # TODO: Preserve precision of src_v while keeping dual-number properties of trg_v
    trg_ladj = _combined_trafo_ladj(trafo_ladj, prev_ladj, any_isinf(trg_v))
    (v = trg_v, ladj = trg_ladj)
end


# ToDo: Remove intermediate type `VariateTransform`?

"""
    abstract type VariateTransform{VT<:AbstractValueShape,VF<:AbstractValueShape}

*BAT-internal, not part of stable public API.*

Abstract parameterized type for change-of-variables transformations.

Subtypes (e.g. `SomeTrafo <: VariateTransform`) must implement:

* `BAT.apply_vartrafo_impl(trafo::SomeTrafo, v)`
* `BAT.apply_vartrafo_impl(inv_trafo::InverseVT{SomeTrafo}, v)`
* `ValueShapes.varshape(trafo::SomeTrafo)`

for real values and/or real-valued vectors `v`.
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

apply_vartrafo(trafo::VariateTransform{<:Any,<:ValueShapes.NamedTupleShape{names}}, v::ShapedAsNT{<:NamedTuple{names}}, prev_ladj::OptionalLADJ) where names =
    apply_vartrafo_impl(trafo, v, prev_ladj)


(trafo::VariateTransform)(v::Any) = apply_vartrafo(trafo, v, Float32(NaN)).v
(trafo::VariateTransform)(v::Any, prev_ladj::OptionalLADJ) = apply_vartrafo(trafo, v, prev_ladj)
(trafo::VariateTransform)(s::DensitySample) = _transform_density_sample(trafo, s)



struct IdentityVT{
    VTF <: AbstractValueShape
} <: VariateTransform{VTF,VTF}
    varshape::VTF
end

Base.inv(trafo::IdentityVT) = trafo

ValueShapes.varshape(trafo::IdentityVT) = trafo.varshape

import Base.∘
@inline ∘(a::AbstractVariateTransform, b::IdentityVT) = a
@inline ∘(a::IdentityVT, b::IdentityVT) = a
@inline ∘(a::IdentityVT, b::AbstractVariateTransform) = b


@inline apply_vartrafo_impl(trafo::IdentityVT, v::Any, prev_ladj::OptionalLADJ) = (v = v, ladj = prev_ladj)

(trafo::IdentityVT)(s::DensitySample) = s


# Custom broadcast(::IdentityVT, DensitySampleVector), multithreaded:

function Base.copy(
    instance::Base.Broadcast.Broadcasted{
        <:Base.Broadcast.AbstractArrayStyle,
        <:Any,
        <:IdentityVT,
        <:Tuple{<:Union{ArrayOfSimilarVectors{<:Real},ShapedAsNTArray,DensitySampleVector}}
    }
)
    deepcopy(instance.args[1])
end
