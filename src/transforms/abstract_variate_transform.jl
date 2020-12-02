# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc doc"""
    AbstractVariateTransform <: Function

Abstract super-type for change-of-variables transformations.

Subtypes (e.g. `SomeTrafo <: AbstractVariateTransform`) must support
[`ValueShapes.varshape`](@ref) and

```julia
    (trafo)(v_prev::SomeVariate) == v_new
    (trafo)(v_prev::SomeVariate, ladj_prev::Real)) == (v = v_new, ladj = ladj_new)
    (trafo)(s_prev::DensitySample) == s_new::DensitySample
    ((trafo2 ∘ trafo1)(v)::AbstractVariateTransform)(v) == trafo2(trafo1(v))
    inv(trafo)(trafo(v)) == v
    inv(inv(trafo)) == trafo
```

with `varshape(v_prev) == varshape(trafo)`.

`ladj` must be `logabsdet(jacobian(trafo, v))`.
"""
abstract type AbstractVariateTransform end
export AbstractVariateTransform



@doc doc"""
    ladjof(r::NamedTuple{(...,:ladj,...)})::Real

Extract the `log(abs(det(jacobian)))`` value that is part of a result `r`.

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

(ladjof_trafo::LADJOfVarTrafo)(v::Any) = ladjof(trafo(v, 1))
(ladjof_trafo::LADJOfVarTrafo)(v::Any, prev_ladj::Real) = ladjof(trafo(v, prev_ladj))


"""
    ladjof(trafo::AbstractVariateTransform)::Function

Returns a function that computes the `log(abs(det(jacobian)))` of `trafo` for
a given variate `v`:

```julia
    ladjof(trafo)(v) == ladjof(trafo(v, 1))
    ladjof(trafo)(v, prev_ladj) == ladjof(trafo(v, prev_ladj))
```
"""
ladjof(trafo::AbstractVariateTransform) = LADJOfVarTrafo(trafo)



"""
    abstract type VariateSpace <: Function

*BAT-internal, not part of stable public API.*

Abstract type for variate spaces.
"""
abstract type VariateSpace end

struct UnitSpace <: VariateSpace end
struct InfiniteSpace <: VariateSpace end
struct MixedSpace <: VariateSpace end


product_varspace(s::VariateSpace) = s

product_varspace(::UnitSpace, ::UnitSpace) = MixedSpace
product_varspace(::InfiniteSpace, ::InfiniteSpace) = MixedSpace
product_varspace(::VariateSpace, ::VariateSpace) = MixedSpace

function product_varspace(s1::VariateSpace, s2::VariateSpace, spcs::VariateSpace...)
    product_varspace(product_varspace(s1, s2), spcs...)
end



@doc doc"""
    VariateTransform{VF:<VariateForm,ST<:VariateSpace,SF<:VariateSpace}

*BAT-internal, not part of stable public API.*

Abstract parameterized type for change-of-variables transformations.

Subtypes (e.g. `SomeTrafo <: VariateTransform`) must implement:

* `BAT.target_space(trafo::SomeTrafo, v)`
* `BAT.source_space(trafo::SomeTrafo, v)`
* `BAT.apply_vartrafo_impl(trafo::SomeTrafo, v)`
* `BAT.apply_vartrafo_impl(inv_trafo::InverseVT{SomeTrafo}, v)`
* `ValueShapes.varshape(trafo::SomeTrafo)`

for real values and/or real-valued vectors `v`.
"""
abstract type VariateTransform{
    VF<:VariateForm,ST<:VariateSpace,SF<:VariateSpace
} <: AbstractVariateTransform end


function target_space end

function source_space end

function apply_vartrafo end

function apply_vartrafo_impl end


apply_vartrafo(trafo::VariateTransform{Univariate}, v::Real, prev_ladj::Real) =
    apply_vartrafo_impl(trafo, v, prev_ladj)

function apply_vartrafo(trafo::VariateTransform{Univariate}, v::AbstractArray{<:Real,0}, prev_ladj::Real)
    r = apply_vartrafo_impl(trafo, v[], prev_ladj)
    (v = fill(r.v), ladj = r.ladj)
end
    
apply_vartrafo(trafo::VariateTransform{Multivariate}, v::AbstractVector{<:Real}, prev_ladj::Real) =
    apply_vartrafo_impl(trafo, v, prev_ladj)

apply_vartrafo(trafo::VariateTransform{Matrixvariate}, v::AbstractMatrix{<:Real}, prev_ladj::Real) =
    apply_vartrafo_impl(trafo, v, prev_ladj)

apply_vartrafo(trafo::VariateTransform{ValueShapes.NamedTupleVariate{names}}, v::NamedTuple{names}, prev_ladj::Real) where names =
    apply_vartrafo_impl(trafo, v, prev_ladj)

apply_vartrafo(trafo::VariateTransform{ValueShapes.NamedTupleVariate{names}}, v::ShapedAsNT{<:NamedTuple{names}}, prev_ladj::Real) where names =
    apply_vartrafo_impl(trafo, v, prev_ladj)


(trafo::VariateTransform)(v::Any) = apply_vartrafo(trafo, v, Float32(NaN)).v
(trafo::VariateTransform)(v::Any, prev_ladj::Real) = apply_vartrafo(trafo, v, prev_ladj)



# ToDo: Move to ValueShapes.jl:
_variate_form(shape::ScalarShape{<:Real}) = Univariate
_variate_form(shape::ArrayShape{<:Real,1}) = Multivariate
_variate_form(shape::ArrayShape{<:Real,2}) = Matrixvariate
_variate_form(shape::NamedTupleShape{names}) where names = ValueShapes.NamedTupleVariate{names}


function var_trafo_result(trg_v::Real, src_v::Real, trafo_ladj::Real, prev_ladj::Real)
    R = float(typeof(src_v))
    ladj_sum = convert(R, trafo_ladj + prev_ladj)
    trg_ladj = if !isnan(ladj_sum)
        ladj_sum
    else
        # Should be safe to assume that target dist goes to zero at infinity, should win out over infinite prev_ladj:
        ladjs_should_cancel = (trafo_ladj == R(-Inf) && prev_ladj == R(+Inf) && isinf(trg_v))
        ladjs_should_cancel ? zero(R) : ladj_sum
    end
    (v = convert(R, trg_v), ladj = trg_ladj)
end

function var_trafo_result(trg_v::Real, src_v::Real)
    R = float(typeof(src_v))
    (v = convert(R, trg_v), ladj = convert(R, NaN))
end

function var_trafo_result(trg_v::AbstractVector{<:Real}, src_v::AbstractVector{<:Real}, trafo_ladj::Real, prev_ladj::Real)
    R = float(eltype(src_v))
    ladj_sum = convert(R, trafo_ladj + prev_ladj)
    trg_ladj = if !isnan(ladj_sum)
        ladj_sum
    else
        # Should be safe to assume that target dist goes to zero at infinity, should win out over infinite prev_ladj:
        ladjs_should_cancel = (trafo_ladj == R(-Inf) && prev_ladj == R(+Inf) && any(isinf, trg_v))
        ladjs_should_cancel ? zero(R) : ladj_sum
    end
    (v = convert_eltype(R, trg_v), ladj = trg_ladj)
end

function var_trafo_result(trg_v::AbstractVector{<:Real}, src_v::AbstractVector{<:Real})
    R = float(eltype(src_v))
    (v = convert_eltype(R, trg_v), ladj = convert(R, NaN))
end



function (trafo::VariateTransform)(s::DensitySample)
    r = trafo(s.v, zero(Float32))
    v = stripscalar(r.v)  # ToDo: Do we want to use stripscalar here?
    logd = s.logd - r.ladj
    DensitySample(v, logd, s.weight, s.info, s.aux)
end

# Custom broadcast(::VariateTransform, DensitySampleVector), multithreaded:
function Base.copy(
    instance::Base.Broadcast.Broadcasted{
        <:Base.Broadcast.AbstractArrayStyle{1},
        <:Any,
        <:VariateTransform,
        <:Tuple{DensitySampleVector}
    }
)
    trafo = instance.f
    s_src = instance.args[1]
    vs_trg = valshape(trafo(first(s_src.v)))
    s_trg_unshaped = deepcopy(unshaped.(s_src))
    @assert axes(s_trg_unshaped) == axes(s_src)
    @assert s_trg_unshaped.v isa ArrayOfSimilarArrays
    @threads for i in eachindex(s_trg_unshaped, s_src)
        r = trafo(s_src.v[i], zero(Float32))
        s_trg_unshaped.v[i] .= unshaped(r.v)
        s_trg_unshaped.logd[i] -= r.ladj
    end
    vs_trg.(s_trg_unshaped)
end


struct InvVT{
    VF <: VariateForm,
    ST <: VariateSpace,
    SF <: VariateSpace,
    FT <: VariateTransform{VF,ST,SF}
} <: VariateTransform{VF,SF,ST}
    orig::FT
end


Base.inv(trafo::VariateTransform) = InvVT(trafo)
Base.inv(trafo::InvVT) = trafo.orig

ValueShapes.varshape(trafo::InvVT) = varshape(trafo.orig)


target_space(trafo::InvVT) = source_space(trafo.orig)

source_space(trafo::InvVT) = target_space(trafo.orig)


const InverseVT{FT} = InvVT{VF,ST,SF,FT} where {VF,ST,SF}



struct IdentityVT{
    VF <: VariateForm,
    S <: VariateSpace,
    VS <: AbstractValueShape
} <: VariateTransform{VF,S,S}
    space::S
    varshape::VS
end

function IdentityVT(space::VariateSpace, shape::AbstractValueShape)
    VF = _variate_form(shape)
    S = typeof(space)
    VS = typeof(shape)
    IdentityVT{VF,S,VS}(space, shape)
end

Base.inv(trafo::IdentityVT) = trafo

ValueShapes.varshape(trafo::IdentityVT) = trafo.varshape


function apply_vartrafo_impl(trafo::IdentityVT, v::Any, prev_ladj::Real)
    (v = v, ladj = prev_ladj)
end

function apply_vartrafo_impl(trafo::InvVT{<:IdentityVT}, v::Any, prev_ladj::Real)
    (v = v, ladj = prev_ladj)
end
