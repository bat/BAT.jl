# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct BispacedMeasure <: AbstractMeasure

*BAT-internal, not part of stable public API.*

A measure in its primary space, together with an optional representation
of it in a transformed space.

Constructors:

```julia
BispacedMeasure(main::AbstractMeasure)  # no transformed representation
BispacedMeasure(f_transform, main)  # transformed side generated via f_transform
BispacedMeasure(main::AbstractMeasure, transformed::Union{AbstractMeasure,Nothing}, f_hash::UInt)
```

As a measure, a `BispacedMeasure` behaves like its `main` side. The pair
itself does not identify the transformed space, that meaning comes from
the `transform_intent` of the [`EvaluatedMeasure`](@ref) the pair is
part of.

# Implementation

`f_hash` is the [`BAT.transform_witness`](@ref) of the transformation
function that produced the `transformed` side. It acts as a cheap
compatibility witness when pairs are adopted into or consumed from an
[`EvaluatedMeasure`](@ref): a non-matching witness results in an error,
never in silently wrong content. `UInt(0)` means that no claim is made,
either because there is no transformed side or because the claim was
invalidated. The witness only covers the connection between the two
sides, supplying a fitting main side is the responsibility of the
supplier. It is a strong practical guard, not a proof of identity: a
hash collision could in principle let incompatible content pass.
"""
struct BispacedMeasure{M<:AbstractMeasure,T<:Union{AbstractMeasure,Nothing}} <: AbstractMeasure
    main::M
    transformed::T
    f_hash::UInt
end

BispacedMeasure(main::AbstractMeasure) = BispacedMeasure(main, nothing, UInt(0))

# Self-building form: the transformed side is the transform result by
# construction, stamped with the hash of the very transformation used:
function BispacedMeasure(f_transform, main, context = get_batcontext())
    f_transform isa AbstractMeasure && throw(ArgumentError("The first argument of BispacedMeasure(f_transform, main) must be a transformation function, not a measure. To adopt an existing transformed representation, use BispacedMeasure(main, transformed, f_hash)."))
    m_main = batmeasure(main)
    m_transformed = bat_transform(f_transform, m_main, context).result
    return BispacedMeasure(m_main, m_transformed, transform_witness(f_transform))
end


"""
    BAT.transform_witness(f)::UInt

*BAT-internal, not part of stable public API.*

Value-based hash of a transformation function, the compatibility witness
of [`BAT.BispacedMeasure`](@ref).

MeasureBase's measures and transports hash by value, but BAT's own
measures and function wrappers holding mutable data (like parameter
arrays) would hash by `objectid`, so their structure is hashed explicitly.
"""
function transform_witness end

transform_witness(f) = _structural_hash(f, hash(:bat_transform_witness))

const _StructurallyHashed = Union{
    AbstractMeasure, TransportFunction, FunctionChain, ComposedFunction, Base.Fix1, Base.Fix2
}

_structural_hash(x, h::UInt) = hash(x, h)
_structural_hash(x::_StructurallyHashed, h::UInt) = _fieldwise_hash(x, h)
_structural_hash(xs::Tuple, h::UInt) = foldl((h_i, x) -> _structural_hash(x, h_i), xs, init = h)
_structural_hash(xs::NamedTuple, h::UInt) = _structural_hash(values(xs), hash(keys(xs), h))
_structural_hash(xs::AbstractArray{<:_StructurallyHashed}, h::UInt) =
    foldl((h_i, x) -> _structural_hash(x, h_i), xs, init = hash(size(xs), h))

function _fieldwise_hash(x::T, h::UInt) where T
    h_x = hash(nameof(T), h)
    for i in 1:fieldcount(T)
        h_x = _structural_hash(getfield(x, i), h_x)
    end
    return h_x
end


_as_bispaced(::Nothing) = nothing
_as_bispaced(p::BispacedMeasure) = p
_as_bispaced(m::AbstractMeasure) = BispacedMeasure(m)

_strip_annex(::Nothing) = nothing
_strip_annex(p::BispacedMeasure) = isnothing(p.transformed) ? p : BispacedMeasure(p.main)

_pair_claims_mismatch(p::BispacedMeasure, f_hash::UInt) = !isnothing(p.transformed) && p.f_hash != f_hash
_pair_claims_mismatch(::Any, ::UInt) = false

function _throw_pair_hash_mismatch(what::AbstractString)
    throw(ArgumentError("$what of EvaluatedMeasure was produced under a different transformation than its view (transformation-hash mismatch). This also happens after deserialization for transformation types whose hash is session-bound; strip the stale view via EvaluatedMeasure(em, transform_intent = DoNotTransform()) and re-evaluate to recover."))
end

_has_pair_annex(::Any) = false
_has_pair_annex(p::BispacedMeasure) = !isnothing(p.transformed)


ValueShapes.varshape(p::BispacedMeasure) = varshape(p.main)

DensityInterface.logdensityof(p::BispacedMeasure, v::Any) = logdensityof(p.main, v)
DensityInterface.logdensityof(p::BispacedMeasure) = logdensityof(p.main)

MeasureBase.getdof(p::BispacedMeasure) = getdof(p.main)
MeasureBase.massof(p::BispacedMeasure) = massof(p.main)

samplesof(p::BispacedMeasure) = samplesof(p.main)
empiricalof(p::BispacedMeasure) = empiricalof(p.main)
getess(p::BispacedMeasure) = getess(p.main)

has_uhc_support(p::BispacedMeasure) = has_uhc_support(p.main)
supports_rand(p::BispacedMeasure) = supports_rand(p.main)

MeasureBase.rand_impl(gen::GenContext, p::BispacedMeasure) = rand(gen, p.main)
_approx_max_logd(p::BispacedMeasure) = _approx_max_logd(p.main)

# The transformed-space representation is already unshaped. Reparametrizing
# the main side invalidates the transformation claim of a free-standing
# pair (EvaluatedMeasure re-stamps its pairs with the hash of the
# correspondingly composed transformation instead):
ValueShapes.unshaped(p::BispacedMeasure, vs::AbstractValueShape) = BispacedMeasure(unshaped(p.main, vs), p.transformed, UInt(0))

# Disambiguates against unshaped(x, ::ConstValueShape) of ValueShapes:
ValueShapes.unshaped(p::BispacedMeasure, vs::ConstValueShape) =
    invoke(unshaped, Tuple{BispacedMeasure,AbstractValueShape}, p, vs)
