# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct BispacedMeasure <: BATMeasure

*BAT-internal, not part of stable public API.*

A measure in its primary space, together with an optional representation
of it in a transformed space.

Constructors:

```julia
BispacedMeasure(main::BATMeasure)  # canonical form without a transformed representation
BispacedMeasure(f_transform, main)  # transformed side generated via f_transform
BispacedMeasure(main::BATMeasure, transformed::Union{BATMeasure,Nothing}, f_hash::UInt)
```

As a measure, a `BispacedMeasure` behaves like its `main` side (common
measure operations delegate to it). The pair itself does not identify the
transformed space: that meaning comes from the transform intent of the
[`EvaluatedMeasure`](@ref) the pair is part of (see its `transform_intent`
property), relative to that measure's own content.

`f_hash` is the hash of the transformation function the `transformed` side
was produced under and serves as a cheap compatibility witness when pairs
are adopted into or consumed from an [`EvaluatedMeasure`](@ref): a
non-matching hash results in an error, never in silently wrong content
(`hash` may be specialized for transformation function types to make the
witness value-based instead of object-based, widening which pairs are
recognized as compatible). The witness is a strong practical guard, not a
logical proof of identity: a hash collision, or an over-coarse user `hash`
specialization, could in principle let incompatible content pass. `UInt(0)` means that no claim is made (no
transformed side, or the claim was invalidated). The witness only covers
the pair-internal connection between the two sides; that the main side
itself belongs where it is supplied is the responsibility of the supplier,
just as with raw samples. Function types without a value-based `hash`
specialization fall back to `objectid`, so their stamps do not survive
serialization or a new session; this errs on the safe side (an error, upon
which the transformed content can simply be re-derived).
"""
struct BispacedMeasure{M<:BATMeasure,T<:Union{BATMeasure,Nothing}} <: BATMeasure
    main::M
    transformed::T
    f_hash::UInt
end

BispacedMeasure(main::BATMeasure) = BispacedMeasure(main, nothing, UInt(0))

# Self-building form: the transformed side is the transform result by
# construction, stamped with the hash of the very transformation used:
function BispacedMeasure(f_transform, main, context = get_batcontext())
    f_transform isa BATMeasure && throw(ArgumentError("The first argument of BispacedMeasure(f_transform, main) must be a transformation function, not a measure. To adopt an existing transformed representation, use BispacedMeasure(main, transformed, f_hash)."))
    m_main = batmeasure(main)
    m_transformed = bat_transform(f_transform, m_main, context).result
    return BispacedMeasure(m_main, m_transformed, hash(f_transform))
end


_as_bispaced(::Nothing) = nothing
_as_bispaced(p::BispacedMeasure) = p
_as_bispaced(m::BATMeasure) = BispacedMeasure(m)

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
_approx_max_logd(p::BispacedMeasure) = _approx_max_logd(p.main)

# The transformed-space representation is already unshaped. Reparametrizing
# the main side invalidates the transformation claim of a free-standing
# pair (EvaluatedMeasure re-stamps its pairs with the hash of the
# correspondingly composed transformation instead):
ValueShapes.unshaped(p::BispacedMeasure, vs::AbstractValueShape) = BispacedMeasure(unshaped(p.main, vs), p.transformed, UInt(0))
