# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct BAT.MeasureEvalInfo

*Experimental feature, not part of stable public API yet.*

Properties:

* `algorithm`: The algorithm used to evaluate the measure.
* `result`: Algorithm-specific evaluation result.
"""
struct MeasureEvalInfo{Alg,R}
    algorithm::Alg
    result::R
    # ToDo: Store original evaluation context?
    # context::Ctx # Ctx<:BATContext
end


"""
    abstract type AbstractSampleGenerator

*BAT-internal, not part of stable public API.*

Abstract super type for sample generators.
"""
abstract type AbstractSampleGenerator end

# Sample generators are not required to expose a proposal/algorithm object:
getproposal(::AbstractSampleGenerator) = missing


function LazyReports.pushcontent!(rpt::LazyReport, generator::AbstractSampleGenerator)
    alg = getproposal(generator)
    if !(isnothing(alg) || ismissing(alg))
        lazyreport!(rpt, """
        ### Sample generation:

        * Algorithm: $(nameof(typeof(alg)))
        """)
    end
end


"""
    struct EvaluatedMeasure <: BATMeasure

Combines a measure with samples and other information on it.

Constructors:

```julia
em = EvaluatedMeasure(
    measure;
    transform_intent = ..., f_transform = ..., empirical = ..., approx = ...,
    dof = ..., mass = ..., modes = ..., samplegen = ..., transformed = ...,
    evalinfo = ...
)

BAT.unevaluated(em) === BAT.unevaluated(batmeasure(measure))
```

[`unevaluated(em)`](@ref) returns the original measure.

If `measure` is itself an `EvaluatedMeasure`, the keyword arguments update
its content: given values replace the corresponding entries,
`ScopedSettings.unchanged` (the default) keeps them, and `nothing` (resp.
`MeasureBase.UnknownMass()` for `mass`) clears them.

The transformed side of a [`BAT.BispacedMeasure`](@ref) pair carries the
hash of the transformation it was produced under; the constructor checks
that hash against the (possibly updated) `f_transform` of the view, so
pairs — including pairs taken from another `EvaluatedMeasure` of the same
measure, like `EvaluatedMeasure(em1, empirical = em2.empirical)` — are
adopted exactly when their transformed-space content is compatible with
the view, and rejected with an error otherwise (never silently mislabeled,
up to hash collisions, which honest value-based hash specializations make
negligibly improbable). `transformed` and `samplegen` carry no such
witness and must be accompanied by an explicit `transform_intent` in the
same update.

The hash witnesses only that the sides of a pair are connected by the
view's transformation; that the main-side content itself belongs to the
measure is the responsibility of whoever supplies it, exactly as when
supplying raw samples. Hashes of transformation types without a
value-based `hash` specialization are session-bound, so their pairs are
rejected after deserialization; strip the stale view via
`transform_intent = DoNotTransform()` and re-evaluate to recover. Use
[`BAT.validate_evalmeasure`](@ref) to verify the full
transformed-space-view contract of an `EvaluatedMeasure` explicitly.

An `EvaluatedMeasure` maintains at most one transformed-space view of its
content, defined by the contract

```julia
unevaluated.transformed, f_transform == transform_and_unshape(transform_intent, unevaluated.main)
empirical.main == bat_transform(inverse(f_transform), empirical.transformed).result
```

(equal by value — up to floating-point inverse-roundtrip in the second
line, whose sample rows are aligned with identical weights; the stored
entries preserve the object identity of one evaluation of the pure
right-hand sides, whose values do not depend on the evaluation context;
the approximation pair satisfies the corresponding pushforward relation,
and the equations are understood relative to the implied transformation
when `f_transform` is not cached). Every `transformed` side of the
[`BAT.BispacedMeasure`](@ref) entries is the representation in this flat
transformed space. `transform_intent === DoNotTransform()` means that no
transformed-space view exists (`f_transform` is `identity` then, by
convention).

Properties:

* `unevaluated`: The original measure, as a
  [`BAT.BispacedMeasure`](@ref): `em.unevaluated.main` is the bare
  measure itself (returned by [`unevaluated(em)`](@ref)),
  `em.unevaluated.transformed` may cache the bare measure in the
  transformed space. The cache preserves object identity across repeated
  evaluations (keeping compiled artifacts like AD preparations valid); it
  is derived purely from the measure and `transform_intent`, so any copy is
  equally valid.
* `transform_intent`: The [`TransformIntent`](@ref) that identifies the
  transformed space of this measure's content.
* `f_transform`: The concrete transformation function of the view, mapping
  variates of the measure to the flat transformed space (see the contract
  above). Cached to preserve object identity across repeated evaluations
  (keeping compiled artifacts like AD preparations and generated code for
  sample transport valid); derived deterministically from `transform_intent`
  and the measure, so any copy is equally valid. `identity` if no view
  exists, `nothing` if not cached.
* `empirical`: A [`BAT.BispacedMeasure`](@ref) that holds a
  [`DensitySampleMeasure`](@ref) based on samples drawn from the measure,
  possibly together with a row-aligned representation of the same samples in
  the transformed space, or `nothing` if no samples are available.
* `approx`: A [`BAT.BispacedMeasure`](@ref) that holds an approximation
  of the measure, possibly together with a representation of the same
  approximation in the transformed space, or `nothing` if no approximation
  is available. An approximation captures the shape of the measure, its own
  total mass may differ from the mass of the measure (approximations will
  typically be probability measures, like a normal distribution under a
  normalizing flow or a normalized mixture, while the measure itself is
  often non-normalized). Total-mass knowledge about the measure itself
  lives in `mass`.
* `dof`: The degrees of freedom of the measure, or `nothing` if unknown.
* `mass`: The mass of the measure, or a `MeasureBase.AbstractUnknownMass` if
  unknown.
* `modes`: The modes of the measure, or `nothing` if unknown.
* `samplegen`: An object that carries the necessary information to generate
  samples, the contents is algorithm-specific and not part of the stable API.
  Operates in the transformed-space view of the measure (in its flat,
  unshaped realization), like all other transformed-space content; if no
  view exists (`DoNotTransform`), it operates in the unshaped space of the
  measure itself. Consumers must not mutate its content, continuing sample
  generation requires a deep copy. May be `nothing` if no sample generation
  scheme has been computed. In principle independent of `empirical` (it
  provides a way to generate samples of the measure, `evalinfo` records
  what produced the current empirical content); for now, algorithms that
  replace the empirical content without using the stored scheme (like
  resampling and i.i.d. sampling) clear it conservatively.
* `evalinfo`: Information on the (last) evaluation step that
  generated/updated this measure, or `nothing` if no evaluation has been
  performed or information on it is not available.

The `transform_intent` keyword switches the transformed-space view:
`ScopedSettings.unchanged` means that any given transformed-space content
refers to the current view; a differing intent adopts the given content and
strips the transformed-space sides off all entries that are kept, including
the cached transformation function unless a new `f_transform` is given with
it. The `transformed` keyword updates the transformed-space cache of
`unevaluated` with a bare measure, `nothing` drops the cache.
"""
struct EvaluatedMeasure{
    M<:BispacedMeasure,
    TI<:TransformIntent,
    TF,
    S<:Union{BispacedMeasure{<:DensitySampleMeasure},Nothing},
    A<:Union{BispacedMeasure,Nothing},
    N<:Union{IntegerLike,Nothing},
    U<:Union{Real,MeasureBase.AbstractUnknownMass},
    P<:Union{AbstractVector,Nothing},
    G<:Union{AbstractSampleGenerator,Nothing},
} <: BATMeasure
    unevaluated::M
    transform_intent::TI
    f_transform::TF
    empirical::S
    approx::A
    dof::N
    mass::U
    modes::P
    samplegen::G
    evalinfo::Union{MeasureEvalInfo,Nothing}
end
export EvaluatedMeasure


Base.convert(::Type{EvaluatedMeasure}, em::EvaluatedMeasure) = em

function Base.convert(::Type{EvaluatedMeasure}, measurelike::MeasureLike)
    m = batmeasure(measurelike)
    return EvaluatedMeasure(
        BispacedMeasure(m),
        DoNotTransform(),
        identity,
        _as_bispaced(empiricalof(m)),
        _as_bispaced(approxof(m)),
        _dofval_or_nothing(getdof(m)),
        _canonical_mass(massof(m)),
        maybe_modes(m),
        nothing, # ToDo: samplegenof(m)?
        nothing
    )
end

# A free-standing pair does not identify its transformed space, so only its
# main side can be adopted:
Base.convert(::Type{EvaluatedMeasure}, p::BispacedMeasure) = convert(EvaluatedMeasure, p.main)


function EvaluatedMeasure(
    measurelike::MeasureLike;
    transform_intent::Union{TransformIntent,Unchanged} = unchanged,
    f_transform = unchanged,
    empirical::Union{DensitySampleMeasure,DensitySampleVector,BispacedMeasure,Nothing,Unchanged} = unchanged,
    approx::Union{BATMeasure,Nothing,Unchanged} = unchanged,
    dof::Union{IntegerLike,MeasureBase.NoDOF,Nothing,Unchanged} = unchanged,
    mass::Union{RealLike,MeasureBase.AbstractUnknownMass,Unchanged} = unchanged,
    modes::Union{AbstractVector,Nothing,Unchanged} = unchanged,
    samplegen::Union{AbstractSampleGenerator,Nothing,Unchanged} = unchanged,
    transformed::Union{BATMeasure,Nothing,Unchanged} = unchanged,
    evalinfo::Union{MeasureEvalInfo,Nothing,Unchanged} = unchanged
)
    em = convert(EvaluatedMeasure, measurelike)
    _check_em_update_args(transform_intent, f_transform, empirical, approx, samplegen, transformed)
    new_em = _update_evalmeasure(
        em, transform_intent = transform_intent, f_transform = f_transform,
        empirical = empirical, approx = approx, dof = dof, mass = mass,
        modes = modes, samplegen = samplegen, transformed = transformed,
        evalinfo = evalinfo
    )
    _em_sanity_checks(new_em)
    return new_em
end

# Pairs witness the transformation of their transformed side by hash (see
# BispacedMeasure), unlabeled content without such a witness must be
# labeled by an explicit transform intent, and DoNotTransform (no view)
# admits no transformed-space content, except for a sample generator, which
# then operates in the unshaped space of the measure itself:
function _check_em_update_args(transform_intent, f_transform, empirical, approx, samplegen, transformed)
    if transform_intent isa Unchanged && (
        !(transformed isa Union{Unchanged,Nothing}) || !(samplegen isa Union{Unchanged,Nothing})
    )
        throw(ArgumentError("transformed and samplegen require an explicit transform_intent in the same EvaluatedMeasure update"))
    end
    if transform_intent isa DoNotTransform
        if _has_pair_annex(empirical) || _has_pair_annex(approx) || !(transformed isa Union{Unchanged,Nothing})
            throw(ArgumentError("No transformed-space view exists under DoNotTransform, can't store transformed-space representations"))
        end
        if !(f_transform isa Unchanged) && f_transform !== identity
            throw(ArgumentError("f_transform must be identity under DoNotTransform (the convention for the no-view state)"))
        end
    end
    return nothing
end

# Cheap structural checks on a freshly updated EvaluatedMeasure. The full
# transformed-space-view contract can't be verified cheaply, use
# `BAT.validate_evalmeasure` for a full (expensive) check:
function _em_sanity_checks(em::EvaluatedMeasure)
    if em.transform_intent isa DoNotTransform
        if !isnothing(em.unevaluated.transformed) || _has_pair_annex(em.empirical) || _has_pair_annex(em.approx)
            throw(ArgumentError("EvaluatedMeasure carries transformed-space representations but has no transformed-space view (transform_intent is DoNotTransform)"))
        end
        em.f_transform === identity || throw(ArgumentError("An EvaluatedMeasure with no transformed-space view must have f_transform === identity"))
    else
        has_trcontent = !isnothing(em.unevaluated.transformed) || _has_pair_annex(em.empirical) || _has_pair_annex(em.approx)
        if has_trcontent && isnothing(em.f_transform)
            throw(ArgumentError("An EvaluatedMeasure with transformed-space content must carry its f_transform"))
        end
        f_hash = hash(em.f_transform)
        if _pair_claims_mismatch(em.empirical, f_hash) || _pair_claims_mismatch(em.approx, f_hash) || _pair_claims_mismatch(em.unevaluated, f_hash)
            _throw_pair_hash_mismatch("Transformed-space content")
        end
    end
    p = em.empirical
    if _has_pair_annex(p)
        p.transformed isa DensitySampleMeasure || throw(ArgumentError("The transformed-space side of an empirical pair must be a DensitySampleMeasure"))
        length(samplesof(p.main)) == length(samplesof(p.transformed)) || throw(ArgumentError("The sides of an empirical pair differ in sample count"))
    end
    return nothing
end


function _update_evalmeasure(
    em::EvaluatedMeasure;
    transform_intent = unchanged,
    f_transform = unchanged,
    empirical = unchanged,
    approx = unchanged,
    dof = unchanged,
    mass = unchanged,
    modes = unchanged,
    samplegen = unchanged,
    transformed = unchanged,
    evalinfo = unchanged
)
    if (
        transform_intent isa Unchanged && f_transform isa Unchanged &&
        empirical isa Unchanged && approx isa Unchanged &&
        dof isa Unchanged && mass isa Unchanged && modes isa Unchanged &&
        samplegen isa Unchanged && transformed isa Unchanged && evalinfo isa Unchanged
    )
        return em
    end

    # An EvaluatedMeasure maintains a single transformed-space view: given
    # transformed-space content refers to the current view if `transform_intent` is
    # unchanged, while a differing intent strips the transformed-space sides
    # off all entries that are kept:
    new_transform_intent = transform_intent isa Unchanged ? em.transform_intent : transform_intent
    keep_annexes = transform_intent isa Unchanged || _intents_match(em.transform_intent, transform_intent)

    cur_f_transform = keep_annexes ? em.f_transform : _no_f_transform(new_transform_intent)
    new_f_transform = f_transform isa Unchanged ? cur_f_transform : f_transform

    cur_unevaluated = keep_annexes ? em.unevaluated : _strip_annex(em.unevaluated)
    cur_empirical = keep_annexes ? em.empirical : _strip_annex(em.empirical)
    cur_approx = keep_annexes ? em.approx : _strip_annex(em.approx)
    # A sample generator is process state native to the view it was produced
    # in, it has no representation in another view:
    cur_samplegen = keep_annexes ? em.samplegen : nothing

    # The transformed-space measure cache holds a bare measure, so any
    # evaluation knowledge is stripped off a given cache value:
    new_unevaluated = transformed isa Unchanged ? cur_unevaluated :
        isnothing(transformed) ? _strip_annex(cur_unevaluated) :
        BispacedMeasure(cur_unevaluated.main, unevaluated(transformed), hash(new_f_transform))
    new_empirical = empirical isa Unchanged ? cur_empirical : _as_empirical_pair(empirical)
    new_approx = approx isa Unchanged ? cur_approx : _as_bispaced(approx)

    # dof enrichment from new empirical/approx content only happens when
    # such content is actually written, so that an explicitly cleared dof
    # is not resurrected by unrelated updates:
    new_dof = if !(dof isa Unchanged)
        _dofval_or_nothing(dof)
    elseif !(empirical isa Unchanged) || !(approx isa Unchanged)
        choose_something(
            _getdof_or_nothing(em),
            _getdof_or_nothing(new_empirical),
            _getdof_or_nothing(new_approx),
        )
    else
        em.dof
    end

    new_mass = mass isa Unchanged ? _getmass_or_unknown(em) : _canonical_mass(mass)

    # ToDo: Set DOF in empirical if not there yet and inferrable from em.unevaluated?

    return EvaluatedMeasure(
        new_unevaluated,
        new_transform_intent,
        new_f_transform,
        new_empirical,
        new_approx,
        new_dof,
        new_mass,
        modes isa Unchanged ? em.modes : modes,
        samplegen isa Unchanged ? cur_samplegen : samplegen,
        evalinfo isa Unchanged ? em.evalinfo : evalinfo
    )
end

# The transformation function of a fresh view that was reported without one:
_no_f_transform(::DoNotTransform) = identity
_no_f_transform(::TransformIntent) = nothing

_as_empirical_pair(::Nothing) = nothing
_as_empirical_pair(p::BispacedMeasure) = p
_as_empirical_pair(x::Union{DensitySampleMeasure,DensitySampleVector}) = BispacedMeasure(convert(DensitySampleMeasure, x))

_getdof_or_nothing(::Nothing) = nothing
_getdof_or_nothing(measure::BATMeasure) = _dofval_or_nothing(getdof(measure))

_dofval_or_nothing(::Nothing) = nothing
_dofval_or_nothing(dof::IntegerLike) = dof
_dofval_or_nothing(::MeasureBase.NoDOF) = nothing
_dofval_or_nothing(dof) = throw(ArgumentError("Degrees of freedom must be an integer or MeasureBase.NoDOF, not $(nameof(typeof(dof)))."))

_getmass_or_unknown(::Nothing) = MeasureBase.UnknownMass()
_getmass_or_unknown(measure::BATMeasure) = massof(measure)


@inline unevaluated(em::EvaluatedMeasure) = em.unevaluated.main

# A DensitySampleMeasure acts as its own empirical representation:
function _empirical_rep(em::EvaluatedMeasure)
    if !isnothing(em.empirical)
        return em.empirical
    elseif em.unevaluated.main isa DensitySampleMeasure
        return BispacedMeasure(em.unevaluated.main)
    else
        return nothing
    end
end

function empiricalof(em::EvaluatedMeasure)
    p = _empirical_rep(em)
    return isnothing(p) ? nothing : p.main
end

function samplesof(em::EvaluatedMeasure)
    dsm = empiricalof(em)
    return isnothing(dsm) ? nothing : samplesof(dsm)
end

function approxof(em::EvaluatedMeasure)
    p = em.approx
    return isnothing(p) ? nothing : p.main
end
MeasureBase.getdof(em::EvaluatedMeasure) = something(em.dof, MeasureBase.NoDOF{typeof(unevaluated(em))}())
MeasureBase.massof(em::EvaluatedMeasure) = em.mass
maybe_modes(em::EvaluatedMeasure) = em.modes
getess(em::EvaluatedMeasure) = getess(_empirical_or_unevaluated(em))
@inline evalinfo(em::EvaluatedMeasure) = em.evalinfo

@inline samplegenof(em::EvaluatedMeasure) = em.samplegen

function StatsBase.modes(em::EvaluatedMeasure)
    em_modes = maybe_modes(em)
    if isnothing(em_modes)
        throw(ArgumentError("No mode information available for EvaluatedMeasure"))
    else
        return em_modes
    end
end

function StatsBase.mode(em::EvaluatedMeasure)
    em_modes = modes(em)
    if length(em_modes) > 1
        throw(ArgumentError("EvaluatedMeasure has multiple modes"))
    else
        return only(em_modes)
    end
end

function DensitySampleVector(em::EvaluatedMeasure)
    dsm = empiricalof(em)
    if isnothing(dsm)
        throw(ArgumentError("EvaluatedMeasure has no empirical samples attached to it."))
    else
        return DensitySampleVector(dsm)
    end
end
Base.convert(::Type{DensitySampleVector}, em::EvaluatedMeasure) = DensitySampleVector(em)


Base.showable(::MIME"text/plain", ::EvaluatedMeasure) = true
Base.show(io::IO, mime::MIME"text/plain", em::EvaluatedMeasure) = _show_evaluated_measure(io, mime, em)

Base.showable(::MIME"text/html", ::EvaluatedMeasure) = true
Base.show(io::IO, mime::MIME"text/html", em::EvaluatedMeasure) = _show_evaluated_measure(io, mime, em)

# ToDo: Support ::MIME"juliavscode/html" ?
# Base.showable(::MIME"juliavscode/html", ::EvaluatedMeasure) = true
# Base.show(io::IO, mime::MIME"juliavscode/html", em::EvaluatedMeasure) = _show_evaluated_measure(io, mime, em)

function Base.show(io::IO, em::EvaluatedMeasure)
    print(io, "EvaluatedMeasure(")
    show(io, unevaluated(em))
    print(io, "; ...)")
end

function _show_evaluated_measure(@nospecialize(io::IO), @nospecialize(mime::MIME), @nospecialize(em::EvaluatedMeasure))
    smpls = samplesof(em)
    if get(io, :compact, false) || isnothing(smpls)
        show(io, em)
    else
        rpt = lazyreport()
        lazyreport!(rpt, em)
        show(io, mime, rpt)
    end
end


DensityInterface.logdensityof(em::EvaluatedMeasure, v::Any) = logdensityof(unevaluated(em), v)
DensityInterface.logdensityof(em::EvaluatedMeasure) = logdensityof(unevaluated(em))

# Random generation uses the underlying measure, never the empirical
# content (`rand` promises truly IID samples):
Base.rand(gen::GenContext, em::EvaluatedMeasure) = rand(gen, unevaluated(em))
supports_rand(em::EvaluatedMeasure) = supports_rand(unevaluated(em))


ValueShapes.varshape(em::EvaluatedMeasure) = varshape(unevaluated(em))

# `unshaped` is a pure reparametrization, so all measure knowledge is
# transported to the unshaped space. Use `unevaluated` to strip the knowledge
# and obtain a bare measure for performance-critical density evaluation:
function ValueShapes.unshaped(em::EvaluatedMeasure, vs::AbstractValueShape)
    new_f_transform = _unshaped_f(em.f_transform, em.transform_intent, vs)
    # Pairs that witness the view's transformation are re-stamped with the
    # hash of the correspondingly composed transformation, their transformed
    # sides stay valid under it. Any foreign claim is invalidated instead of
    # being relabeled:
    old_f_hash = hash(em.f_transform)
    new_f_hash = hash(new_f_transform)
    new_unevaluated = _unshaped_pair(em.unevaluated, vs, old_f_hash, new_f_hash)
    new_empirical = _unshaped_pair(em.empirical, vs, old_f_hash, new_f_hash)
    new_approx = _unshaped_pair(em.approx, vs, old_f_hash, new_f_hash)
    new_modes = isnothing(em.modes) ? nothing : unshaped.(em.modes, Ref(vs))
    return EvaluatedMeasure(
        new_unevaluated, em.transform_intent, new_f_transform, new_empirical, new_approx,
        em.dof, em.mass, new_modes, em.samplegen, em.evalinfo
    )
end

# Disambiguates against unshaped(x, ::ConstValueShape) of ValueShapes:
ValueShapes.unshaped(em::EvaluatedMeasure, vs::ConstValueShape) =
    invoke(unshaped, Tuple{EvaluatedMeasure,AbstractValueShape}, em, vs)

_unshaped_pair(::Nothing, ::AbstractValueShape, ::UInt, ::UInt) = nothing

function _unshaped_pair(p::BispacedMeasure, vs::AbstractValueShape, old_f_hash::UInt, new_f_hash::UInt)
    BispacedMeasure(
        unshaped(p.main, vs), p.transformed,
        _has_pair_annex(p) && p.f_hash == old_f_hash ? new_f_hash : UInt(0)
    )
end

ValueShapes.unshaped(em::EvaluatedMeasure) = unshaped(em, varshape(em))

# The domain of the cached transformation function moves along with the
# reparametrized main representation, its flat target space stays:
function _unshaped_f(f, intent::TransformIntent, vs::AbstractValueShape)
    if intent isa DoNotTransform
        identity
    elseif isnothing(f)
        nothing
    else
        ffcomp(f, inverse(Base.Fix2(unshaped, vs)))
    end
end


has_uhc_support(em::EvaluatedMeasure) = has_uhc_support(unevaluated(em))


# ToDo: truncate_batmeasure(em::EvaluatedMeasure, bounds::AbstractArray{<:Interval})

function MeasureBase.weightedmeasure(logweight::Real, em::EvaluatedMeasure)
    # approx captures the shape of the measure up to total mass, so it is
    # invariant under reweighting and kept as-is.
    # The transformed-space cache refers to the unweighted measure, so it
    # must not be carried over. transform_intent and the other transformed-space
    # content stay: the implied transform function does not change under
    # reweighting. samplegen survives on purpose as well, sample generation
    # only sees the normalized measure, which reweighting does not change:
    new_unevaluated = BispacedMeasure(weightedmeasure(logweight, unevaluated(em)))
    new_empirical = _renormalize_empirical(logweight, _empirical_rep(em))
    new_mass = _reweighted_mass(logweight, em.mass)
    return EvaluatedMeasure(
        new_unevaluated, em.transform_intent, em.f_transform, new_empirical, em.approx,
        em.dof, new_mass, em.modes, em.samplegen, nothing
    )
end

_renormalize_empirical(::Real, ::Nothing) = nothing
function _renormalize_empirical(logweight::Real, p::BispacedMeasure)
    BispacedMeasure(
        _renormalize_empirical_logd(logweight, p.main),
        isnothing(p.transformed) ? nothing : _renormalize_empirical_logd(logweight, p.transformed),
        p.f_hash
    )
end


"""
    BAT.validate_evalmeasure(
        em::EvaluatedMeasure;
        context::BATContext = get_batcontext()
    )::EvaluatedMeasure

*Experimental feature, not part of stable public API.*

Verify the transformed-space-view contract of `em` by re-deriving the view
from `(em.transform_intent, unevaluated(em))` and comparing values, and
spot-check the stored sample log-densities against the measure (and its
transformed representation). NaN sample log-densities are admissible (they
mark values lost in LADJ-less sample transport), and log-density checks
are skipped where the measure in question can't be point-evaluated.
Computationally expensive, intended for tests and debugging, not for
performance-critical code. Throws an exception if `em` violates its
contract (or if transformed-space content exists but no validation points
can be derived to verify it), returns `em` otherwise.
"""
function validate_evalmeasure(em::EvaluatedMeasure; context = get_batcontext())
    _em_sanity_checks(em)
    atol = rtol = √(eps(Float64))
    test_vs = _em_validation_points(em)

    # The sample log-densities of the empirical content must be the
    # log-densities of the measure itself:
    p_emp = _empirical_rep(em)
    if !isnothing(p_emp) && !(unevaluated(em) isa DensitySampleMeasure)
        smpls = samplesof(p_emp.main)
        for i in _em_spotcheck_idxs(smpls)
            logd_ref = _try_logdensityof(unevaluated(em), smpls.v[i])
            if !ismissing(logd_ref) && !isnan(smpls.logd[i])
                isapprox(smpls.logd[i], logd_ref, rtol = rtol, atol = atol) || throw(ArgumentError("Empirical content of EvaluatedMeasure carries sample log-densities that disagree with the log-density of the measure (sample $i)"))
            end
        end
    end

    intent = em.transform_intent
    if !(intent isa DoNotTransform)
        has_view_content = !isnothing(em.f_transform) || !isnothing(em.unevaluated.transformed) ||
            _has_pair_annex(em.empirical) || _has_pair_annex(em.approx)
        if has_view_content && isempty(test_vs)
            throw(ArgumentError("Can't derive validation points for EvaluatedMeasure, unable to verify its transformed-space content"))
        end

        m_z_fresh, f_fresh = transform_and_unshape(intent, unevaluated(em), context)
        m_z_fresh_uneval = unevaluated(m_z_fresh)

        # Value-compare the cached transformation function and measure by
        # application, function objects can't be compared reliably:
        f = em.f_transform
        if !isnothing(f)
            for v in test_vs
                isapprox(f(v), f_fresh(v), rtol = rtol, atol = atol) || throw(ArgumentError("Cached f_transform of EvaluatedMeasure disagrees with the transformation re-derived from its transform intent and measure"))
            end
        end

        cached_m_z = em.unevaluated.transformed
        if !isnothing(cached_m_z)
            getdof(cached_m_z) == getdof(m_z_fresh_uneval) || throw(ArgumentError("Transformed-space measure cache of EvaluatedMeasure has wrong degrees of freedom"))
            for v in test_vs
                z = f_fresh(v)
                isapprox(logdensityof(cached_m_z, z), logdensityof(m_z_fresh_uneval, z), rtol = rtol, atol = atol) || throw(ArgumentError("Transformed-space measure cache of EvaluatedMeasure disagrees with the measure re-derived from its transform intent and measure"))
            end
        end

        p = em.empirical
        if _has_pair_annex(p)
            smpls_main, smpls_z = samplesof(p.main), samplesof(p.transformed)
            for i in eachindex(smpls_main.v, smpls_z.v)
                isapprox(f_fresh(smpls_main.v[i]), smpls_z.v[i], rtol = rtol, atol = atol) || throw(ArgumentError("The sides of the empirical pair of an EvaluatedMeasure are not related by the transformation of its view (sample $i)"))
            end
            smpls_main.weight == smpls_z.weight || throw(ArgumentError("The sides of the empirical pair of an EvaluatedMeasure differ in sample weights"))
            # The transformed-side sample log-densities must be the
            # log-densities of the transformed measure:
            for i in _em_spotcheck_idxs(smpls_z)
                logd_ref = _try_logdensityof(m_z_fresh_uneval, smpls_z.v[i])
                if !ismissing(logd_ref) && !isnan(smpls_z.logd[i])
                    isapprox(smpls_z.logd[i], logd_ref, rtol = rtol, atol = atol) || throw(ArgumentError("The transformed-space side of the empirical pair of an EvaluatedMeasure carries sample log-densities that disagree with the transformed measure (sample $i)"))
                end
            end
        end

        pa = em.approx
        if _has_pair_annex(pa)
            # The transformed side of the approximation pair must be the
            # pushforward of its main side under the view's transformation:
            approx_z_fresh = unevaluated(bat_transform(f_fresh, batmeasure(pa.main), context).result)
            for v in test_vs
                z = f_fresh(v)
                logd_pair = _try_logdensityof(pa.transformed, z)
                logd_ref = _try_logdensityof(approx_z_fresh, z)
                if !ismissing(logd_pair) && !ismissing(logd_ref) && !(isnan(logd_pair) && isnan(logd_ref))
                    isapprox(logd_pair, logd_ref, rtol = rtol, atol = atol) || throw(ArgumentError("The sides of the approximation pair of an EvaluatedMeasure are not related by the transformation of its view"))
                end
            end
        end
    end
    return em
end

_em_spotcheck_idxs(smpls::DensitySampleVector) = firstindex(smpls.v):min(lastindex(smpls.v), firstindex(smpls.v) + 2)

_try_logdensityof(m, v) = try logdensityof(m, v) catch; missing end

# A few variate values of the measure, to compare transformations by
# application:
function _em_validation_points(em::EvaluatedMeasure)
    smpls = samplesof(em)
    if !isnothing(smpls) && !isempty(smpls)
        idxs = firstindex(smpls.v):(firstindex(smpls.v) + min(length(smpls), 3) - 1)
        return collect(view(smpls.v, idxs))
    else
        return try
            [MeasureBase.testvalue(unevaluated(em))]
        catch
            Vector{Any}()
        end
    end
end


function LazyReports.pushcontent!(rpt::LazyReport, em::EvaluatedMeasure)
    smpls = samplesof(em)
    isnothing(smpls) || lazyreport!(rpt, smpls)
    samplegen = samplegenof(em)
    isnothing(samplegen) || lazyreport!(rpt, samplegen)
    return nothing
end


function _empirical_or_unevaluated(em::EvaluatedMeasure)
    empirical = empiricalof(em)
    return !isnothing(empirical) ? empirical : unevaluated(em)
end


Statistics.mean(em::EvaluatedMeasure) = mean(_empirical_or_unevaluated(em))
Statistics.median(em::EvaluatedMeasure) = median(_empirical_or_unevaluated(em))
Statistics.var(em::EvaluatedMeasure) = var(_empirical_or_unevaluated(em))
Statistics.std(em::EvaluatedMeasure) = std(_empirical_or_unevaluated(em))
Statistics.cov(em::EvaluatedMeasure) = cov(_empirical_or_unevaluated(em))

_approx_mean(em::EvaluatedMeasure, n) = _approx_mean(_empirical_or_unevaluated(em), n)
_approx_cov(em::EvaluatedMeasure, n) = _approx_cov(_empirical_or_unevaluated(em), n)


function _approx_max_logd(em::EvaluatedMeasure)
    max_logd = _approx_max_logd(_empirical_or_unevaluated(em))
    @assert ismissing(max_logd) || !isnan(max_logd)
    return max_logd
end
