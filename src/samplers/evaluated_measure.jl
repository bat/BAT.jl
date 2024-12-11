# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    BAT.unevaluated(obj)

If `obj` is an evaluated object, like a [`EvaluatedMeasure`](@ref),
return the original (unevaluated) object. Otherwise, return `obj`.
"""
function unevaluated end

unevaluated(obj) = obj


"""
    struct EvaluatedMeasure <: BATMeasure

Combined a measure with samples, and other information on it.

Constructors:

```julia
em = EvaluatedMeasure(
    measure;
    samples = ..., empirical = ..., mass = ..., mode = ...,
    _samplegen = ...
)

BAT.unevaluated(em) === measure
```

!!! note

    Field `_samplegen` does not form part of the stable public
    API and is subject to change without deprecation.
"""
struct EvaluatedMeasure{
    D<:BATMeasure,
    S<:Union{DensitySampleVector,Missing},
    A<:Union{BATMeasure,Missing},
    M<:Union{Number,MeasureBase.UnknownMass},
    P<:Union{AbstractVector,Missing},
    G<:Union{AbstractSampleGenerator,Missing}
} <: BATMeasure
    measure::D
    empirical::S
    approx::A
    mass::M
    modes::P
    _samplegen::G
end
export EvaluatedMeasure

function EvaluatedMeasure(
    measurelike::MeasureLike;
    empirical = missing,
    approx = missing,
    mass = MeasureBase.UnknownMass(),
    modes = missing,
    _samplegen = missing
)
    measure = batmeasure(measurelike)
    @argcheck DensityKind(measure) isa HasDensity
    return EvaluatedMeasure(measure, empirical, approx, mass, modes, _samplegen)
end

function EvaluatedMeasure(
    em::EvaluatedMeasure;
    empirical = em.empirical,
    approx = em.approx,
    mass = em.mass,
    modes = em.modes,
    _samplegen = em._samplegen
)
    @argcheck DensityKind(em) isa HasDensity
    return EvaluatedMeasure(em.measure, convert(DensitySampleMeasure, empirical), approx, mass, modes, _samplegen)
end

unevaluated(em::EvaluatedMeasure) = em.measure

DensityInterface.logdensityof(em::EvaluatedMeasure, v::Any) = logdensityof(em.measure, v)
DensityInterface.logdensityof(em::EvaluatedMeasure) = logdensityof(em.measure)

MeasureBase.getdof(em::EvaluatedMeasure) = MeasureBase.getdof(em.measure)

MeasureBase.massof(em::EvaluatedMeasure) = em.mass

ValueShapes.varshape(em::EvaluatedMeasure) = varshape(em.measure)

function ValueShapes.unshaped(em::EvaluatedMeasure, vs::AbstractValueShape)
    new_measure = unshaped(em.measure, vs)
    @assert varshape(em.empirical) <= vs
    new_sampled = unshaped(em.empirical)
    return EvaluatedMeasure(new_measure, new_sampled, em.approx, em.mass, em.modes, em._samplegen)
end

measure_support(em::EvaluatedMeasure) = measure_support(em.measure)

@inline maybe_empiricalof(em::EvaluatedMeasure) = em.empirical
@inline maybe_samplesof(em::EvaluatedMeasure) = maybe_empiricalof(em) isa Missing ? missing : convert(DensitySamplesVector, em.empirical)
@inline maybe_modesof(em::EvaluatedMeasure) = em.modes
@inline maybe_approxof(em::EvaluatedMeasure) = em.approx
@inline maybe_samplegen(em::EvaluatedMeasure) = em._samplegen


get_initsrc_from_target(em::EvaluatedMeasure) = em.empirical


_get_deep_prior_for_trafo(em::EvaluatedMeasure) = _get_deep_prior_for_trafo(em.measure)

function bat_transform_impl(target::AbstractTransformTarget, em::EvaluatedMeasure, algorithm::PriorSubstitution, context::BATContext)
    new_measure, f_transform = bat_transform_impl(target, em.measure, algorithm, context)
    empirical = em.empirical
    smpl_trafoalg = bat_default(bat_transform, Val(:algorithm), f_transform, empirical)
    new_samples, _ = bat_transform_impl(f_transform, empirical, smpl_trafoalg, context)
    new_em = EvaluatedMeasure(new_measure, new_samples, em.approx, em.mass, em.modes, em._samplegen)
    (result = new_em, f_transform = f_transform)
end

# ToDo: truncate_batmeasure(em::EvaluatedMeasure, bounds::AbstractArray{<:Interval})

function MeasureBase.weightedmeasure(logweight::Real, em::EvaluatedMeasure)
    new_measure = weightedmeasure(logweight, em.measure)
    new_empirical = _renormalize_empirical_logd(logweight, maybe_empiricalof(em))
    return EvaluatedMeasure(new_measure, new_empirical, em.approx, em.mass, em.modes, em._samplegen)
end


# function Base.show(io::IO, mime::MIME"text/plain", sd::EvaluatedMeasure)
#     if get(io, :compact, false)
#         println(io, "EvaluatedMeasure(...)")
#     else
#         # ...
#     end
# end


function bat_report!(md::Markdown.MD, em::EvaluatedMeasure)
    bat_report!(md, maybe_samplesof(em))
    bat_report!(md, maybe_samplegen(em))
    return md
end


function _empirical_or_unevaluated(em::EvaluatedMeasure)
    empirical = maybe_empiricalof(em)
    return !isnothing(empirical) ? empirical : unevaluated(em)
end

_approx_mean(em::EvaluatedMeasure, n) = _approx_mean(_empirical_or_unevaluated(em))
_approx_cov(em::EvaluatedMeasure, n) = _approx_cov(_empirical_or_unevaluated(em))

function _estimated_max_logd(em::EvaluatedMeasure)
    _max_logd = _estimated_max_logd(_empirical_or_unevaluated(em))
    @assert !isnan(_max_logd)
    return _max_logd
end
