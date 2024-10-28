
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
    samples = ..., approx = ..., mass = ..., mode = ...,
    _generator = ...
)

BAT.unevaluated(em) === measure
```

!!! note

    Field `_generator` does not form part of the stable public
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
    samples::S
    approx::A
    mass::M
    modes::P
    _generator::G
end
export EvaluatedMeasure

function EvaluatedMeasure(
    measurelike::MeasureLike;
    samples = missing,
    approx = missing,
    mass = MeasureBase.UnknownMass(),
    modes = missing,
    _generator = missing
)
    measure = batmeasure(measurelike)
    @argcheck DensityKind(measure) isa HasDensity
    return EvaluatedMeasure(measure, samples, approx, mass, modes, _generator)
end

function EvaluatedMeasure(
    em::EvaluatedMeasure;
    samples = em.samples,
    approx = em.approx,
    mass = em.mass,
    modes = em.modes,
    _generator = em._generator
)
    @argcheck DensityKind(em) isa HasDensity
    return EvaluatedMeasure(em.measure, samples, approx, mass, modes, _generator)
end

unevaluated(em::EvaluatedMeasure) = em.measure

DensityInterface.logdensityof(em::EvaluatedMeasure, v::Any) = logdensityof(em.measure, v)
DensityInterface.logdensityof(em::EvaluatedMeasure) = logdensityof(em.measure)

MeasureBase.getdof(em::EvaluatedMeasure) = MeasureBase.getdof(em.measure)

MeasureBase.massof(em::EvaluatedMeasure) = em.mass

ValueShapes.varshape(em::EvaluatedMeasure) = varshape(em.measure)

function _unshaped_density(em::EvaluatedMeasure, vs::AbstractValueShape)
    new_measure = unshaped(em.measure, vs)
    @assert elshape(em.samples.v) <= vs
    new_samples = unshaped.(em.samples)
    return EvaluatedMeasure(new_measure, new_samples, em.approx, em.mass, em.modes, em._generator)
end

measure_support(em::EvaluatedMeasure) = measure_support(em.measure)

maybe_samplesof(em::EvaluatedMeasure) = em.samples
maybe_modesof(em::EvaluatedMeasure) = em.modes
maybe_approxof(em::EvaluatedMeasure) = em.approx
maybe_generator(em::EvaluatedMeasure) = em._generator


get_initsrc_from_target(em::EvaluatedMeasure) = em.samples


_get_deep_prior_for_trafo(em::EvaluatedMeasure) = _get_deep_prior_for_trafo(em.measure)

function bat_transform_impl(target::AbstractTransformTarget, em::EvaluatedMeasure, algorithm::PriorSubstitution, context::BATContext)
    new_measure, f_transform = bat_transform_impl(target, em.measure, algorithm, context)
    samples = em.samples
    smpl_trafoalg = bat_default(bat_transform, Val(:algorithm), f_transform, samples)
    new_samples, _ = bat_transform_impl(f_transform, samples, smpl_trafoalg, context)
    new_em = EvaluatedMeasure(new_measure, new_samples, em.approx, em.mass, em.modes, em._generator)
    (result = new_em, f_transform = f_transform)
end

# ToDo: truncate_batmeasure(em::EvaluatedMeasure, bounds::AbstractArray{<:Interval})

function MeasureBase.weightedmeasure(logweight::Real, em::EvaluatedMeasure)
    new_measure = weightedmeasure(logweight, em.measure)
    samples = em.samples
    new_samples = DensitySampleVector((samples.v, samples.logd .+ logweight, samples.weight, samples.info, samples.aux))
    return EvaluatedMeasure(new_measure, new_samples, em.approx, em.mass, em.modes, em._generator)
end


# function Base.show(io::IO, mime::MIME"text/plain", sd::EvaluatedMeasure)
#     if get(io, :compact, false)
#         println(io, "EvaluatedMeasure(...)")
#     else
#         # ...
#     end
# end


function bat_report!(md::Markdown.MD, em::EvaluatedMeasure)
    if !isnothing(em.samples)
        bat_report!(md, em.samples)
    end
    if !isnothing(em._generator)
        bat_report!(md, em._generator)
    end

    return md
end


function _approx_mean(em::EvaluatedMeasure, n)
    smpls = maybe_samplesof(em)
    if !isnothing(smpls)
        return mean(smpls)
    else
        return _approx_mean(unevaluated(em))
    end
end


function _approx_cov(em::EvaluatedMeasure, n)
    smpls = maybe_samplesof(em)
    if !isnothing(smpls)
        return cov(smpls)
    else
        return _approx_cov(unevaluated(em))
    end
end

function _estimated_max_logd(em::EvaluatedMeasure)
    smpls = maybe_samplesof(em)
    if !isnothing(smpls) && !any(isnan, smpls.logd)
        return _estimated_max_logd(smpls)
    else
        return _estimated_max_logd(unevaluated(em))
    end
end
