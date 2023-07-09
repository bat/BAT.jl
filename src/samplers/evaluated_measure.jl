
"""
    struct EvaluatedMeasure <: BATMeasure

Combined a measure with samples, and other information on it.

Constructors:

```julia
EvaluatedMeasure(
    measure;
    samples = ..., approx = ..., mass = ..., mode = ...,
    _generator = ...
)
```

!!! note

    Field `_generator` does not form part of the stable public
    API and is subject to change without deprecation.
"""
struct EvaluatedMeasure{
    D<:AbstractMeasureOrDensity,
    S<:Union{DensitySampleVector,Nothing},
    A<:NamedTuple,
    M<:Number,
    P<:Union{DensitySampleVector,Nothing},
    G<:Union{AbstractSampleGenerator,Nothing}
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
    measurelike::AnyMeasureOrDensity;
    samples = nothing,
    approx = NamedTuple(),
    mass = MeasureBase.UnknownMass(),
    modes = nothing,
    _generator = nothing
)
    measure = convert(AbstractMeasureOrDensity, measurelike)
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

var_bounds(em::EvaluatedMeasure) = var_bounds(em.measure)


# ToDo: Distributions.sampler(em::EvaluatedMeasure)
# ToDo: bat_sampler(em::EvaluatedMeasure)


get_initsrc_from_target(em::EvaluatedMeasure) = em.samples


_get_deep_prior_for_trafo(em::EvaluatedMeasure) = _get_deep_prior_for_trafo(em.measure)

function bat_transform_impl(target::AbstractTransformTarget, em::EvaluatedMeasure, algorithm::PriorSubstitution, context::BATContext)
    new_measure, trafo = bat_transform_impl(target, em.measure, algorithm, context)
    samples = em.samples
    smpl_trafoalg = bat_default(bat_transform, Val(:algorithm), trafo, samples)
    new_samples, _ = bat_transform_impl(trafo, samples, smpl_trafoalg, context)
    new_em = EvaluatedMeasure(new_measure, new_samples, em.approx, em.mass, em.modes, em._generator)
    (result = new_em, trafo = trafo)
end

# ToDo: truncate_density(em::EvaluatedMeasure, bounds::AbstractArray{<:Interval})

_approx_cov(em::EvaluatedMeasure) = cov(em.samples)

function renormalize_density(em::EvaluatedMeasure, logrenormf::Real)
    new_measure = renormalize_density(em.measure, logrenormf)
    samples = em.samples
    new_samples = DensitySampleVector((samples.v, samples.logd .+ logrenormf, samples.weight, samples.info, samples.aux))
    return EvaluatedMeasure(new_measure, new_samples, em.approx, em.mass, em.modes, em._generator)
end

_estimated_max_logd(measure::EvaluatedMeasure) = _estimated_max_logd(something(measure.modes, measure.samples, missing))

# function Base.show(io::IO, mime::MIME"text/plain", sd::EvaluatedMeasure)
#     if get(io, :compact, false)
#         println(io, "EvaluatedMeasure(...)")
#     else
#         # ...
#     end
# end


function bat_report!(md::Markdown.MD, sd::EvaluatedMeasure)
    #bat_report!(md, sd.measure)
    bat_report!(md, sd.samples)
    bat_report!(md, sd._generator)

    return md
end
