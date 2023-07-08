
"""
    struct EvaluatedMeasure <: BATMeasure

Stores a density and samples drawn from it.

A report on the density, samples, properties of variates/parameters, etc. can
be generated via `Base.show`.

Constructors:

* ```EvaluatedMeasure(density::AbstractPosteriorMeasure, samples::DensitySampleVector)```

Fields:

* ```density::BATMeasure```

* ```samples::DensitySamplesVector```

* ```_generator::AbstractSampleGenerator```

!!! note

    Field `_generator` do not form part of the stable public
    API and is subject to change without deprecation.
"""
struct EvaluatedMeasure{D<:BATMeasure,S<:DensitySampleVector, G<:AbstractSampleGenerator} <: BATMeasure
    density::D
    samples::S
    _generator::G
end
export EvaluatedMeasure


function EvaluatedMeasure(density::BATMeasure, samples::DensitySampleVector)
    return EvaluatedMeasure(density, samples, UnknownSampleGenerator())
end

function EvaluatedMeasure(density::EvaluatedMeasure, samples::DensitySampleVector)
    return EvaluatedMeasure(density.density, samples)
end

function EvaluatedMeasure(target::AnyMeasureOrDensity, samples::DensitySampleVector)
    @argcheck DensityKind(target) isa HasDensity
    return EvaluatedMeasure(convert(AbstractMeasureOrDensity, target), samples)
end

MeasureBase.getdof(m::EvaluatedMeasure) = MeasureBase.getdof(m.density)

eval_logval(density::EvaluatedMeasure, v::Any, T::Type{<:Real}) = eval_logval(density.density, v, T)

ValueShapes.varshape(density::EvaluatedMeasure) = varshape(density.density)

var_bounds(density::EvaluatedMeasure) = var_bounds(density.density)


# ToDo: Distributions.sampler(density::EvaluatedMeasure)
# ToDo: bat_sampler(density::EvaluatedMeasure)


get_initsrc_from_target(target::EvaluatedMeasure) = target.samples


_get_deep_prior_for_trafo(density::EvaluatedMeasure) = _get_deep_prior_for_trafo(density.density)

function bat_transform_impl(target::AbstractTransformTarget, density::EvaluatedMeasure, algorithm::PriorSubstitution, context::BATContext)
    new_parent_density, trafo = bat_transform_impl(target, density.density, algorithm, context)
    smpls = density.samples
    smpl_trafoalg = bat_default(bat_transform, Val(:algorithm), trafo, smpls)
    new_samples, _ = bat_transform_impl(trafo, smpls, smpl_trafoalg, context)
    (result = EvaluatedMeasure(new_parent_density, new_samples), trafo = trafo)
end

# ToDo: truncate_density(density::EvaluatedMeasure, bounds::AbstractArray{<:Interval})

_approx_cov(target::EvaluatedMeasure) = cov(target.samples)



# function Base.show(io::IO, mime::MIME"text/plain", sd::EvaluatedMeasure)
#     if get(io, :compact, false)
#         println(io, "EvaluatedMeasure(...)")
#     else
#         # ...
#     end
# end


function bat_report!(md::Markdown.MD, sd::EvaluatedMeasure)
    #bat_report!(md, sd.density)
    bat_report!(md, sd.samples)
    bat_report!(md, sd._generator)

    return md
end
