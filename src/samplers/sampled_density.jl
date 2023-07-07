
"""
    struct SampledMeasure <: BATMeasure

Stores a density and samples drawn from it.

A report on the density, samples, properties of variates/parameters, etc. can
be generated via `Base.show`.

Constructors:

* ```SampledMeasure(density::AbstractPosteriorMeasure, samples::DensitySampleVector)```

Fields:

* ```density::BATMeasure```

* ```samples::DensitySamplesVector```

* ```_generator::AbstractSampleGenerator```

!!! note

    Field `_generator` do not form part of the stable public
    API and is subject to change without deprecation.
"""
struct SampledMeasure{D<:BATMeasure,S<:DensitySampleVector, G<:AbstractSampleGenerator} <: BATMeasure
    density::D
    samples::S
    _generator::G
end
export SampledMeasure


function SampledMeasure(density::BATMeasure, samples::DensitySampleVector)
    return SampledMeasure(density, samples, UnknownSampleGenerator())
end

function SampledMeasure(density::SampledMeasure, samples::DensitySampleVector)
    return SampledMeasure(density.density, samples)
end

function SampledMeasure(target::AnyMeasureLike, samples::DensitySampleVector)
    @argcheck DensityKind(target) isa HasDensity
    return SampledMeasure(convert(BATMeasure, target), samples)
end

MeasureBase.getdof(m::SampledMeasure) = MeasureBase.getdof(m.density)

eval_logval(density::SampledMeasure, v::Any, T::Type{<:Real}) = eval_logval(density.density, v, T)

ValueShapes.varshape(density::SampledMeasure) = varshape(density.density)

var_bounds(density::SampledMeasure) = var_bounds(density.density)


# ToDo: Distributions.sampler(density::SampledMeasure)
# ToDo: bat_sampler(density::SampledMeasure)


get_initsrc_from_target(target::SampledMeasure) = target.samples


_get_deep_transformable_base(density::SampledMeasure) = _get_deep_transformable_base(density.density)

function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, density::SampledMeasure, algorithm::PriorSubstitution)
    new_parent_density, trafo = bat_transform_impl(target, density.density, algorithm)
    new_samples = trafo.(density.samples)
    (result = SampledMeasure(new_parent_density, new_samples), trafo = trafo)
end

# ToDo: truncate_density(density::SampledMeasure, bounds::AbstractArray{<:Interval})

_approx_cov(target::SampledMeasure) = cov(target.samples)



# function Base.show(io::IO, mime::MIME"text/plain", sd::SampledMeasure)
#     if get(io, :compact, false)
#         println(io, "SampledMeasure(...)")
#     else
#         # ...
#     end
# end


function bat_report!(md::Markdown.MD, sd::SampledMeasure)
    #bat_report!(md, sd.density)
    bat_report!(md, sd.samples)
    bat_report!(md, sd._generator)

    return md
end
