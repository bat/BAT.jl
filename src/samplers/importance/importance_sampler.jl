"""
    struct SobolSampler <: AbstractSamplingAlgorithm

*Experimental feature, not part of stable public API.*

Sample from Sobol sequence. Also see [Sobol.jl](https://github.com/stevengj/Sobol.jl).

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct SobolSampler{TR<:AbstractTransformTarget} <: AbstractSamplingAlgorithm
    trafo::TR = PriorToUniform()
    nsamples::Int = 10^5
end
export SobolSampler



"""
    struct GridSampler <: AbstractSamplingAlgorithm

*Experimental feature, not part of stable public API.*

Sample from equidistantly distributed points in each dimension.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct GridSampler{TR<:AbstractTransformTarget} <: AbstractSamplingAlgorithm
    trafo::TR = PriorToUniform()
    ppa::Int = 100
end
export GridSampler


function bat_sample_impl(
    target::AnyMeasureLike,
    algorithm::Union{SobolSampler, GridSampler},
    context::BATContext
)
    orig_measure = convert(BATMeasure, target)
    transformed_measure, trafo = transform_and_unshape(algorithm.trafo, orig_measure)
    shape = varshape(transformed_measure)

    samples = _gen_samples(transformed_measure, algorithm)

    logvals = map(logdensityof(transformed_measure), samples)
    weights = exp.(logvals)

    vol = exp(BigFloat(log_volume(spatialvolume(var_bounds(transformed_measure)))))
    est_integral = mean(weights) * vol
    # ToDo: Add integral error estimate

    samples_trafo = shape.(DensitySampleVector(samples, logvals, weight = weights))
    samples_notrafo = inverse(trafo).(samples_trafo)

    return (result = samples_notrafo, result_trafo = samples_trafo, trafo = trafo, integral = est_integral)
end


function _gen_samples(density::AbstractMeasureOrDensity, algorithm::SobolSampler)
    bounds = var_bounds(density)
    isinf(bounds) && throw(ArgumentError("SobolSampler doesn't support densities with infinite support"))
    sobol = Sobol.SobolSeq(bounds.vol.lo, bounds.vol.hi)
    p = vcat([[Sobol.next!(sobol)] for i in 1:algorithm.nsamples]...)
    return p
end


function _gen_samples(density::AbstractMeasureOrDensity, algorithm::GridSampler)
    bounds = var_bounds(density)
    isinf(bounds) && throw(ArgumentError("SobolSampler doesn't support densities with infinite support"))
    dim = totalndof(density)
    ppa = algorithm.ppa
    ranges = [range(bounds.vol.lo[i], bounds.vol.hi[i], length = trunc(Int, ppa)) for i in 1:dim]
    p = vec(collect(Iterators.product(ranges...)))
    return [collect(p[i]) for i in 1:length(p)]
end


"""
    struct PriorImportanceSampler <: AbstractSamplingAlgorithm

*Experimental feature, not part of stable public API.*

Importance sampler using IID samples from the prior.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct PriorImportanceSampler <: AbstractSamplingAlgorithm
    nsamples::Int = 10^5
end
export PriorImportanceSampler

function bat_sample_impl(
    posterior::AbstractPosteriorMeasure,
    algorithm::PriorImportanceSampler,
    context::BATContext
)
    shape = varshape(posterior)

    prior = getprior(posterior)
    prior_samples = bat_sample(prior, IIDSampling(nsamples = algorithm.nsamples), context).result
    unshaped_prior_samples = unshaped.(prior_samples)

    v = unshaped_prior_samples.v
    prior_weight = unshaped_prior_samples.weight
    posterior_logd = map(logdensityof(unshaped(posterior)), v)
    weight = exp.(posterior_logd - unshaped_prior_samples.logd) .* prior_weight

    est_integral = mean(weight)
    # ToDo: Add integral error estimate

    posterior_samples = shape.(DensitySampleVector(v, posterior_logd, weight = weight))

    return (result = posterior_samples, prior_samples = prior_samples, integral = est_integral)
end
