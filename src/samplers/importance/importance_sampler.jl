"""
    struct SobolSampler <: AbstractSamplingAlgorithm

*Experimental feature, not part of stable public API.*

Sample from Sobol sequence. Also see [Sobol.jl](https://github.com/stevengj/Sobol.jl).

Constructors:

* ```SobolSampler(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct SobolSampler{TR<:AbstractDensityTransformTarget} <: AbstractSamplingAlgorithm
    trafo::TR = PriorToUniform()
    nsamples::Int = 10^5
end
export SobolSampler



"""
    struct GridSampler <: AbstractSamplingAlgorithm

*Experimental feature, not part of stable public API.*

Sample from equidistantly distributed points in each dimension.

Constructors:

* ```GridSampler(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct GridSampler{TR<:AbstractDensityTransformTarget} <: AbstractSamplingAlgorithm
    trafo::TR = PriorToUniform()
    ppa::Int = 100
end
export GridSampler


function bat_sample_impl(
    rng::AbstractRNG,
    target::AnyDensityLike,
    algorithm::Union{SobolSampler, GridSampler}
)
    density_notrafo = convert(AbstractDensity, target)
    density, trafo = bat_transform(algorithm.trafo, density_notrafo)
    shape = varshape(density)

    samples = _gen_samples(density, algorithm)

    logvals = eval_logval.(Ref(density), samples)
    weights = exp.(logvals)

    samples_trafo = shape.(DensitySampleVector(samples, logvals, weight = weights))
    samples_notrafo = inv(trafo).(samples_trafo)

    return (result = samples_notrafo, result_trafo = samples_trafo, trafo = trafo)
end


function _gen_samples(density::AbstractDensity, algorithm::SobolSampler)
    bounds = var_bounds(density)
    isinf(bounds) && throw(ArgumentError("SobolSampler doesn't support densities with infinite support"))
    sobol = Sobol.SobolSeq(bounds.vol.lo, bounds.vol.hi)
    p = vcat([[Sobol.next!(sobol)] for i in 1:algorithm.nsamples]...)
    return p
end


function _gen_samples(density::AbstractDensity, algorithm::GridSampler)
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

* ```PriorImportanceSampler(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct PriorImportanceSampler <: AbstractSamplingAlgorithm
    nsamples::Int = 10^5
end
export PriorImportanceSampler

function bat_sample_impl(
    rng::AbstractRNG,
    posterior::AbstractPosteriorDensity,
    algorithm::PriorImportanceSampler
)
    shape = varshape(posterior)

    prior = getprior(posterior)
    prior_samples = bat_sample(prior, IIDSampling(nsamples = algorithm.nsamples)).result
    unshaped_prior_samples = unshaped.(prior_samples)

    v = unshaped_prior_samples.v
    prior_weight = unshaped_prior_samples.weight
    posterior_logd = eval_logval.(Ref(posterior), v)
    weight = exp.(posterior_logd - unshaped_prior_samples.logd) .* prior_weight

    posterior_samples = shape.(DensitySampleVector(v, posterior_logd, weight = weight))

    return (result = posterior_samples, prior_samples = prior_samples)
end
