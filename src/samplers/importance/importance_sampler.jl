"""
    SobolSampler

Constructors:

    SobolSampler(nsamples::Int = 10^5)

Sample from Sobol sequence. Also see [Sobol.jl](https://github.com/stevengj/Sobol.jl).
"""
@with_kw struct SobolSampler <: AbstractSamplingAlgorithm
    nsamples::Int = 10^5
end
export SobolSampler



"""
    GridSampler

Constructors:

    GridSampler(ppa::Int = 100)

Sample from equidistantly distributed points in each dimension.
"""
@with_kw struct GridSampler <: AbstractSamplingAlgorithm
    ppa::Int = 100
end
export GridSampler


function bat_sample_impl(
    rng::AbstractRNG,
    density::AnyDensityLike,
    algorithm::Union{SobolSampler, GridSampler}
)
    shape = varshape(density)

    bounds = var_bounds(density)
    truncated_density = if isinf(bounds)
        TruncatedDensity(density, estimate_finite_bounds(density), 0)
    else
        TruncatedDensity(density, bounds, 0)
    end

    samples = _gen_samples(truncated_density, algorithm)

    logvals = eval_logval.(Ref(truncated_density), samples)
    weights = exp.(logvals)

    bat_samples = shape.(DensitySampleVector(samples, logvals, weight = weights))

    return (result = bat_samples, )
end


function _gen_samples(density::TruncatedDensity, algorithm::SobolSampler)
    bounds = var_bounds(density)
    sobol = Sobol.SobolSeq(bounds.vol.lo, bounds.vol.hi)
    p = vcat([[Sobol.next!(sobol)] for i in 1:algorithm.nsamples]...)
    return p
end


function _gen_samples(density::TruncatedDensity, algorithm::GridSampler)
    bounds = var_bounds(density)
    dim = length(bounds.bt)
    ppa = algorithm.ppa
    ranges = [range(bounds.vol.lo[i], bounds.vol.hi[i], length = trunc(Int, ppa)) for i in 1:dim]
    p = vec(collect(Iterators.product(ranges...)))
    return [collect(p[i]) for i in 1:length(p)]
end


"""
    PriorImportanceSampler

Constructors:

    PriorImportanceSampler(nsamples::Int = 10^5)

Sample randomly from prior distribution.
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
