abstract type ImportanceSampler <: AbstractSamplingAlgorithm end

"""
    SobolSampler

Constructors:

    SobolSampler()

Sample from Sobol sequence. Also see [Sobol.jl](https://github.com/stevengj/Sobol.jl).
"""
struct SobolSampler <: ImportanceSampler end
export SobolSampler



"""
    GridSampler

Constructors:

    GridSampler()

Sample from equidistantly distributed points in each dimension.
"""
struct GridSampler <: ImportanceSampler end
export GridSampler


function bat_sample_impl(
    rng::AbstractRNG,
    density::AnyDensityLike,
    n::Integer,
    algorithm::ImportanceSampler
)
    shape = varshape(density)

    bounds = var_bounds(density)
    truncated_density = if isinf(bounds)
        TruncatedDensity(density, estimate_finite_bounds(density), 0)
    else
        TruncatedDensity(density, bounds, 0)
    end

    samples = _gen_samples(algorithm, n, truncated_density)

    logvals = eval_logval.(Ref(truncated_density), samples)
    weights = exp.(logvals)

    bat_samples = shape.(DensitySampleVector(samples, logvals, weight = weights))

    return (result = bat_samples, )
end


function _gen_samples(algorithm::SobolSampler, n::Integer, density::TruncatedDensity)
    bounds = var_bounds(density)
    sobol = Sobol.SobolSeq(bounds.vol.lo, bounds.vol.hi)
    p = vcat([[Sobol.next!(sobol)] for i in 1:n]...)
    return p
end


function _gen_samples(algorith::GridSampler, n::Integer, density::TruncatedDensity)
    bounds = var_bounds(density)
    dim = length(bounds.bt)
    ppa = n^(1/dim)
    ranges = [range(bounds.vol.lo[i], bounds.vol.hi[i], length = trunc(Int, ppa)) for i in 1:dim]
    p = vec(collect(Iterators.product(ranges...)))
    return [collect(p[i]) for i in 1:length(p)]
end


"""
    PriorImportanceSampler

Constructors:

    PriorImportanceSampler()

Sample randomly from prior distribution.
"""
struct PriorImportanceSampler <: AbstractSamplingAlgorithm end
export PriorImportanceSampler

function bat_sample_impl(
    rng::AbstractRNG,
    posterior::AbstractPosteriorDensity,
    n::AnyNSamples,
    algorithm::PriorImportanceSampler
)
    shape = varshape(posterior)

    prior = getprior(posterior)
    priorsmpl = bat_sample(prior, n)
    unshaped_prior_samples = unshaped.(priorsmpl.result)

    v = unshaped_prior_samples.v
    prior_weight = unshaped_prior_samples.weight
    posterior_logd = eval_logval.(Ref(posterior), v)
    weight = exp.(posterior_logd - unshaped_prior_samples.logd) .* prior_weight

    posterior_samples = shape.(DensitySampleVector(v, posterior_logd, weight = weight))
    priorsmpl = bat_sample(prior, n)

    return (result = posterior_samples, priorsmpl = priorsmpl)
end
