abstract type ImportanceSampler <: AbstractSamplingAlgorithm end
export SobolSampler, GridSampler, PriorImportanceSampler


"""
    SobolSampler

Constructors:

    SobolSampler()

Sample from Sobol sequence. Also see [Sobol.jl](https://github.com/stevengj/Sobol.jl).
"""
struct SobolSampler <: ImportanceSampler end


"""
    GridSampler

Constructors:

    GridSampler()

Sample from equidistantly distributed points in each dimension.
"""
struct GridSampler <: ImportanceSampler end


function bat_sample_impl(
    rng::AbstractRNG,
    density::AnyPosterior,
    n::AnyNSamples,
    algorithm::ImportanceSampler
)
    shape = varshape(density)

    bounds = var_bounds(density)
    truncated_density = if isinf(bounds)
        TruncatedDensity(density, estimate_finite_bounds(density), 0)
    else
        TruncatedDensity(density, bounds, 0)
    end

    n_samples = isa(n, Tuple{Integer,Integer}) ? n[1] * n[2] : n[1]

    @info "Generating $n_samples samples with $(string(algorithm))."
    samples = _gen_samples(algorithm, n_samples, truncated_density)

    logvals = density_logval.(Ref(truncated_density), samples)
    weights = exp.(logvals)

    bat_samples = shape.(DensitySampleVector(samples, logvals, weight = weights))
    return (result = bat_samples,)
end


function _gen_samples(algorithm::SobolSampler, n_samples::Int, density::TruncatedDensity)
    bounds = var_bounds(density)
    sobol = Sobol.SobolSeq(bounds.vol.lo, bounds.vol.hi)
    p = vcat([[Sobol.next!(sobol)] for i in 1:n_samples]...)
    return p
end


function _gen_samples(algorith::GridSampler, n_samples::Int, density::TruncatedDensity)
    bounds = var_bounds(density)
    dim = length(bounds.bt)
    ppa = n_samples^(1/dim)
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

function bat_sample_impl(
    rng::AbstractRNG,
    posterior::AbstractPosteriorDensity,
    n::AnyNSamples,
    algorithm::PriorImportanceSampler
)
    shape = varshape(posterior)
    n_samples = isa(n, Tuple{Integer,Integer}) ? n[1] * n[2] : n[1]

    @info "Generating $n_samples samples with $(string(algorithm))."
    samples = _gen_samples(algorithm, n_samples, posterior)

    logvals = density_logval.(Ref(posterior), samples)
    logpriorvals = density_logval.(Ref(getprior(posterior)), samples)
    weights = exp.(logvals-logpriorvals)

    bat_samples = shape.(DensitySampleVector(samples, logvals, weight = weights))
    return (result = bat_samples,)
end

function _gen_samples(algorithm::PriorImportanceSampler, n_samples::Int, posterior::AnyPosterior)
    p = rand(getprior(posterior).dist, n_samples)
    return collect(eachcol(p))
end
