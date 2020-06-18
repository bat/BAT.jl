abstract type ImportanceSampler <: AbstractSamplingAlgorithm end
export SobolSampler, GridSampler, PriorImportanceSampler


struct SobolSampler <: ImportanceSampler end
struct GridSampler <: ImportanceSampler end


function bat_sample(
    posterior::AnyPosterior,
    n::AnyNSamples,
    algorithm::ImportanceSampler;
    bounds::Any = var_bounds(posterior)
)
    shape = varshape(posterior)

    truncated_posterior = if isinf(bounds)
        TruncatedDensity(posterior, estimate_finite_bounds(posterior))
    else
        TruncatedDensity(posterior, bounds)
    end

    n_samples = isa(n, Tuple{Integer,Integer}) ? n[1] * n[2] : n[1]

    samples = get_samples(algorithm, n_samples, truncated_posterior)
    stats = [(stat = nothing, ) for i in n_samples] # TODO

    logvals = density_logval.(Ref(truncated_posterior), samples)
    weights = exp.(logvals)

    bat_samples = shape.(DensitySampleVector(samples, logval = logvals, weight = weights))
    return (result = bat_samples, chains = stats)
end


"""
    get_samples(
        algorithm::SobolSampler,
        n_samples::Int,
        posterior::TruncatedDensity
    )

Get `n_samples` calculated with the Sobol sequence in between the lower and
upper bound of `posterior.bounds.vol` for each dimension in volume vol.

Returns a Vector{Array{Float64, 1}} of length `n_samples`.

# Arguments
- `n_samples::Int`: number of samples
"""
function get_samples(algorithm::SobolSampler, n_samples::Int, posterior::TruncatedDensity)
    sobol = SobolSeq(posterior.bounds.vol.lo, posterior.bounds.vol.hi)
    p = vcat([[Sobol.next!(sobol)] for i in 1:n_samples]...)
    return p
end


"""
    get_samples(
        algorithm::GridSampler,
        n_samples::Int,
        posterior::TruncatedDensity
    )

Get `n_samples` distributed equidistantly on each axis with distance `ppa`
in the `dim`-dimensional volume.

Returns a Vector{Array{Float64, 1}} of length `n_samples`.

# Arguments
- `n_samples::Int`: number of samples
"""
function get_samples(algorith::GridSampler, n_samples::Int, posterior::TruncatedDensity)
    bounds = posterior.bounds
    dim = length(bounds.bt)
    ppa = n_samples^(1/dim)
    ranges = [range(bounds.vol.lo[i], bounds.vol.hi[i], length = trunc(Int, ppa)) for i in 1:dim]
    p = vec(collect(Iterators.product(ranges...)))
    return [collect(p[i]) for i in 1:length(p)]
end



struct PriorImportanceSampler <: AbstractSamplingAlgorithm end

function bat_sample(
    posterior::AnyPosterior,
    n::AnyNSamples,
    algorithm::PriorImportanceSampler
)
    shape = varshape(posterior)
    n_samples = isa(n, Tuple{Integer,Integer}) ? n[1] * n[2] : n[1]

    samples = get_samples(algorithm, n_samples, posterior)
    stats = [(stat = nothing, ) for i in n_samples] # TODO

    logvals = density_logval.(Ref(posterior), samples)
    logpriorvals = density_logval.(Ref(getprior(posterior)), samples)
    weights = exp.(logvals-logpriorvals)

    bat_samples = shape.(DensitySampleVector(samples, logval = logvals, weight = weights))
    return (result = bat_samples, chains = stats)
end

function bat_sample_(
    posterior::AnyPosterior,
    n::AnyNSamples,
    algorithm::PriorImportanceSampler
)
    shape = varshape(posterior)
    n_samples = isa(n, Tuple{Integer,Integer}) ? n[1] * n[2] : n[1]

    samples = get_samples(algorithm, n_samples, posterior)
    stats = [(stat = nothing, ) for i in n_samples] # TODO

    logvals = density_logval.(Ref(posterior), samples)
    weights = exp.(logvals)

    bat_samples = shape.(DensitySampleVector(samples, logval = logvals, weight = weights))
    return (result = bat_samples, chains = stats)
end


"""
    get_samples(
        algorithm::PriorImportanceSampler,
        n_samples::Int,
        posterior::AnyPosterior
    )

Get `n_samples` randomly chosen from the prior distribution.

Returns a Vector{Array{Float64, 1}} of length `n_samples`.

`posterior` may be a

* [`BAT.AbstractPosteriorDensity`](@ref)

* [`BAT.DistLikeDensity`](@ref)

* [`BAT.DensitySampleVector`](@ref)

* `Distributions.MultivariateDistribution`

# Arguments
- `n_samples::Int`: number of samples
"""
function get_samples(algorithm::PriorImportanceSampler, n_samples::Int, posterior::AnyPosterior)
    p = rand(getprior(posterior).dist, n_samples)
    return collect(eachcol(p))
end
