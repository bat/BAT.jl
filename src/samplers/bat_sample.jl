# This file is a part of BAT.jl, licensed under the MIT License (MIT).

const _default_PT = Float32 # Default type for parameter values
const _default_float_WT = Float64 # Default type for float weights
const _default_int_WT = Int # Default type for int weights
const _default_LDT = Float64 # Default type for log-density values


const RandSampleable = Union{
    DistLikeDensity,
    MultivariateDistribution,
}


const AnyPosterior = Union{
    PosteriorDensity,
    PosteriorSampleVector,
    RandSampleable,
}


const AnyNSamples = Union{
    Integer,
    Tuple{Integer,Integer},
}



"""
    BAT.AbstractSamplingAlgorithm

Abstract type for BAT sampling algorithms. See [`bat_sample`](@ref).
"""
abstract type AbstractSamplingAlgorithm end


"""
    BAT.default_sampling_algorithm(posterior)

Get BAT's default sampling algorithm for `posterior`.
"""
function default_sampling_algorithm end


"""
    bat_sample(
        [rng::AbstractRNG],
        posterior::BAT.AnyPosterior,
        n::BAT.AnyNSamples,
        [algorithm::BAT.AbstractSamplingAlgorithm]
    )::PosteriorSampleVector

Draw `n` samples from `posterior`.

Returns a NamedTuple of the shape

```julia
(
    samples = X::PosteriorSampleVector,...
    stats = s::@test stats isa NamedTuple{(:mode,:mean,:cov,...)},
    ...
)
```

Result properties not listed here are algorithm-specific and are not part
of the stable BAT API.


`posterior` may be a

* [`BAT.AbstractPosteriorDensity`](@ref)

* [`BAT.DistLikeDensity`](@ref)

* [`BAT.PosteriorSampleVector`](@ref)

* `Distributions.MultivariateDistribution`

Depending on the type of posterior, `n` may be of type

* `Integer`: Number of samples

* `Tuple{Integer,Integer}`: Tuple of number of samples per sample source
   and number of sample sources (e.g. number of MCMC chains). The total number
   of samples is `product(n)`.

Depending on the type of `posterior`, the number of samples returned may be
somewhat larger or smaller than specified by `product(n)`.

Also depending on the `posterior` type, the samples may be independent or
correlated (e.g. when using MCMC).
"""
function bat_sample end
export bat_sample


@inline function bat_sample(
    posterior::AnyPosterior, n::AnyNSamples;
    kwargs...
)
    rng = bat_default_rng()
    bat_sample(rng, posterior, n; kwargs...)
end


@inline function bat_sample(
    posterior::AnyPosterior, n::AnyNSamples, algorithm::AbstractSamplingAlgorithm;
    kwargs...
)
    rng = bat_default_rng()
    bat_sample(rng, posterior, n, algorithm; kwargs...)
end


@inline function bat_sample(
    rng::AbstractRNG, posterior::AnyPosterior, n::AnyNSamples;
    kwargs...
)
    algorithm = default_sampling_algorithm(posterior)
    bat_sample(rng, posterior, n, algorithm; kwargs...)
end



"""
    BAT.RandSampling

Constructors:

    BAT.RandSampling()

Sample via `Random.rand`. Only supported for posteriors of type
`Distributions.MultivariateDistribution` and `BAT.DistLikeDensity`.
"""
struct RandSampling <: AbstractSamplingAlgorithm end


default_sampling_algorithm(posterior::RandSampleable) = RandSampling()


function bat_sample(rng::AbstractRNG, posterior::RandSampleable, n::Integer, algorithm::RandSampling)
    npar = length(posterior)
    samples = PosteriorSampleVector{_default_PT,_default_LDT,_default_int_WT,Nothing}(undef, n, npar)

    rand!(rng, sampler(posterior), flatview(samples.params))
    let log_posterior = samples.log_posterior, params = samples.params
        @uviews log_posterior .= logpdf.(Ref(posterior), params)
    end
    samples.log_prior .= 0
    samples.weight .= 1
    
    stats = bat_stats(samples)

    (samples = samples, stats = stats)
end



"""
    BAT.RandomResampling

Constructors:

    BAT.RandomResampling()

Resample from a given set of samples.
"""
struct RandomResampling <: AbstractSamplingAlgorithm end


default_sampling_algorithm(posterior::PosteriorSampleVector) = RandomResampling()


function bat_sample(rng::AbstractRNG, posterior::PosteriorSampleVector, n::Integer, algorithm::RandomResampling)
    orig_idxs = eachindex(posterior)
    weights = FrequencyWeights(float(posterior.weight))
    resampled_idxs = sample(orig_idxs, weights, n, replace=true, ordered=false)

    samples = posterior[resampled_idxs]
    samples.weight .= 1

    stats = bat_stats(samples)

    (samples = samples, stats = stats)
end
