# This file is a part of BAT.jl, licensed under the MIT License (MIT).


const RandSampleable = Union{
    DistLikeDensity,
    MultivariateDistribution,
    Histogram
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
    )::DensitySampleVector

Draw `n` samples from `posterior`.

Returns a NamedTuple of the shape

```julia
(result = X::DensitySampleVector, ...)
```

Result properties not listed here are algorithm-specific and are not part
of the stable BAT API.


`posterior` may be a

* [`BAT.AbstractPosteriorDensity`](@ref)

* [`BAT.DistLikeDensity`](@ref)

* [`BAT.DensitySampleVector`](@ref)

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
    rng = bat_rng()
    bat_sample(rng, posterior, n; kwargs...)
end


@inline function bat_sample(
    posterior::AnyPosterior, n::AnyNSamples, algorithm::AbstractSamplingAlgorithm;
    kwargs...
)
    rng = bat_rng()
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
    RandSampling

Constructors:

    RandSampling()

Sample via `Random.rand`. Only supported for posteriors of type
`Distributions.MultivariateDistribution` and `BAT.DistLikeDensity`.
"""
struct RandSampling <: AbstractSamplingAlgorithm end
export RandSampling


default_sampling_algorithm(posterior::RandSampleable) = RandSampling()


function bat_sample(rng::AbstractRNG, posterior::RandSampleable, n::Integer, algorithm::RandSampling)
    vs = varshape(posterior)

    P = Vector{_default_PT}
    #P = ValueShapes.shaped_type(vs)

    shape = varshape(posterior)
    npar = totalndof(shape)
    unshaped_samples = DensitySampleVector{P,_default_LDT,_default_int_WT,Nothing,Nothing}(undef, n, npar)

    rand!(rng, sampler(posterior), flatview(unshaped_samples.v))
    let logd = unshaped_samples.logd, params = unshaped_samples.v
        @uviews logd .= logpdf.(Ref(posterior), params)
    end
    unshaped_samples.weight .= 1

    samples = shape.(unshaped_samples)
    (result = samples,)
end



"""
    RandResampling <: AbstractSamplingAlgorithm

Constructors:

    RandResampling()

Resamples from a given set of samples.
"""
struct RandResampling <: AbstractSamplingAlgorithm end
export RandResampling


function bat_sample(rng::AbstractRNG, posterior::DensitySampleVector, n::Integer, algorithm::RandResampling)
    orig_idxs = eachindex(posterior)
    weights = FrequencyWeights(float(posterior.weight))
    resampled_idxs = sample(orig_idxs, weights, n, replace=true, ordered=false)

    samples = posterior[resampled_idxs]
    samples.weight .= 1

    (result = samples,)
end



"""
OrderedResampling <: AbstractSamplingAlgorithm

Constructors:

    OrderedResampling()

    Efficiently resamples from a given series of samples, keeping the order of samples.

    Can be used to efficiently convert weighted samples into samples with uniform
"""
struct OrderedResampling <: AbstractSamplingAlgorithm end
export OrderedResampling


default_sampling_algorithm(posterior::DensitySampleVector) = OrderedResampling()


function bat_sample(rng::AbstractRNG, samples::DensitySampleVector, n::Integer, algorithm::OrderedResampling)
    @assert axes(samples) == axes(samples.weight)
    W = samples.weight
    idxs = eachindex(samples)

    resampled_idxs = Vector{Int}()
    sizehint!(resampled_idxs, n)

    p_factor = n / sum(W)

    for i in eachindex(W)
        w_eff_0 = p_factor * W[i]
        w_eff::typeof(w_eff_0) = w_eff_0
        while w_eff > 0
            rand(rng) < w_eff && push!(resampled_idxs, i)
            w_eff = w_eff - 1
        end
    end

    new_samples = samples[resampled_idxs]
    new_samples.weight .= 1

    (result = new_samples,)
end
