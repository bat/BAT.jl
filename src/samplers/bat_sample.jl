# This file is a part of BAT.jl, licensed under the MIT License (MIT).


const RandSampleable = Union{
    DistLikeDensity,
    MultivariateDistribution,
    Histogram
}


"""
    RandSampling

Constructors:

    RandSampling()

Sample via `Random.rand`. Only supported for posteriors of type
`Distributions.MultivariateDistribution` and `BAT.DistLikeDensity`.
"""
struct RandSampling <: AbstractSamplingAlgorithm end
export RandSampling


function bat_sample_impl(rng::AbstractRNG, posterior::RandSampleable, n::Integer, algorithm::RandSampling)
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


function bat_sample_impl(rng::AbstractRNG, posterior::DensitySampleVector, n::Integer, algorithm::RandResampling)
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


function bat_sample_impl(rng::AbstractRNG, samples::DensitySampleVector, n::Integer, algorithm::OrderedResampling)
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
