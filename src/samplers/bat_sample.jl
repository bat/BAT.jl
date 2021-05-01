# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# when constructing a without generator infos like `SampledDensity(density, samples)`:
struct UnknownSampleGenerator<: AbstractSampleGenerator end
getalgorithm(sg::UnknownSampleGenerator) = nothing

# for samplers without specific infos, e.g. current ImportanceSamplers:
struct GenericSampleGenerator{A <: AbstractSamplingAlgorithm} <: AbstractSampleGenerator
    algorithm::A
end
getalgorithm(sg::GenericSampleGenerator) = sg.algorithm



"""
    struct IIDSampling <: AbstractSamplingAlgorithm

Sample via `Random.rand`. Only supported for posteriors of type
`Distributions.MultivariateDistribution` and `BAT.DistLikeDensity`.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct IIDSampling <: AbstractSamplingAlgorithm
    nsamples::Int = 10^5
end
export IIDSampling


function bat_sample_impl(rng::AbstractRNG, target::AnyIIDSampleable, algorithm::IIDSampling)
    shaped_density = convert(DistLikeDensity, target)
    density = unshaped(shaped_density)
    shape = varshape(shaped_density)
    n = algorithm.nsamples

    # ToDo: Parallelize, using hierarchical RNG (separate RNG for each sample)
    v = nestedview(rand(rng, sampler(density), n))
    logd = map(logdensityof(density), v)

    weight = fill(_default_int_WT(1), length(eachindex(logd)))
    info = fill(nothing, length(eachindex(logd)))
    aux = fill(nothing, length(eachindex(logd)))

    unshaped_samples = DensitySampleVector((v, logd, weight, info, aux))

    samples = shape.(unshaped_samples)
    (result = samples,)
end


"""
    struct RandResampling <: AbstractSamplingAlgorithm

Resamples from a given set of samples.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct RandResampling <: AbstractSamplingAlgorithm
    nsamples::Int = 10^5
end
export RandResampling


function bat_sample_impl(rng::AbstractRNG, posterior::DensitySampleVector, algorithm::RandResampling)
    n = algorithm.nsamples
    orig_idxs = eachindex(posterior)
    weights = FrequencyWeights(float(posterior.weight))
    resampled_idxs = sample(orig_idxs, weights, n, replace=true, ordered=false)

    samples = posterior[resampled_idxs]
    samples.weight .= 1

    (result = samples,)
end



"""
    struct OrderedResampling <: AbstractSamplingAlgorithm

Efficiently resamples from a given series of samples, keeping the order of
samples.

Can be used to efficiently convert weighted samples into samples with unity
weights.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct OrderedResampling <: AbstractSamplingAlgorithm
    nsamples::Int = 10^5
end
export OrderedResampling


function bat_sample_impl(rng::AbstractRNG, samples::DensitySampleVector, algorithm::OrderedResampling)
    @assert axes(samples) == axes(samples.weight)
    W = samples.weight
    idxs = eachindex(samples)

    n = algorithm.nsamples
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
