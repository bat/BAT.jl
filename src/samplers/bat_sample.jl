# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# when constructing a without generator infos like `EvaluatedMeasure(density, samples)`:
struct UnknownSampleGenerator<: AbstractSampleGenerator end
getproposal(sg::UnknownSampleGenerator) = nothing

# for samplers without specific infos, e.g. current ImportanceSamplers:
struct GenericSampleGenerator{A <: AbstractSamplingAlgorithm} <: AbstractSampleGenerator
    algorithm::A
end
getproposal(sg::GenericSampleGenerator) = sg.algorithm


function sample_and_verify(
    target::AnySampleable, samplingalg::AbstractSamplingAlgorithm,
    ref_dist::Distribution = target, context::BATContext = get_batcontext();
    max_retries::Integer = 1
)
    measure = batsampleable(target)
    initial_smplres = bat_sample_impl(measure, samplingalg, context)
    smplres::typeof(initial_smplres) = initial_smplres
    verified::Bool = test_dist_samples(ref_dist, smplres.result, context)
    n_retries::Int = 0
    while !(verified) && n_retries < max_retries
        n_retries += 1
        smplres = bat_sample_impl(measure, samplingalg, context)
        verified = test_dist_samples(ref_dist, smplres.result, context)
    end
    merge(smplres, (verified = verified, n_retries = n_retries))
end


"""
    struct IIDSampling <: AbstractSamplingAlgorithm

Sample via `Random.rand`.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct IIDSampling <: AbstractSamplingAlgorithm
    nsamples::Int = 10^5
end
export IIDSampling


function bat_sample_impl(m::BATMeasure, algorithm::IIDSampling, context::BATContext)
    #@assert false
    cunit = get_compute_unit(context)
    rng = get_rng(context)
    n = algorithm.nsamples

    v = rand(rng, m^n)
    # ToDo: Parallelize:
    logd = map(logdensityof(m), v)

    weight = adapt(cunit, fill(one(_IntWeightType), length(eachindex(logd))))
    info = adapt(cunit, fill(nothing, length(eachindex(logd))))
    aux = adapt(cunit, fill(nothing, length(eachindex(logd))))

    smpls = DensitySampleVector((v, logd, weight, info, aux))
    return (result = smpls,)
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


function bat_sample_impl(m::DensitySampleMeasure, algorithm::RandResampling, context::BATContext)
    new_smpls = _bat_rand_subsample_impl(m, samplesof(m), algorithm, context)
    (result = new_smpls,)
end

function bat_sample_impl(smpls::DensitySampleVector, algorithm::RandResampling, context::BATContext)
    new_smpls = _bat_rand_subsample_impl(smpls, smpls, algorithm, context)
    (result = new_smpls,)
end

function _bat_rand_subsample_impl(idxsrc, smpls, algorithm::RandResampling, context::BATContext)
    gen = get_gencontext(context)
    n = algorithm.nsamples
    idxs = _rand_subsample_idxs(gen, idxsrc, n)
    new_smpls = smpls[idxs]
    new_smpls.weight .= 1
    return new_smpls
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


function bat_sample_impl(m::DensitySampleMeasure, algorithm::OrderedResampling, context::BATContext)
    # ToDo: Use PSIS

    # ToDo: Utilize m._cw to speed up sampling:
    bat_sample_impl(getsamples(m), algorithm, context)
end

function bat_sample_impl(smpls::DensitySampleVector, algorithm::OrderedResampling, context::BATContext)
    # ToDo: Use PSIS

    rng = get_rng(context)
    @assert axes(smpls) == axes(smpls.weight)
    W = smpls.weight
    idxs = eachindex(smpls)

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

    new_samples = smpls[resampled_idxs]
    new_samples.weight .= 1

    (result = new_samples,)
end
