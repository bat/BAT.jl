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
    measure = convert_for(bat_sample, target)
    initial_smplres = bat_sample(measure, samplingalg, context)
    smplres::typeof(initial_smplres) = initial_smplres
    verified::Bool = test_dist_samples(ref_dist, smplres.result, context)
    n_retries::Int = 0
    while !(verified) && n_retries < max_retries
        n_retries += 1
        smplres = bat_sample(measure, samplingalg, context)
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


function evalmeasure_impl(measure::BATMeasure, algorithm::IIDSampling, context::BATContext)
    m = unevaluated(measure)
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
    # ToDo: Get DOF
    dsm = DensitySampleMeasure(smpls, ess = length(smpls))
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

function evalmeasure_impl(m::BATMeasure, algorithm::RandResampling, context::BATContext)
    dsm = getempirical(m)
    if dsm isa Nothing
        throw(ArgumentError("No samples available for RandResampling."))
    else
        new_dsm = evalmeasure_impl(dsm, algorithm, context)
        return EvalMeasureImplReturn(empirical = new_dsm)
    end
end

function evalmeasure_impl(dsm::DensitySampleMeasure, algorithm::RandResampling, context::BATContext)
    gen = get_gencontext(context)
    resampled_idxs = _rand_subsample_idxs(gen, dsm, algorithm.nsamples)
    new_dsm = _unweighted_resampling_byidxs(dsm, resampled_idxs)
    return new_dsm
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

function evalmeasure_impl(m::EvaluatedMeasure, algorithm::RandResampling, context::BATContext)
    dsm = getempirical(m)
    if dsm isa Nothing
        throw(ArgumentError("No samples available for RandResampling."))
    else
        new_dsm = evalmeasure_impl(dsm, algorithm, context)
        return EvalMeasureImplReturn(empirical = new_dsm)
    end
end

function evalmeasure_impl(dsm::DensitySampleMeasure, algorithm::RandResampling, context::BATContext)
    resampled_idxs = _ordered_resampling_idxs(samplesof(dsm), algorithm.n, context)
    new_dsm = _unweighted_resampling_byidxs(dsm, resampled_idxs)
    return new_dsm
end

function _ordered_resampling_idxs(smpls::DensitySampleVector, n::Integer, context::BATContext)
    # ToDo: Use PSIS

    rng = get_rng(context)
    @assert axes(smpls) == axes(smpls.weight)
    W = smpls.weight

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

    return resampled_idxs
end
