# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function sample_and_verify(
    target::MeasureLike, samplingalg::AbstractSamplingAlgorithm,
    ref_dist::Distribution = target, context::BATContext = get_batcontext();
    max_retries::Integer = 1, essalg = nothing
)
    measure = convert_for(bat_sample, target)
    initial_em = evalmeasure(measure, samplingalg, context)
    em::typeof(initial_em) = initial_em
    verified::Bool = test_dist_samples(ref_dist, samplesof(em), context; essalg = essalg)
    n_retries::Int = 0
    while !(verified) && n_retries < max_retries
        n_retries += 1
        em = evalmeasure(measure, samplingalg, context)
        verified = test_dist_samples(ref_dist, samplesof(em), context; essalg = essalg)
    end
    (result = samplesof(em), evaluated = em, verified = verified, n_retries = n_retries)
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


function evalmeasure_impl(
    ::EvaluatedMeasure{<:BispacedMeasure{<:DensitySampleMeasure}},
    ::IIDSampling,
    ::BATContext,
)
    throw(ArgumentError(
        "IIDSampling is not supported for DensitySampleMeasure; use RandResampling or SystematicResampling instead.",
    ))
end

function evalmeasure_impl(em::EvaluatedMeasure, algorithm::IIDSampling, context::BATContext)
    m = unevaluated(em)
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
    dsm = DensitySampleMeasure(smpls, dof = _dofval_or_nothing(getdof(m)), ess = length(smpls))
    # A stored sample generation scheme did not produce the new empirical
    # content, so it is cleared conservatively (see the EvaluatedMeasure
    # docs on samplegen):
    return EvaluatedMeasure(em; empirical = dsm, samplegen = nothing, evalinfo = MeasureEvalInfo(algorithm, (;)))
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

function _resampled_empirical(p::BispacedMeasure, algorithm::RandResampling, context::BATContext)
    gen = get_gencontext(context)
    resampled_idxs = _rand_subsample_idxs(gen, p.main, algorithm.nsamples)
    return _without_sampleids(_unweighted_resampling_byidxs(p, resampled_idxs))
end


"""
    struct SystematicResampling <: AbstractSamplingAlgorithm

Systematic resampling from a given series of samples, keeping the order
of the samples: a single stratified uniform yields exactly `nsamples`
draws in one order-preserving pass. It typically gives lower variance
than multinomial resampling, though its conditional variance is
ordering-dependent and does not uniformly dominate the other standard
resampling schemes.

See [G. Kitagawa, "Monte Carlo Filter and Smoother for Non-Gaussian
Nonlinear State Space Models", J. Comput. Graph. Stat. 5(1)
(1996)](https://doi.org/10.1080/10618600.1996.10474692).

Can be used to efficiently convert weighted samples into samples with unity
weights.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct SystematicResampling <: AbstractSamplingAlgorithm
    nsamples::Int = 10^5
end
export SystematicResampling

function evalmeasure_impl(em::EvaluatedMeasure, algorithm::Union{RandResampling,SystematicResampling}, context::BATContext)
    emp = _empirical_rep(em)
    if isnothing(emp)
        throw(ArgumentError("No samples available for $(nameof(typeof(algorithm)))."))
    else
        # The resampled pair stays in the current transformed-space view. A
        # stored sample generation scheme did not produce the new empirical
        # content, so it is cleared conservatively (see the EvaluatedMeasure
        # docs on samplegen):
        new_emp = _resampled_empirical(emp, algorithm, context)
        return EvaluatedMeasure(em; empirical = new_emp, samplegen = nothing, evalinfo = MeasureEvalInfo(algorithm, (;)))
    end
end

function _resampled_empirical(p::BispacedMeasure, algorithm::SystematicResampling, context::BATContext)
    resampled_idxs, is_identity =
        _systematic_resampling_idxs(samplesof(p.main), algorithm.nsamples, context)
    return _unweighted_resampling_byidxs(p, resampled_idxs; preserve_ess = is_identity)
end

function _systematic_resampling_idxs(smpls::DensitySampleVector, n::Integer, context::BATContext)
    # ToDo: Use PSIS

    rng = get_rng(context)
    @assert axes(smpls) == axes(smpls.weight)
    # Canonical relative weights: validated (any negative weight would
    # make the cumulative weights non-monotone and invalidate the
    # systematic-resampling semantics) and normalized so that the
    # cumulative sum can neither overflow nor wrap around:
    W = _canonical_rel_weights(smpls.weight)
    T = _weight_accum_type(W)
    W_total = sum(T, W)
    iszero(W_total) && !iszero(n) && throw(ArgumentError(
        "Can't draw from zero-weight samples"
    ))

    # Systematic resampling (Kitagawa 1996): a single stratified uniform
    # yields exactly n draws in one order-preserving pass, typically with
    # lower variance than multinomial resampling:
    u = rand(rng)
    resampled_idxs = Vector{Int}(undef, n)
    j = 0
    cw = zero(T)
    # Equal weights at the same size select each row once, so systematic
    # resampling adds no conditional variance:
    is_identity = n == length(W) && !isempty(W)
    for i in eachindex(W)
        is_identity &= isone(W[i])
        cw += W[i]
        thresh = cw * n / W_total
        while j < n && u + j < thresh
            j += 1
            resampled_idxs[j] = i
        end
    end
    # Guard against floating-point shortfall at the very end:
    while j < n
        j += 1
        resampled_idxs[j] = lastindex(W)
    end

    return resampled_idxs, is_identity
end
