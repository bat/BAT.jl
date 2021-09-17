# This file is a part of BAT.jl, licensed under the MIT License (MIT).

_default_min_ess(samples::DensitySampleVector) = minimum(unshaped(bat_eff_sample_size(samples).result))


function dist_samples_pvalue(
    rng::AbstractRNG, dist::Distribution, samples::DensitySampleVector;
    nsamples::Integer = floor(Int, _default_min_ess(samples)),
    ess::Integer = floor(Int, _default_min_ess(samples)),
)
    samples_v = bat_sample(rng, samples, OrderedResampling(nsamples = ess)).result.v
    samples_dist_logpdfs = logpdf.(Ref(dist), samples_v)
    ref_samples = bat_sample(rng, dist, IIDSampling(nsamples = nsamples)).result
    ref_dist_logpdfs = ref_samples.logd
    samples_dist_logpdfs, ref_dist_logpdfs
    #HypothesisTests.pvalue(HypothesisTests.ApproximateTwoSampleKSTest(samples_dist_logpdfs, ref_dist_logpdfs))
    HypothesisTests.pvalue(HypothesisTests.KSampleADTest(Vector(samples_dist_logpdfs), Vector(ref_dist_logpdfs)))
end

function dist_samples_pvalue(dist::Distribution, samples::DensitySampleVector; kwargs...)
    dist_samples_pvalue(bat_rng(), dist, samples; kwargs...)
end
