# This file is a part of BAT.jl, licensed under the MIT License (MIT).

_default_min_ess(samples::DensitySampleVector) = minimum(unshaped(bat_eff_sample_size(samples).result))


function mean_logpdf(dist::Distribution, samples::DensitySampleVector)
    # Need to filter out zero-weight samples to avoid problems with -Inf log-likelihood values:
    sel_idxs = findall(x -> x > eps(float(eltype(samples.weight))), samples.weight)
    samples_v = view(samples.v, sel_idxs)
    samples_weight = view(samples.weight, sel_idxs)

    mean(logpdf.(Ref(dist), samples_v), FrequencyWeights(samples_weight))
end


function likelihood_pvalue(
    dist::Distribution, samples::DensitySampleVector;
    ess::Integer = floor(Int, _default_min_ess(samples)),
    ensemble_size::Integer = 100
)
    ref_test_stats = [BAT.mean_logpdf(dist, bat_sample(bat_rng(), dist, IIDSampling(nsamples = ess)).result) for i in 1:ensemble_size]
    samples_test_stat = mean_logpdf(dist, samples)
    count(x -> x <= samples_test_stat, ref_test_stats) / length(eachindex(ref_test_stats))
end
