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

    # KS and AD have trouble with large number of samples on 32-bit systems:
    #HypothesisTests.pvalue(HypothesisTests.ApproximateTwoSampleKSTest(samples_dist_logpdfs, ref_dist_logpdfs))
    #HypothesisTests.pvalue(HypothesisTests.KSampleADTest(Vector(samples_dist_logpdfs), Vector(ref_dist_logpdfs)))
    # So use custom KS-calculation instead:
    ks_pvalue(fast_ks_delta(samples_dist_logpdfs, ref_dist_logpdfs), length(samples_dist_logpdfs), length(ref_dist_logpdfs))
end


function dist_samples_pvalue(dist::Distribution, samples::DensitySampleVector; kwargs...)
    dist_samples_pvalue(bat_rng(), dist, samples; kwargs...)
end


function fast_ks_delta(unsorted_x::AbstractVector{<:Real}, unsorted_y::AbstractVector{<:Real})
    x = sort(unsorted_x)
    y = sort(unsorted_y)
    idxs_x = eachindex(x)
    idxs_y = eachindex(y)
    n_x = length(idxs_x)
    n_y = length(idxs_y)
    i0_x::Int = first(idxs_x)
    i0_y::Int = first(idxs_y)
    i_x::Int = i0_x
    i_y::Int = i0_y
    max_delta::Float64 = 0.0
    while i_x in idxs_x && i_y in idxs_y
        x_ge_y = x[i_x] >= y[i_y]
        delta = abs((i_x - i0_x) / n_x - (i_y - i0_y) / n_y)
        max_delta = max(delta, max_delta)
        i_x = ifelse(!x_ge_y, i_x + 1, i_x)
        i_y = ifelse(x_ge_y, i_y + 1, i_y)
    end
    return max_delta
end


function ks_pvalue(delta::Real, n_x::Integer, n_y::Integer)
    nx, ny = float(n_x), float(n_y)
    n = nx * ny / (nx + ny)
    ccdf(HypothesisTests.Kolmogorov(), sqrt(n) * delta)
end
