# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function ksstats(
        x::AbstractVector{T},
        y::AbstractVector{S},
        w_x::AbstractVector{D},
        w_y::AbstractVector{M}
    ) where {T<:Real, S<:Real, D<:Real, M<:Real}

    sort_idx = sortperm([x; y])
    pdf_diffs = [w_x/sum(w_x); -w_y/sum(w_y)][sort_idx]
    cdf_diffs = cumsum(pdf_diffs)
    δp = maximum(cdf_diffs)
    δn = -minimum(cdf_diffs)
    δ = max(δp, δn)
    (δ, δp, δn)
end

"""
    bat_compare(
        samples_1::DensitySampleVector,
        samples_2::DensitySampleVector;
        nsamples::Symbol=:effective
    )

Compares two `DensitySampleVector`s given by `samples_1` and `samples_2` applying the Kolmogorov-Smirnov test for all marginals.

`nsamples` specifies how to define a number of samples in the Kolmogorov-Smirnov
distribution. The default value is `nsamples=:effective`, which uses the
effective number of samples estimated by `bat_eff_sample_size`. The optimal keywords:

* `:length`  — length of the `DensitySamplesVector` is used

* `:weights` — the sum of the weights is used

Returns a NamedTuple of the shape

```julia
(result = X::TypedTables.Table, ...)
```
"""
function bat_compare(samples_1::DensitySampleVector, samples_2::DensitySampleVector;  nsamples::Symbol=:effective)

    n_params_1 = size(flatview(unshaped.(samples_1.v)))[1]
    n_params_2 = size(flatview(unshaped.(samples_2.v)))[1]

    if n_params_1 !== n_params_2
        throw(ArgumentError("Samples with different number of parameters cannot be compared"))
    end

    samples_1_flat = flatview(unshaped.(samples_1.v))
    samples_2_flat = flatview(unshaped.(samples_2.v))

    if nsamples == :effective
        @info "Using effective number of samples in the Kolmogorov–Smirnov distribution"
        smpl_size_1 = round.(Int64, bat_eff_sample_size(unshaped.(samples_1)).result)
        smpl_size_2 = round.(Int64, bat_eff_sample_size(unshaped.(samples_2)).result)
    elseif nsamples == :length
        @info "Using lengths of the vectors in the Kolmogorov–Smirnov distribution"
        length_1 = length(samples_1)
        length_2 = length(samples_1)
        smpl_size_1 = repeat([length_1], n_params_1)
        smpl_size_2 = repeat([length_2], n_params_2)
    elseif nsamples == :weights
        @info "Using sum of the weights in the Kolmogorov–Smirnov distribution"
        tot_weights_1 = sum(samples_1.weight)
        tot_weights_2 = sum(samples_2.weight)
        smpl_size_1 = repeat([tot_weights_1], n_params_1)
        smpl_size_2 = repeat([tot_weights_2], n_params_2)
    else
        throw(ArgumentError("Unknown keyword"))
    end

    p_values_array = map(i-> begin
        test_result = ksstats( samples_1_flat[i, :], samples_2_flat[i, :], samples_1.weight, samples_2.weight)
        ks_test = HypothesisTests.ApproximateTwoSampleKSTest(smpl_size_1[i], smpl_size_2[i], test_result...)
        HypothesisTests.pvalue(ks_test)
    end, Base.OneTo(n_params_1))

    table_result = TypedTables.Table(
            dim_ind = Base.OneTo(n_params_1), # indices of parameters
            nsamples_1 = smpl_size_1, # number of samples for `samples_1`
            nsamples_2 = smpl_size_2, # number of samples for `samples_2`
            ks_p_values = p_values_array, # Kolmogorov-Smirnov p-values
            # additional tests can be added to this table/function later
    )

    return (result=table_result, )
end

export bat_compare
