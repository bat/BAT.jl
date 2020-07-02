# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function ApproximateTwoSampleKSTest(
        x::AbstractVector{T},
        y::AbstractVector{S},
        w_x::AbstractVector{D},
        w_y::AbstractVector{M}) where {T<:Real, S<:Real, D<:Real, M<:Real}
    HypothesisTests.ApproximateTwoSampleKSTest(ksstats(x, y, w_x, w_y)...)
end
export ApproximateTwoSampleKSTest

function ksstats(
        x::AbstractVector{T},
        y::AbstractVector{S},
        w_x::AbstractVector{D},
        w_y::AbstractVector{M}
    ) where {T<:Real, S<:Real, D<:Real, M<:Real}

    n_x, n_y = length(w_x), length(w_y) # ToDo: sum(w_x), length(w_x) or bat_eff_sample_size(...)
    sort_idx = sortperm([x; y])
    pdf_diffs = [w_x/sum(w_x); -w_y/sum(w_y)][sort_idx]
    cdf_diffs = cumsum(pdf_diffs)
    δp = maximum(cdf_diffs)
    δn = -minimum(cdf_diffs)
    δ = max(δp, δn)
    (n_x, n_y, δ, δp, δn)
end

function bat_compare(samples_1::DensitySampleVector, samples_2::DensitySampleVector)

    n_params = size(flatview(unshaped.(samples_2.v)))[1]
    p_values_array = Float64[]

    samples_1_flat = flatview(unshaped.(samples_1.v))
    samples_2_flat = flatview(unshaped.(samples_2.v))

    for param_ind in Base.OneTo(n_params)
        test_result = ApproximateTwoSampleKSTest(
            samples_1_flat[param_ind, :],
            samples_2_flat[param_ind, :],
            samples_1.weight,
            samples_2.weight
        )
        push!(p_values_array, HypothesisTests.pvalue(test_result))
    end

    # ToDo: Add Chi-squared test; Use bat_eff_sample_size to estimate n_x, n_y
    # Return: TypedTables.Table(dims = Base.OneTo(n_params), ks_p_values = p_values_array, cs_p_values = [...])

    return TypedTables.Table(dims = Base.OneTo(n_params), ks_p_values = p_values_array)
end

export bat_compare
