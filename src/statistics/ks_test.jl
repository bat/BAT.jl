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

    n_x, n_y = length(w_x), length(w_y) #sum(w_x), sum(w_y) # Length or weights?
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

    for param_ind in Base.OneTo(n_params)
        test_result = ApproximateTwoSampleKSTest(
            flatview(unshaped.(samples_1.v))[param_ind, :],
            flatview(unshaped.(samples_2.v))[param_ind, :],
            samples_1.weight,
            samples_2.weight
        )
        push!(p_values_array, HypothesisTests.pvalue(test_result))
    end
    return p_values_array
end

export bat_compare
