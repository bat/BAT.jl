# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    smallest_credible_intervals(
        X::AbstractVector{<:Real}, W::AbstractWeights = UnitWeights(...);
        nsigma_equivalent::Real = 1, credibility::Union{Nothing,Real} = nothing
    )

*BAT-internal, not part of stable public API.*

Find smallest credible intervals with `nsigma_equivalent` of 1, 2 or 3
(containing 68.27%, 90.00%, 95.45% or 99.73% of the total probability mass).
"""
function smallest_credible_intervals(
    X::AbstractVector{<:Real},
    W::AbstractWeights = UnitWeights{eltype(X)}(length(eachindex(X)));
    nsigma_equivalent::Real = 1,
    credibility::Union{Nothing,Real} = nothing
)
    nsigma_90percent = quantile(Normal(), 0.5 + 0.9/2)   # 90% = 1.6448536269514717

    m, n, partial_probability = if !isnothing(credibility)
        if !(isfinite(credibility) && 0 < credibility < 1)
            throw(ArgumentError("credibility must be finite and between zero and one"))
        end
        resolution = 10_000
        grid_position = credibility * resolution
        nearest_grid_position = round(Int, grid_position)
        is_grid_value = 0 < nearest_grid_position < resolution &&
            abs(grid_position - nearest_grid_position) <= 8 * eps(float(grid_position))
        if is_grid_value
            credibility_fraction = nearest_grid_position // resolution
            numerator(credibility_fraction), denominator(credibility_fraction), nothing
        else
            m = floor(Int, grid_position)
            m, resolution, credibility - m // resolution
        end
    elseif nsigma_equivalent ≈ oftype(nsigma_equivalent, 1)
        28,  41, nothing  # 0.6827 ≈ 28//41
    elseif nsigma_equivalent ≈ oftype(nsigma_equivalent, 2)
        42,  44, nothing  # 0.9545 ≈ 42//44
    elseif nsigma_equivalent ≈ oftype(nsigma_equivalent, 3)
        369,  370, nothing  # 0.9973 ≈ 369/370
    elseif isapprox(nsigma_equivalent, nsigma_90percent, atol = 0.01)   # 0.90 ≈ 1.64
        90, 100, nothing
    else
        throw(ArgumentError("nsigma_equivalent must be 1, 2, 3 or 1.64 (for 90% credibility interval)"))
    end

    qs = quantile(X, W, range(0, 1, length = n + 1))
    ivs = ClosedInterval.(qs[begin:end-1], qs[begin+1:end])

    sorted_idxs = sortperm(ivs, by = width)
    sel_idxs = sort(sorted_idxs[begin:begin+m-1])

    r_idxs = eachindex(sel_idxs)
    for i in r_idxs
        s = sel_idxs[i]
        if (i == first(r_idxs) || sel_idxs[i-1] != s-1) && (i == last(r_idxs) || sel_idxs[i+1] != s+1)
            if i >= first(r_idxs)+2 && sel_idxs[i-1] == s-2 && width(ivs[s-1]) >= width(ivs[s])/2
                sel_idxs[i] = s-1
            end
            if i <= last(r_idxs)-2 && sel_idxs[i+1] == s+2 && width(ivs[s+1]) >= width(ivs[s])/2
                sel_idxs[i] = s+1
            end
        end
    end

    sel_ranges = UnitRange{Int}[]
    for i in sel_idxs
        if isempty(sel_ranges) || i != last(sel_ranges[end]) + 1
            push!(sel_ranges, i:i)
        else
            sel_ranges[end] = first(sel_ranges[end]):i
        end
    end

    intervals = [ClosedInterval(minimum(ivs[first(r)]), maximum(ivs[last(r)])) for r in sel_ranges]

    if !isnothing(partial_probability)
        selected = falses(n)
        selected[sel_idxs] .= true
        partial_idx = sorted_idxs[findfirst(i -> !selected[i], sorted_idxs)]
        bin_start = (partial_idx - 1) / n
        bin_end = partial_idx / n
        lower_partial = ClosedInterval(
            quantile(X, W, bin_start),
            quantile(X, W, bin_start + partial_probability),
        )
        upper_partial = ClosedInterval(
            quantile(X, W, bin_end - partial_probability),
            quantile(X, W, bin_end),
        )
        has_lower_neighbor = partial_idx > 1 && selected[partial_idx - 1]
        has_upper_neighbor = partial_idx < n && selected[partial_idx + 1]
        partial_interval = if has_lower_neighbor && !has_upper_neighbor
            lower_partial
        elseif has_upper_neighbor && !has_lower_neighbor
            upper_partial
        else
            width(lower_partial) <= width(upper_partial) ? lower_partial : upper_partial
        end
        push!(intervals, partial_interval)
        sort!(intervals, by = minimum)

        merged_intervals = empty(intervals)
        for interval in intervals
            if !isempty(merged_intervals) && minimum(interval) <= maximum(last(merged_intervals))
                merged_intervals[end] = ClosedInterval(
                    minimum(last(merged_intervals)),
                    max(maximum(last(merged_intervals)), maximum(interval)),
                )
            else
                push!(merged_intervals, interval)
            end
        end
        intervals = merged_intervals
    end

    intervals
end


"""
    smallest_credible_intervals(smpl::DensitySampleVector{<:AbstractVector{<:Real}}; kwargs...)

*BAT-internal, not part of stable public API.*
"""
function smallest_credible_intervals(smpl::DensitySampleVector{<:AbstractVector{<:Real}}; kwargs...)
    V = flatview(smpl.v)
    W = Weights(smpl.weight)
    [smallest_credible_intervals(V[i,:], W; kwargs...) for i in axes(V,1)]
end

function smallest_credible_intervals(smpl::DensitySampleVector; kwargs...)
    # ToDo: Make type-stable.
    vs = elshape(smpl.v)
    ivs = smallest_credible_intervals(unshaped.(smpl); kwargs...)
    idxs = replace_const_shapes(x -> ConstValueShape(nothing), vs)(eachindex(ivs))
    fmap(x -> isnothing(x) ? x : map(Base.Fix1(getindex, ivs), x), idxs)
end
