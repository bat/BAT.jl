# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    smallest_credible_intervals(
        X::AbstractVector{<:Real}, W::AbstractWeights = UnitWeights(...);
        nsigma_equivalent::Integer = 1
    )

*BAT-internal, not part of stable public API.*

Find smalles credible intervals with `nsigma_equivalent` of 1, 2 or 3
(containing 68.27%, 95.45%, 90.00% or 99.73% of the total probability mass).
"""
function smallest_credible_intervals(
    X::AbstractVector{<:Real},
    W::AbstractWeights = UnitWeights{eltype(X)}(length(eachindex(X)));
    nsigma_equivalent::Real = 1
)
    nsigma_90percent = quantile(Normal(), 0.5 + 0.9/2)   # 90% = 1.6448536269514717

    m, n = if nsigma_equivalent ≈ oftype(nsigma_equivalent, 1)
        28,  41  # 0.6827 ≈ 28//41
    elseif nsigma_equivalent ≈ oftype(nsigma_equivalent, 2)
        42,  44  # 0.9545 ≈ 42//44
    elseif nsigma_equivalent ≈ oftype(nsigma_equivalent, 3)
        369,  370  # 0.9973 ≈ 369/370
    elseif isapprox(nsigma_equivalent, nsigma_90percent, atol = 0.01)   # 0.90 ≈ 1.64
        90, 100
    else
        throw(ArgumentError("nsigma_equivalent must be 1, 2, 3 or 1.64 (for 90% credibility interval)"))
    end

    qs = quantile(X, W, range(0, 1, length = n + 1))
    ivs = ClosedInterval.(qs[begin:end-1], qs[begin+1:end])
    
    sel_idxs = sort(sortperm(ivs, by = width)[begin:begin+m-1])

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

    [ClosedInterval(minimum(ivs[first(r)]), maximum(ivs[last(r)])) for r in sel_ranges]
end


"""
    smallest_credible_intervals(smpl::DensitySampleVector{<:AbstractVector{<:Real}}; kwargs...)
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
