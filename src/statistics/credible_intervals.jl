# This file is a part of BAT.jl, licensed under the MIT License (MIT).


_credible_exact(x::Real) = Rational{BigInt}(x)
_credible_exact(x::DoubleFloat) = _credible_exact(x.hi) + _credible_exact(x.lo)
function _credible_exact(x::BigFloat)
    n, e, d = Base.decompose(x)
    (n // d) * (big(2) // 1)^e
end

function _credible_masses(w)
    r = _credible_exact.(w)
    scale = foldl(lcm, denominator.(r); init = big(1))
    masses = numerator.(r) .* (scale .÷ denominator.(r))
    sum(masses) <= typemax(Int128) ? Int128.(masses) : masses
end
_credible_masses(w::UnitWeights) = ones(Int, length(w))
_credible_masses(w::AbstractVector{T}) where {T<:Integer} =
    isbitstype(T) && sizeof(T) <= sizeof(Int) ? Int128.(w) : big.(w)
function _credible_masses(w::AbstractVector{<:ULogarithmic})
    r = _canonical_rel_weights(w)
    if any(iszero.(r) .& .!iszero.(w))
        logs = BigFloat.(log.(w))
        r = exp.(logs .- maximum(logs))
        any(iszero.(r) .& .!iszero.(w)) && throw(ArgumentError("logarithmic weight range is too large"))
    end
    _credible_masses(r)
end

_credible_width(a) = _credible_exact(a[2]) - _credible_exact(a[1])
_credible_shorter(a, b) = _credible_width(a) < _credible_width(b)
function _credible_shorter(a::Tuple{T,T}, b::Tuple{T,T}) where {T<:Union{Float16,Float32,Float64}}
    wa, wb = a[2] - a[1], b[2] - b[1]
    isfinite(wa) && isfinite(wb) && wa != wb ? wa < wb :
        _credible_width(a) < _credible_width(b)
end

function _credible_connected(x, c, target)
    left, best = 1, (first(x), last(x))
    for right in eachindex(x)
        while left < right && c[right + 1] - c[left + 1] >= target
            left += 1
        end
        candidate = (x[left], x[right])
        c[right + 1] - c[left] >= target && _credible_shorter(candidate, best) && (best = candidate)
    end
    [ClosedInterval(best...)]
end

function _credible_disjoint(x, c, target)
    edges = max.(1, searchsortedfirst.(Ref(c[2:end]), last(c) .* (big(0):100) .// 100))
    endpoints(i) = (x[edges[i]], x[edges[i + 1]])
    ranking = sortperm(1:100; lt = (i, j) -> _credible_shorter(endpoints(i), endpoints(j)))
    covered, selected, mass = falses(length(x)), Int[], zero(last(c))
    for i in ranking
        push!(selected, i)
        for k in edges[i]:edges[i + 1]
            covered[k] || (mass += c[k + 1] - c[k])
            covered[k] = true
        end
        mass >= target && break
    end
    intervals = [ClosedInterval(endpoints(i)...) for i in sort!(selected)]
    merged = eltype(intervals)[]
    for interval in intervals
        if isempty(merged) || minimum(interval) > maximum(last(merged))
            push!(merged, interval)
        else
            merged[end] = ClosedInterval(minimum(last(merged)), maximum(interval))
        end
    end
    merged
end

"""
    smallest_credible_intervals(X, W = UnitWeights(...); p = nothing,
        nsigma_equivalent = nothing, mode = :disjoint)

*BAT-internal, not part of stable public API.*

Return empirical credible intervals. Use `:connected` for the shortest single
interval. Set `p` in `(0, 1]` or `nsigma_equivalent`; the default is one sigma.
"""
function smallest_credible_intervals(X::AbstractVector{<:Real},
        W::AbstractWeights = UnitWeights{eltype(X)}(length(X));
        p::Union{Real,Nothing} = nothing, nsigma_equivalent::Union{Real,Nothing} = nothing,
        mode::Symbol = :disjoint)
    isnothing(p) || isnothing(nsigma_equivalent) || throw(ArgumentError("p and nsigma_equivalent are mutually exclusive"))
    p = isnothing(p) ? erf(something(nsigma_equivalent, 1) / sqrt(2)) : p
    0 < p <= 1 || throw(ArgumentError("p must be in (0, 1]"))
    mode in (:connected, :disjoint) || throw(ArgumentError("mode must be :connected or :disjoint"))
    length(X) == length(W) || throw(DimensionMismatch("sample values and weights must have equal lengths"))
    all(isfinite, X) && all(w -> isfinite(w) && w >= 0, W) || throw(ArgumentError("samples and weights must be finite, with nonnegative weights"))
    isempty(X) && throw(ArgumentError("samples must contain positive mass"))
    x = collect(X)
    if W isa UnitWeights && mode == :connected
        sort!(x)
        c = 0:length(x)
    else
        w = collect(_credible_masses(W))
        order = filter(i -> !iszero(w[i]), sortperm(x))
        isempty(order) && throw(ArgumentError("samples must contain positive mass"))
        x, c = x[order], cumsum(w[order])
        ends = [findall(x[1:end-1] .!= x[2:end]); length(x)]
        x, c = x[ends], [zero(eltype(c)); c[ends]]
    end
    target = ceil(typeof(last(c)), _credible_exact(p) * last(c))
    mode == :connected ? _credible_connected(x, c, target) : _credible_disjoint(x, c, target)
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
