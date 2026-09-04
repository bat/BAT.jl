# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function _credible_dyadic(w::Integer)
    w <= typemax(UInt128) || return nothing
    coefficient = UInt128(w)
    iszero(coefficient) && return (coefficient, 0)
    shift = trailing_zeros(coefficient)
    coefficient >>> shift, shift
end

function _credible_dyadic(w::AbstractFloat)
    significand, exponent, sign = Base.decompose(w)
    significand <= typemax(UInt128) || return nothing
    coefficient = UInt128(sign * significand)
    iszero(coefficient) && return (coefficient, 0)
    shift = trailing_zeros(coefficient)
    coefficient >>> shift, exponent + shift
end

function _credible_dyadic(w::Rational)
    ispow2(denominator(w)) || return nothing
    dyadic = _credible_dyadic(numerator(w))
    isnothing(dyadic) && return nothing
    coefficient, exponent = dyadic
    coefficient, exponent - trailing_zeros(denominator(w))
end
_credible_dyadic(::Real) = nothing

function _credible_uint_coefficients(W)
    min_exponent = typemax(Int)
    # Validate exact dyadic weights.
    for w in W
        dyadic = _credible_dyadic(w)
        isnothing(dyadic) && return nothing
        coefficient, exponent = dyadic
        !iszero(coefficient) && (min_exponent = min(min_exponent, exponent))
    end
    coefficients = Vector{UInt128}(undef, length(W))
    total = zero(UInt128)
    # Accumulate exact integer mass.
    for i in eachindex(W)
        coefficient, exponent = _credible_dyadic(W[i])
        shift = iszero(coefficient) ? 0 : exponent - min_exponent
        shift < 128 && coefficient <= typemax(UInt128) >>> shift || return nothing
        coefficient <<= shift
        total, overflow = Base.Checked.add_with_overflow(total, coefficient)
        overflow && return nothing
        coefficients[i] = coefficient
    end
    total <= typemax(UInt128) ÷ UInt128(370) ? coefficients : nothing
end

function _credible_exact_value(x::AbstractFloat)
    significand, exponent, sign = Base.decompose(x)
    numerator = BigInt(sign) * significand
    exponent < 0 ? numerator // (big(1) << -exponent) : (numerator << exponent) // big(1)
end
_credible_exact_value(x::Real) = Rational{BigInt}(x)

function _credible_coefficients(W)
    any(w -> w isa ULogarithmic, W) && throw(ArgumentError(
        "logarithmic weights must use homogeneous storage"
    ))
    T = eltype(W)
    if isbitstype(T) && T <: Integer && sizeof(T) <= sizeof(UInt)
        total = sum(UInt128, W)
        total <= typemax(UInt) ÷ UInt(370) && return UInt.(W)
    end
    coefficients = _credible_uint_coefficients(W)
    !isnothing(coefficients) && return coefficients
    weights = _credible_exact_value.(W)
    common_denominator = foldl(lcm, denominator.(weights); init = big(1))
    [numerator(w) * div(common_denominator, denominator(w)) for w in weights]
end
function _credible_coefficients(W::UnitWeights)
    T = length(W) <= typemax(UInt) ÷ UInt(370) ? UInt : UInt128
    UnitWeights{T}(length(W))
end
_credible_coefficients(W::AbstractVector{<:ULogarithmic}) =
    _credible_coefficients(_canonical_rel_weights(W))
function _credible_atoms(X, coefficients)
    atoms = sort!(collect(zip(X, coefficients)); by = first)
    coefficients isa UnitWeights || filter!(atom -> !iszero(last(atom)), atoms)
    isempty(atoms) && return atoms
    write_idx = 1
    # Merge adjacent equal atoms.
    for read_idx in firstindex(atoms)+1:lastindex(atoms)
        if first(atoms[read_idx]) == first(atoms[write_idx])
            atoms[write_idx] = (
                first(atoms[write_idx]), last(atoms[write_idx]) + last(atoms[read_idx])
            )
        else
            write_idx += 1
            atoms[write_idx] = atoms[read_idx]
        end
    end
    resize!(atoms, write_idx)
    coefficients isa UnitWeights && return atoms
    divisor = last(first(atoms))
    # Compute a shared mass scale.
    for atom in @view atoms[2:end]
        divisor = gcd(divisor, last(atom))
        isone(divisor) && break
    end
    divisor == one(divisor) || map!(atom -> (first(atom), div(last(atom), divisor)), atoms, atoms)
    atoms
end

function _credible_width_key(lo, hi)
    standard = lo isa Union{Integer,Rational,AbstractFloat} &&
        hi isa Union{Integer,Rational,AbstractFloat}
    standard ? _credible_exact_value(hi) - _credible_exact_value(lo) : hi - lo
end

_credible_cmp(a, b) = a < b ? -1 : b < a ? 1 : 0
_credible_width_cmp(lo, hi, other_lo, other_hi) = _credible_cmp(
    _credible_width_key(lo, hi), _credible_width_key(other_lo, other_hi)
)
function _credible_width_cmp(lo::T, hi::T, other_lo::T, other_hi::T) where {T<:Union{Float16,Float32,Float64}}
    width = Base.TwicePrecision(hi) - Base.TwicePrecision(lo)
    other_width = Base.TwicePrecision(other_hi) - Base.TwicePrecision(other_lo)
    _credible_width_cmp(lo, hi, other_lo, other_hi, width, other_width)
end
_credible_width_state(lo, hi) = nothing
_credible_width_state(lo::T, hi::T) where {T<:Union{Float16,Float32,Float64}} =
    Base.TwicePrecision(hi) - Base.TwicePrecision(lo)
_credible_width_cmp(lo, hi, other_lo, other_hi, ::Nothing, ::Nothing) =
    _credible_width_cmp(lo, hi, other_lo, other_hi)
@inline function _credible_width_cmp(lo, hi, other_lo, other_hi,
        width::Base.TwicePrecision, other_width::Base.TwicePrecision)
    isfinite(width.hi) && isfinite(other_width.hi) ? _credible_cmp(width, other_width) :
        _credible_cmp(_credible_exact_value(hi) - _credible_exact_value(lo),
            _credible_exact_value(other_hi) - _credible_exact_value(other_lo))
end

function _credible_grid(atoms, n)
    boundaries = Vector{Int}(undef, n + 1)
    boundaries[1] = firstindex(atoms)
    total = sum(last, atoms)
    cumulative = zero(total)
    atom_idx = firstindex(atoms) - 1
    # Advance the inverse ECDF.
    for j in 1:n
        while n * cumulative < j * total
            atom_idx += 1
            cumulative += last(atoms[atom_idx])
        end
        boundaries[j + 1] = atom_idx
    end
    boundaries
end

function _credible_connected(atoms, threshold)
    left = firstindex(atoms)
    mass = zero(threshold)
    best = (first(first(atoms)), first(last(atoms)))
    best_width = _credible_width_state(best...)
    # Slide the mass window.
    for right in eachindex(atoms)
        mass += last(atoms[right])
        while left < right && mass - last(atoms[left]) >= threshold
            mass -= last(atoms[left])
            left += 1
        end
        if mass >= threshold
            lo, hi = first(atoms[left]), first(atoms[right])
            width = _credible_width_state(lo, hi)
            order = _credible_width_cmp(lo, hi, best..., width, best_width)
            if order < 0 || order == 0 && isless((lo, hi), best)
                best, best_width = (lo, hi), width
            end
        end
    end
    [ClosedInterval(best...)]
end

function _credible_count_interval(X, m, n)
    values = sort!(collect(X))
    isfinite(first(values)) && isfinite(last(values)) ||
        throw(ArgumentError("sample values must be finite"))
    window = cld(m * length(values), n)
    best = (first(values), values[window])
    best_width = _credible_width_state(best...)
    # Compare each fixed-count window.
    @inbounds for left in 2:length(values)-window+1
        candidate = (values[left], values[left + window - 1])
        width = _credible_width_state(candidate...)
        order = _credible_width_cmp(candidate..., best..., width, best_width)
        if order < 0 || order == 0 && isless(candidate, best)
            best, best_width = candidate, width
        end
    end
    [ClosedInterval(best...)]
end

function _credible_disjoint(atoms, m, n, threshold)
    boundaries = _credible_grid(atoms, n)
    bin_endpoints(i) = (first(atoms[boundaries[i]]), first(atoms[boundaries[i + 1]]))
    ranking = sortperm(1:n; lt = (i, j) -> begin
        order = _credible_width_cmp(bin_endpoints(i)..., bin_endpoints(j)...)
        order < 0 || order == 0 && i < j
    end)
    selected = falses(n)
    covered = falses(length(atoms))
    mass = zero(threshold)
    # Track overlapping selected bins.
    for r in eachindex(ranking)
        bin = ranking[r]
        selected[bin] = true
        # Count each atom once.
        for atom_idx in boundaries[bin]:boundaries[bin + 1]
            if !covered[atom_idx]
                covered[atom_idx] = true
                mass += last(atoms[atom_idx])
            end
        end
        r >= m && mass >= threshold && break
    end

    intervals = [ClosedInterval(
        first(atoms[boundaries[i]]), first(atoms[boundaries[i + 1]])
    ) for i in findall(selected)]
    merged = eltype(intervals)[]
    # Merge adjacent selected intervals.
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
    smallest_credible_intervals(
        X::AbstractVector{<:Real}, W::AbstractWeights = UnitWeights(...);
        nsigma_equivalent::Real = 1, mode::Symbol = :disjoint
    )

*BAT-internal, not part of stable public API.*

Construct a ranked exact atom-quantile-grid region containing the requested
empirical mass. Use `mode = :connected` for the shortest connected atom-aligned
interval. Atomic mass is indivisible, so either result may exceed the target.
"""
function smallest_credible_intervals(
    X::AbstractVector{<:Real},
    W::AbstractWeights = UnitWeights{eltype(X)}(length(eachindex(X)));
    nsigma_equivalent::Real = 1,
    mode::Symbol = :disjoint,
)
    nsigma_90percent = quantile(Normal(), 0.5 + 0.9/2)
    m, n = if nsigma_equivalent ≈ oftype(nsigma_equivalent, 1)
        28, 41
    elseif nsigma_equivalent ≈ oftype(nsigma_equivalent, 2)
        42, 44
    elseif nsigma_equivalent ≈ oftype(nsigma_equivalent, 3)
        369, 370
    elseif isapprox(nsigma_equivalent, nsigma_90percent, atol = 0.01)
        90, 100
    else
        throw(ArgumentError("nsigma_equivalent must be 1, 2, 3 or 1.64 (for 90% credibility interval)"))
    end
    mode in (:disjoint, :connected) ||
        throw(ArgumentError("mode must be :disjoint or :connected"))
    length(X) == length(W) ||
        throw(ArgumentError("data and weight vectors must have the same size"))
    isempty(X) && throw(ArgumentError("credible intervals of an empty array are undefined"))
    mode === :connected && W isa UnitWeights && return _credible_count_interval(X, m, n)
    all(isfinite, X) || throw(ArgumentError("sample values must be finite"))
    (W isa UnitWeights || all(w -> isfinite(w) && w >= zero(w), W)) ||
        throw(ArgumentError("sample weights must be finite and non-negative"))
    atoms = _credible_atoms(X, _credible_coefficients(W))
    isempty(atoms) && throw(ArgumentError("sample weights must contain positive mass"))
    threshold = cld(m * sum(last, atoms), n)
    mode == :connected ? _credible_connected(atoms, threshold) :
        _credible_disjoint(atoms, m, n, threshold)
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
