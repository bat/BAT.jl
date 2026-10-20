# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function _ks_relative_weights(weights::AbstractVector{<:Real})
    get_zero_subnormals() && throw(ArgumentError("KS comparison requires gradual underflow"))
    isempty(weights) && throw(ArgumentError("KS comparison requires nonempty populations"))
    all(w -> isfinite(w) && w >= 0, weights) ||
        throw(ArgumentError("KS weights must be finite and nonnegative"))
    largest = maximum(weights)
    largest > 0 || throw(ArgumentError("KS weights must have positive total mass"))
    T = promote_type(Float64, float(eltype(weights)))
    T === BigFloat && (rounding(BigFloat) != RoundNearest || precision(BigFloat) < 53) &&
        throw(ArgumentError("KS comparison requires round-to-nearest BigFloat arithmetic with at least 53 bits"))
    # Widen before division, but normalize logarithmic weights before converting
    # to linear space, as in the existing core relative-weight convention.
    u = if eltype(weights) <: Rational
        # Widen before division: fixed-width rational products can overflow.
        # Keep ordinary rational conversion independent of ambient BigFloat
        # precision (conversion through Rational{BigInt} would use MPFR).
        T.(weights) ./ T(largest)
    else
        _normalize_rel_weights(weights, largest, T)
    end
    all(i -> isfinite(u[i]) && (iszero(weights[i]) || u[i] > 0), eachindex(weights, u)) ||
        throw(ArgumentError("positive KS relative mass is outside the working numerical range"))
    u
end

function _ks_weight_total(weights, order)
    state = (zero(eltype(weights)), zero(eltype(weights)))
    for i in order
        state = kbn_add(state, weights[i])
    end
    total = sum(state)
    isfinite(total) && total > 0 || throw(ArgumentError("KS weight total is outside the working numerical range"))
    total
end

function _ksstats_prepared(x, y, ux, uy)
    # Separate permutations preserve comparisons between original coordinate
    # types (e.g. Int64(2^53+1) must not become Float64(2^53)).
    order_x, order_y = sortperm(x), sortperm(y)
    nx, ny = length(x), length(y)
    total_x, total_y = _ks_weight_total(ux, order_x), _ks_weight_total(uy, order_y)
    sx = (zero(eltype(ux)), zero(eltype(ux)))
    sy = (zero(eltype(uy)), zero(eltype(uy)))
    i, j = 1, 1
    delta_p = delta_n = zero(promote_type(eltype(ux), eltype(uy)))
    while i <= nx || j <= ny
        value = if j > ny || (i <= nx && x[order_x[i]] <= y[order_y[j]])
            x[order_x[i]]
        else
            y[order_y[j]]
        end
        while i <= nx && x[order_x[i]] == value
            sx = kbn_add(sx, ux[order_x[i]])
            i += 1
        end
        while j <= ny && y[order_y[j]] == value
            sy = kbn_add(sy, uy[order_y[j]])
            j += 1
        end
        # Right-continuous CDFs change only at complete tie groups. A fully
        # consumed population has CDF exactly one, including trailing zeros.
        fx = i > nx ? one(total_x) : sum(sx) / total_x
        fy = j > ny ? one(total_y) : sum(sy) / total_y
        delta = fx - fy
        delta_p = max(delta_p, delta)
        delta_n = max(delta_n, -delta)
    end
    (max(delta_p, delta_n), delta_p, delta_n)
end

function ksstats(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
    w_x::AbstractVector{<:Real}, w_y::AbstractVector{<:Real})
    Base.require_one_based_indexing(x, y, w_x, w_y)
    length(x) == length(w_x) && length(y) == length(w_y) ||
        throw(DimensionMismatch("KS coordinates and weights must have matching lengths"))
    all(isfinite, x) && all(isfinite, y) || throw(ArgumentError("KS coordinates must be finite"))
    _ksstats_prepared(x, y, _ks_relative_weights(w_x), _ks_relative_weights(w_y))
end

function _ks_sample_count(value::Real; rounded::Bool=false)
    isfinite(value) && value > 0 || throw(ArgumentError("KS sample counts must be finite and positive"))
    n = rounded ? round(value) : value
    isinteger(n) && 0 < n <= typemax(Int) ||
        throw(ArgumentError("KS sample counts must be positive integers representable as Int"))
    Int(n)
end

function _ks_effective_counts(value, n_params)
    if value isa Real
        fill(_ks_sample_count(value; rounded=true), n_params)
    else
        value isa AbstractVector{<:Real} && length(value) == n_params ||
            throw(ArgumentError("KS effective sample size must be scalar or have one entry per parameter"))
        [_ks_sample_count(v; rounded=true) for v in value]
    end
end

function _ks_count_fraction(value::Real)
    if value isa Union{Integer,Rational}
        Rational{BigInt}(value)
    elseif value isa AbstractFloat
        isinteger(value) && return Rational{BigInt}(iszero(value) ? 0 : _ks_sample_count(value))
        v = Float64(value)
        isfinite(v) && v == value || throw(ArgumentError(
            "fractional KS count weights must be exactly representable as Float64 or supplied as rationals"))
        Rational{BigInt}(v)
    else
        throw(ArgumentError("absolute KS counts require ordinary numeric frequency weights; use :length or :effective for relative weights"))
    end
end

function _ks_absolute_count(weights)
    all(w -> w isa Union{Integer,Rational,AbstractFloat}, weights) ||
        throw(ArgumentError("absolute KS counts require ordinary numeric frequency weights; use :length or :effective for relative weights"))
    if all(isinteger, weights)
        total = 0
        for w in weights
            # Zero rows contribute no count. Each positive count is admitted
            # before checked addition, so neither conversion nor sum can wrap.
            iszero(w) && continue
            n = _ks_sample_count(w)
            n <= typemax(Int) - total || throw(ArgumentError("absolute KS count total exceeds Int range"))
            total += n
        end
        return _ks_sample_count(total)
    end
    # Exact binary/rational accumulation is needed here: floating summation can
    # incorrectly turn [1, eps()/4] or [2^52, .5] into integer totals.
    total = zero(Rational{BigInt})
    for w in weights
        total += _ks_count_fraction(w)
        total <= typemax(Int) || throw(ArgumentError("absolute KS count total exceeds Int range"))
    end
    _ks_sample_count(total)
end

function _ks_comparison_pvalue(delta, nx::Int, ny::Int)
    a, b = minmax(Float64(nx), Float64(ny))
    effective_n = a / (1 + a / b)
    ccdf(HypothesisTests.Kolmogorov(), sqrt(effective_n) * delta)
end

"""
    bat_compare(
        samples_1::DensitySampleVector,
        samples_2::DensitySampleVector;
        nsamples::Symbol=:effective
    )

*Experimental feature, not part of stable public API.*

Compares two `DensitySampleVector`s given by `samples_1` and `samples_2` applying the Kolmogorov-Smirnov test for all marginals.

See [N. Smirnov, "Table for Estimating the Goodness of Fit of Empirical
Distributions" (1948)](https://doi.org/10.1214/aoms/1177730256).

`nsamples` specifies how to define a number of samples in the Kolmogorov-Smirnov
distribution. The default value is `nsamples=:effective`, which uses the
effective number of samples estimated by `bat_eff_sample_size`. The optimal keywords:

* `:length` — each sample vector's own row count, including zero-weight rows.
* `:weights` — the original absolute weight sum, which must be a positive
  integer representable as `Int`. Integer, rational and ordinary binary floating
  frequency weights are supported. Fractional wider floats must be exactly
  representable as Float64; logarithmic count encodings are not supported.
  The total is checked exactly, without normalizing or rounding it.
* `:nested` — legacy mode: population 1 uses estimated ESS; population 2 uses
  `round(nested_eff_ss / number_of_free_parameters)` for every marginal.

Estimated ESS values are rounded to positive `Int` counts; a scalar estimate
is shared by all marginals. Inputs must have at least one free parameter,
nonempty populations, finite coordinates, and finite nonnegative weights with
positive total mass. Relative CDF weights use at least Float64 precision;
unrepresentable positive relative masses are rejected. Gradual underflow and
round-to-nearest BigFloat arithmetic with at least 53 bits are required.

The distance compares complete tied-coordinate groups and is invariant to
independent positive weight scaling within floating precision. This does not
make `:weights` sample counts or p-values scale invariant. P-values use the
asymptotic continuous-null KS distribution with the selected counts; ties,
correlation or arbitrary importance weights do not have exact calibration.

Returns a NamedTuple of the shape

```julia
(result = X::TypedTables.Table, ...)
```
"""
function bat_compare(samples_1::DensitySampleVector, samples_2::DensitySampleVector;
    nsamples::Symbol=:effective, nested_eff_ss=0)
    context = _g_dummy_context
    (isempty(samples_1) || isempty(samples_2)) &&
        throw(ArgumentError("KS comparison requires nonempty populations"))
    samples_1_flat = flatview(unshaped.(samples_1.v))
    samples_2_flat = flatview(unshaped.(samples_2.v))
    n_params = size(samples_1_flat, 1)
    n_params == size(samples_2_flat, 1) ||
        throw(ArgumentError("Samples with different number of parameters cannot be compared"))
    n_params > 0 || throw(ArgumentError("KS comparison requires at least one free parameter"))
    all(isfinite, samples_1_flat) && all(isfinite, samples_2_flat) ||
        throw(ArgumentError("KS coordinates must be finite"))
    ux, uy = _ks_relative_weights(samples_1.weight), _ks_relative_weights(samples_2.weight)

    if nsamples == :effective
        smpl_size_1 = _ks_effective_counts(bat_eff_sample_size(unshaped.(samples_1), context).result, n_params)
        smpl_size_2 = _ks_effective_counts(bat_eff_sample_size(unshaped.(samples_2), context).result, n_params)
    elseif nsamples == :nested
        nested_eff_ss isa Real && isfinite(nested_eff_ss) && nested_eff_ss > 0 ||
            throw(ArgumentError("positive finite nested_eff_ss is required for :nested comparison"))
        smpl_size_1 = _ks_effective_counts(bat_eff_sample_size(unshaped.(samples_1), context).result, n_params)
        smpl_size_2 = fill(_ks_sample_count(nested_eff_ss / n_params; rounded=true), n_params)
    elseif nsamples == :length
        smpl_size_1 = fill(length(samples_1), n_params)
        smpl_size_2 = fill(length(samples_2), n_params)
    elseif nsamples == :weights
        smpl_size_1 = fill(_ks_absolute_count(samples_1.weight), n_params)
        smpl_size_2 = fill(_ks_absolute_count(samples_2.weight), n_params)
    else
        throw(ArgumentError("unknown KS sample-count policy: $nsamples"))
    end

    p_values = map(Base.OneTo(n_params)) do i
        delta = first(_ksstats_prepared(view(samples_1_flat, i, :), view(samples_2_flat, i, :), ux, uy))
        _ks_comparison_pvalue(delta, smpl_size_1[i], smpl_size_2[i])
    end
    table_result = TypedTables.Table(
        dim_ind=Base.OneTo(n_params), nsamples_1=smpl_size_1,
        nsamples_2=smpl_size_2, ks_p_values=p_values)
    return (result=table_result,)
end

export bat_compare
