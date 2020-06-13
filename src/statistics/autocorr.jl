# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# Port of the autocorrelation length estimation of the emcee Python package,
# under MIT License. Original authors Dan Foreman-Mackey et al.
#
# See also:
#
# https://dfm.io/posts/autocorr/
# https://github.com/dfm/emcee/
# https://github.com/dfm/emcee/issues/209


function _ac_next_pow_two(i::Integer)
    nbits = 8 * sizeof(i)
    1 << (nbits - leading_zeros(i > 0 ? i - 1 : i))
end


"""
    bat_autocorr(v::AbstractVector)

Estimate the normalized autocorrelation function of variate series `v`,
separately for each degree of freedom.

Derived from the FFT-based autocorrelation implementation in the emcee Python
package, under MIT License (original authors Dan Foreman-Mackey et al.).

Returns a NamedTuple of the shape

```julia
(result = autocorr, ...)
```

Result properties not listed here are algorithm-specific and are not part
of the stable BAT API.
"""
function bat_autocorr end
export bat_autocorr

function bat_autocorr(x::AbstractVector{<:Real})
    n = length(eachindex(x))
    n2 = 2 * _ac_next_pow_two(n)
    x2 = zeros(eltype(x), n2)
    idxs2 = firstindex(x2):(firstindex(x2) + n - 1)
    x2_view = view(x2, idxs2)
    x2_view .= x .- mean(x)

    x2_fft = rfft(x2)
    x2_fft .= abs2.(x2_fft)
    acf = irfft(x2_fft, size(x2, 1))[idxs2]
    acf ./= first(acf)

    (result = acf,)
end

function bat_autocorr(v::AbstractVectorOfSimilarVectors{<:Real})
    X = flatview(v)
    n = size(X, 2)
    n2 = 2 * _ac_next_pow_two(n)
    X2 = zeros(eltype(X), size(X, 1), n2)
    idxs = axes(X2,2)
    idxs2 = first(idxs):(first(idxs) + n - 1)
    X2_view = view(X2, :, idxs2)
    X2_view .= X .- mean(X, dims = 2)

    X2_fft = rfft(X2, 2)
    X2_fft .= abs2.(X2_fft)
    acf = irfft(X2_fft, size(X2, 2), 2)[:, idxs2]
    acf ./= acf[:, first(axes(acf, 2))]

    (result = nestedview(acf),)
end

bat_autocorr(x::AbstractVector) = bat_autocorr(unshaped.(x))



function emcee_auto_window(taus::AbstractVector{<:Real}, c::Real)
    idxs = eachindex(taus)
    m = count(i - firstindex(taus) < c * taus[i] for i in idxs)
    m > 0 ? m : length(idxs) - 1
end


"""
    bat_integrated_autocorr_len(
        v::AbstractVector;
        c::Integer = 5, tol::Integer = 50, strict = true
    )

Estimate the integrated autocorrelation length of variate series `v`,
separately for each degree of freedom.

* `c`: Step size for window search.

* `tol`: Minimum number of autocorrelation times needed to trust the
  estimate.

* `strict`: Throw exception if result is not trustworthy

This estimate uses the iterative procedure described on page 16 of
[Sokal's notes](http://www.stat.unc.edu/faculty/cji/Sokal.pdf)
to determine a reasonable window size.

Ported to Julia from the emcee Python package, under MIT License. Original
authors Dan Foreman-Mackey et al.

Returns a NamedTuple of the shape

```julia
(result = integrated_autocorr_len, ...)
```

Result properties not listed here are algorithm-specific and are not part
of the stable BAT API.
"""
function bat_integrated_autocorr_len end
export bat_integrated_autocorr_len

function bat_integrated_autocorr_len(v::AbstractVector{<:Real}; c::Integer = 5, tol::Integer = 50, strict::Bool = true)
    taus = bat_autocorr(v).result
    cumsum!(taus, taus)
    taus .= 2 .* taus .- 1

    window = BAT.emcee_auto_window(taus, c)
    tau_est = taus[window]

    n_samples = length(eachindex(v))
    converged = tol * tau_est <= n_samples

    if !converged && strict
        throw(ErrorException(
            "Length of samples is shorter than $tol times integrated " *
            "autocorrelation times $tau_est"
        ))
    end
   
    (result = tau_est,)
end

function bat_integrated_autocorr_len(v::AbstractVector; c::Integer = 5, tol::Integer = 50, strict::Bool = true)
    taus = flatview(bat_autocorr(v).result)
    cumsum!(taus, taus, dims = 2)
    taus .= 2 .* taus .- 1

    tau_est = map(axes(taus, 1)) do i
        window = BAT.emcee_auto_window(view(taus, i, :), c)
        taus[i, window]
    end

    n_samples = length(eachindex(v))
    converged = tol .* tau_est .<= n_samples

    if !all(converged) && strict
        throw(ErrorException(
            "Length of samples is shorter than $tol times integrated " *
            "autocorrelation times $tau_est for some dimensions"
        ))
    end
   
    (result = tau_est,)
end
