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

Estimate the normalized autocorrelation function of variate series `v`.

Derived from the FFT-based autocorrelation implementation in the emcee Python
package, under MIT License (original authors Dan Foreman-Mackey et al.).
"""
function bat_autocorr end
export bat_autocorr


function bat_autocorr(v::AbstractVectorOfSimilarVectors{<:Real})
    X = flatview(v)
    n = size(X, 2)
    n2 = 2 * _ac_next_pow_two(n)
    X2 = zeros(eltype(X), size(X, 1), n2)
    idxs = axes(X2,2)
    idxs2 = first(idxs):(first(idxs) + n - 1)
    X2_view = view(X2, :, idxs2)
    X2_view .= X .- mean(X, dims = 2)

    X2_fft = fft(X2, 2)
    acf = real.(view(ifft(X2_fft .* conj.(X2_fft), 2), :, idxs2))
    acf ./= acf[:, first(axes(acf, 2))]

    (result = nestedview(acf),)
end


function bat_autocorr(x::AbstractVector{<:Real})
    n = length(eachindex(x))
    n2 = 2 * _ac_next_pow_two(n)
    x2 = zeros(eltype(x), n2)
    idxs2 = firstindex(x2):(firstindex(x2) + n - 1)
    x2_view = view(x2, idxs2)
    x2_view .= x .- mean(x)

    x2_fft = fft(x2)
    acf = real.(view(ifft(x2_fft .* conj.(x2_fft)), idxs2))
    acf ./= first(acf)

    (result = acf,)
end



function emcee_auto_window(taus::AbstractVector{<:Real}, c::Real)
    idxs = eachindex(taus)
    m = count(i - firstindex(taus) < c * taus[i] for i in idxs)
    m > 0 ? m : length(idxs) - 1
end


"""
    bat_integrated_autocorr_len(
        v::AbstractVectorOfSimilarVectors{<:Real};
        c::Integer = 5, tol::Integer = 50, strict = true
    )

Estimate the integrated autocorrelation length of variate series `v`.

* `c`: Step size for window search.

* `tol`: Minimum number of autocorrelation times needed to trust the
  estimate.

* `strict`: Throw exception if result is not trustworthy

This estimate uses the iterative procedure described on page 16 of
[Sokal's notes](http://www.stat.unc.edu/faculty/cji/Sokal.pdf)
to determine a reasonable window size.

Ported to Julia from the emcee Python package, under MIT License. Original
authors Dan Foreman-Mackey et al.
"""
function bat_integrated_autocorr_len end
export bat_integrated_autocorr_len

function bat_integrated_autocorr_len(v::AbstractVectorOfSimilarVectors{<:Real}; c::Integer = 5, tol::Integer = 50, strict::Bool = true)
    n_samples = length(eachindex(v))

    taus = flatview(bat_autocorr(v).result)
    cumsum!(taus, taus, dims = 2)
    taus .= 2 .* taus .- 1

    tau_est = map(axes(taus, 1)) do i
        window = BAT.emcee_auto_window(view(taus, i, :), c)
        taus[i, window]
    end

    converged = tol .* tau_est .<= n_samples

    if !all(converged) && strict
        throw(ErrorException(
            "Length of samples is shorter than $tol times integrated " *
            "autocorrelation times $tau_est for some dimensions"
        ))
    end
   
    (result = tau_est,)
end


"""
    bat_integrated_autocorr_weight(
        samples::DensitySampleVector;
        c::Integer = 5, tol::Integer = 50, strict = true
    )

Estimate the integrated autocorrelation weight of `samples`.

Uses [`bat_integrated_autocorr_len`](@ref).     
"""
function bat_integrated_autocorr_weight end
export bat_integrated_autocorr_weight

function bat_integrated_autocorr_weight(samples::DensitySampleVector; kwargs...)
    mean_w = mean(samples.weight)
    unshaped_v = unshaped.(samples.v)
    tau_f_unweighted = bat_integrated_autocorr_len(unshaped_v; kwargs...).result
    (result = mean_w * tau_f_unweighted,)
end
