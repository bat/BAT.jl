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
    fft_autocov(v::AbstractVector{<:Real})
    fft_autocov(v::AbstractVectorOfSimilarVectors{<:Real})

Compute the autocovariance of of variate series `v`, separately for each
degree of freedom.

Uses FFT, in contract to `StatsBase.autocov`.
"""
function fft_autocov end

function fft_autocov(x::AbstractVector{<:Real})
    n = length(eachindex(x))
    n2 = 2 * _ac_next_pow_two(n)
    x2 = zeros(eltype(x), n2)
    idxs2 = firstindex(x2):(firstindex(x2) + n - 1)
    x2_view = view(x2, idxs2)
    x2_view .= x .- mean(x)

    x2_fft = rfft(x2)
    x2_fft .= abs2.(x2_fft)
    acf = irfft(x2_fft, size(x2, 1))[idxs2]

    acf .*= inv(n)

    acf
end

function fft_autocov(v::AbstractVectorOfSimilarVectors{<:Real})
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

    acf .*= inv(n)

    nestedview(acf)
end


"""
    fft_autocor(v::AbstractVector{<:Real})
    fft_autocor(v::AbstractVectorOfSimilarVectors{<:Real})

Compute the autocorrelation function (ACF) of variate series `v`, separately
for each degree of freedom.

Uses FFT, in contract to `StatsBase.autocor`.
"""
function fft_autocor end

function fft_autocor(v::AbstractVector{<:Real})
    acf = fft_autocov(v)
    acf ./= first(acf)
    acf
end

function fft_autocor(v::AbstractVectorOfSimilarVectors{<:Real})
    acf = flatview(fft_autocov(v))
    acf ./= acf[:, first(axes(acf, 2))]
    nestedview(acf)
end
