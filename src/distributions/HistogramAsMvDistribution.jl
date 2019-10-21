# This file is a part of BAT.jl, licensed under the MIT License (MIT).

struct HistogramAsMvDistribution{T, N} <: Distributions.ContinuousMultivariateDistribution
    h::StatsBase.Histogram{<:Real, N}
    edges::NTuple{N, <:AbstractVector{T}}
    cart_inds::CartesianIndices{N, NTuple{N, Base.OneTo{Int}}}

    probabilty_edges::AbstractVector{T}

    μ::AbstractVector{T}
    var::AbstractVector{T}
    cov::AbstractMatrix{T}
end

Base.length(d::HistogramAsMvDistribution{T, N}) where {T, N} = N
Base.size(d::HistogramAsMvDistribution{T, N}) where {T, N} = (N,)
Base.eltype(d::HistogramAsMvDistribution{T, N}) where {T, N} = T

Statistics.mean(d::HistogramAsMvDistribution{T, N}) where {T, N} = d.μ
Statistics.var(d::HistogramAsMvDistribution{T, N}) where {T, N} = d.var
Statistics.cov(d::HistogramAsMvDistribution{T, N}) where {T, N} = d.cov

_np_valshape(d::HistogramAsMvDistribution{T, N}) where {T, N} = ArrayShape{Real}(size(d)...)

function _np_bounds(d::HistogramAsMvDistribution{T, N}) where {T, N} 
    left_bounds  = [ first(d.h.edges[i]) for i in 1:N ]
    right_bounds = [  prevfloat(last(d.h.edges[i])) for i in 1:N ]
    HyperRectBounds(T.(left_bounds), T.(right_bounds), hard_bounds)
end

function HistogramAsMvDistribution(h::StatsBase.Histogram{<:Real, N}, T::DataType = Float64) where {N}
    nh = normalize(h)

    probabilty_widths = nh.weights * inv(sum(nh.weights))
    probabilty_edges::Vector{T} = Vector{Float64}(undef, length(h.weights) + 1)  
    probabilty_edges[1] = 0
    for (i, w) in enumerate(probabilty_widths)
        v = probabilty_edges[i] + probabilty_widths[i]
        probabilty_edges[i+1] = v > 1 ? 1 : v
    end

    mean = _mean(h)
    var = _var(h, mean = mean)
    cov = _cov(h, mean = mean)

    return HistogramAsMvDistribution{T, N}(
        nh,
        collect.(nh.edges),
        CartesianIndices(nh.weights),
        probabilty_edges,
        mean,
        var,
        cov
    )
end


function _mean(h::StatsBase.Histogram{<:Real, N}; T::DataType = Float64) where {N}
    s_inv::T = inv(sum(h.weights))
    m::Vector{T} = zeros(T, N)
    mps = StatsBase.midpoints.(h.edges)
    cart_inds = CartesianIndices(h.weights)
    for i in cart_inds
        for idim in 1:N
            m[idim] += s_inv * mps[idim][i[idim]] * h.weights[i]
        end
    end
    return m
end

function _var(h::StatsBase.Histogram{<:Real, N}; T::DataType = Float64, mean = StatsBase.mean(h, T = T), ) where {N}
    s_inv::T = inv(sum(h.weights))
    v::Vector{T} = zeros(T, N)
    mps = StatsBase.midpoints.(h.edges)
    cart_inds = CartesianIndices(h.weights)
    for i in cart_inds
        for idim in 1:N
            v[idim] += s_inv * (mps[idim][i[idim]] - mean[idim])^2 * h.weights[i]
        end
    end
    return v
end

function _cov(h::StatsBase.Histogram{<:Real, N}; T::DataType = Float64, mean = StatsBase.mean(h, T = T)) where {N}
    s_inv::T = inv(sum(h.weights))
    c::Matrix{T} = zeros(T, N, N)
    mps = StatsBase.midpoints.(h.edges)
    cart_inds = CartesianIndices(h.weights)
    for i in cart_inds
        for idim in 1:N
            for jdim in 1:N
                c[idim, jdim] += s_inv * (mps[idim][i[idim]] - mean[idim]) * (mps[jdim][i[jdim]] - mean[jdim]) * h.weights[i]
            end
        end
    end
    return c
end


function Distributions._rand!(r::AbstractRNG, d::HistogramAsMvDistribution{T,N}, A::AbstractVector{<:Real}) where {T, N}
    rand!(r, A)
    next_inds::UnitRange{Int} = searchsorted(d.probabilty_edges::Vector{T}, A[1]::T)
    cell_lin_index::Int = min(next_inds.start, next_inds.stop)
    cell_car_index = d.cart_inds[cell_lin_index]
    for idim in Base.OneTo(N)
        i = cell_car_index[idim]
        sub_int = d.edges[idim][i:i+1]
        sub_int_width::T = sub_int[2] - sub_int[1]
        A[idim] = sub_int[1] + sub_int_width * A[idim]
    end
    return A
end

function Distributions._rand!(r::AbstractRNG, d::HistogramAsMvDistribution{T,N}, A::AbstractMatrix{<:Real}) where {T, N}
    Distributions._rand!.((r,), (d,), nestedview(A))
end

function Distributions.pdf(d::HistogramAsMvDistribution{T, N}, x::AbstractArray{<:Real, 1}) where {T, N}
    return @inbounds d.h.weights[StatsBase.binindex(d.h, Tuple(x))...]
end
function Distributions.logpdf(d::HistogramAsMvDistribution{T, N}, x::AbstractArray{<:Real, 1}) where {T, N}
    return log(pdf(d, x))
end
function Distributions._logpdf(d::HistogramAsMvDistribution{T,N}, x::AbstractArray{<:Real, 1}) where {T, N}
    return logpdf(d, x)
end


