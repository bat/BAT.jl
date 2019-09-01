struct HistogramAsUvDistribution{T <: AbstractFloat} <: ContinuousUnivariateDistribution
    h::Histogram{<:Real, 1}
    inv_weights::Vector{T} 
    edges::Vector{T}
    volumes::Vector{T}

    _edges::Vector{T}
    _volumes::Vector{T}
    _inv_volumes::Vector{T}
    
    _acc_prob::Vector{T}

    mean::T
    var::T
end

function HistogramAsUvDistribution(h::Histogram{<:Real, 1}, T::DataType = Float64)
    nh = normalize(h)
    _widths::Vector{T} = h.weights * inv(sum(h.weights))
    _edges::Vector{T} = Vector{Float64}(undef, length(_widths) + 1)
    _edges[1] = 0
    @inbounds for (i, w) in enumerate(_widths)
        _edges[i+1] = _edges[i] + _widths[i]
    end
    _edges[end] = 1
    volumes = diff(h.edges[1])
    mean = Statistics.mean(StatsBase.midpoints(nh.edges[1]), ProbabilityWeights(nh.weights))
    var = Statistics.var(StatsBase.midpoints(nh.edges[1]), ProbabilityWeights(nh.weights), mean = mean)

    _acc_prob::Vector{T} = zeros(T, length(nh.weights))
    for i in 2:length(_acc_prob)
        _acc_prob[i] += _acc_prob[i-1] + nh.weights[i-1] * volumes[i-1]   
    end

    d::HistogramAsUvDistribution{T} = HistogramAsUvDistribution{T}(
        nh,
        inv.(nh.weights),
        nh.edges[1],
        volumes,
        _edges,
        _widths,
        inv.(_widths),
        _acc_prob,
        mean, 
        var
    )
end 

function Base.rand(rng::AbstractRNG, d::HistogramAsUvDistribution{T})::T where {T <: AbstractFloat}
    _r::T = rand()
    next_inds::UnitRange{Int} = searchsorted(d._edges, _r)
    next_ind_l::Int = next_inds.start
    next_ind_r::Int = next_inds.stop
    if next_ind_l > next_ind_r 
        next_ind_l = next_inds.stop
        next_ind_r = next_inds.start
    end
    r::T = d.edges[next_ind_l]
    if next_ind_l < next_ind_r
        r += d.volumes[next_ind_l] * (d._edges[next_ind_r] - _r) * d._inv_volumes[next_ind_l] 
    end
    return r
end

function Distributions.pdf(d::HistogramAsUvDistribution{T}, x::Real)::T where {T <: AbstractFloat}
    i::Int = StatsBase.binindex(d.h, x)
    return @inbounds d.h.weights[i]
end

function Distributions.logpdf(d::HistogramAsUvDistribution{T}, x::Real)::T where {T <: AbstractFloat}
    return log(pdf(d, x))
end

function Distributions.cdf(d::HistogramAsUvDistribution{T}, x::Real)::T where {T <: AbstractFloat}
    i::Int = StatsBase.binindex(d.h, x)
    p::T = @inbounds sum(d.h.weights[1:i-1] .* d.volumes[1:i-1])
    p += (x - d.edges[i]) * d.h.weights[i] 
    return p
end

function Distributions.minimum(d::HistogramAsUvDistribution{T})::T where {T <: AbstractFloat} 
    d.edges[1]
end

function Distributions.maximum(d::HistogramAsUvDistribution{T})::T where {T <: AbstractFloat} 
    d.edges[end]
end

function Distributions.insupport(d::HistogramAsUvDistribution{T}, x::Real)::Bool where {T <: AbstractFloat} 
    d.edges[1] <= x <= d.edges[end]
end

function Distributions.quantile(d::HistogramAsUvDistribution{T}, x::Real)::T where {T <: AbstractFloat} 
    _range::UnitRange{Int} = searchsorted(d._acc_prob, x)
    _idx::Int = min(_range.start, _range.stop)
    p::T = d._acc_prob[ _idx ]
    q::T = d.edges[_idx]
    missing_p::T = x - p
    inv_weight::T = d.inv_weights[_idx] 
    if !isinf(inv_weight) 
        q += missing_p * inv_weight 
    end
    return min(q, maximum(d))
end


Base.eltype(d::HistogramAsUvDistribution{T}) where {T <: AbstractFloat}= T

_np_bounds(d::HistogramAsUvDistribution) = 
    HyperRectBounds(Vector{eltype(d)}([quantile(d, 0)]), Vector{eltype(d)}([quantile(d, 1)]), hard_bounds)

Statistics.mean(d::HistogramAsUvDistribution) = d.mean
Statistics.var(d::HistogramAsUvDistribution) = d.var
