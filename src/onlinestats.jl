# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Base.@propagate_inbounds
using StatsBase
using DoubleDouble


# SIMD-compatible KBN-summation

"""
    kbn_add(
        a::NTuple{2, Real},
        b::Real
    )


Add `b` to `a` via *SIMD-compatible* Kahan-Babuška-Neumaier summation.

"""
@inline function kbn_add(a::NTuple{2, Real}, b::Real)
    s = a[1] + b
    c = a[2] + ifelse(
        abs(a[1]) >= abs(b),
        (a[1] - s) + b,
        (b - s) + a[1]
    )
    (s, c)
end

"""
    kbn_add(
        a::NTuple{2, Real},
        b::NTuple{2, Real}
    )


Add `b` to `a` via *SIMD-compatible* Kahan-Babuška-Neumaier summation.
"""
@inline function kbn_add(a::NTuple{2, Real}, b::NTuple{2, Real})
    s = a[1] + b[1]
    c = ifelse(
        abs(a[1]) > abs(b[1]),
        (((a[1] - s) + b[1]) + b[2]) + a[2],
        (((b[1] - s) + a[1]) + a[2]) + b[2]
    )
    (s, c)
end



"""
    OnlineMvMean{T<:AbstractFloat} <: AbstractVector{T}

Multi-variate mean implemented via Kahan-Babuška-Neumaier summation.
"""
mutable struct OnlineMvMean{T<:AbstractFloat} <: AbstractVector{T}
    m::Int
    sum_w::Double{T}
    S::Vector{T}
    C::Vector{T}

    OnlineMvMean{T}(m::Integer) where {T<:AbstractFloat} =
        new{T}(m, zero(Int64), zeros(T, m), zeros(T, m))
end

export OnlineMvMean

OnlineMvMean(m::Integer) = OnlineMvMean{Float64}(m::Integer)

"""
    Base.size(
        omn::OnlineMvMean
        )

Return number of moving means of `omn`
"""
@inline Base.size(omn::OnlineMvMean) = size(omn.S)

"""
    
    Base.getindex{T}(
        omn::OnlineMvMean{T},
        idxs::Integer...)

Compute moving means at indices `idxs` 
"""
@propagate_inbounds function Base.getindex{T}(omn::OnlineMvMean{T}, idxs::Integer...)
    T((Single(omn.S[idxs...]) + Single(omn.C[idxs...])) / omn.sum_w)
end


"""
    Base.push!(
        omn::OnlineMvMean,
        data::Vector,
        weight::Real = one(T)
    )

Add `data` to `omn` weighted with `weight`
"""
@inline Base.push!{T}(omn::OnlineMvMean{T}, data::Vector, weight::Real = one(T)) =
    push_contiguous!(omn, data, first(linearindices(data)), weight)

"""
    Base.merge!(
        target::OnlineMvMean, 
        others::OnlineMvMean...
    )

    Merge `others` to `target`. All moving means must be of same size.
"""
function Base.merge!(target::OnlineMvMean, others::OnlineMvMean...)
    for x in others
        target.m != x.m && throw(ArgumentError("can't merge OnlineMvMean instances with different size"))
    end
    for x in others
        target.sum_w += x.sum_w
        target_S = target.S; target_C = target.C
        x_S = x.S; x_C = x.C
        @assert eachindex(target_S) == eachindex(x.S) == eachindex(target_C) == eachindex(x.C)
        @inbounds @simd for i in eachindex(target_S)
            target_S[i], target_C[i] = kbn_add(
                (target_S[i], target_C[i]),
                (x.S[i], x.C[i])
            )
        end
    end
    target
end

"""
    Base.merge(
        x::OnlineMvMean,
        others::OnlineMvMean...
    )

    Merges `others` to a `deepcopy` of `target`. All moving means must be of same size.
"""
Base.merge(x::OnlineMvMean, others::OnlineMvMean...) = merge!(deepcopy(x), others...)


"""
    push_contiguous!{T}(
        omn::OnlineMvMean,
        data::Vector,
        weight::Real = one(T)
    )

Adds `data` to `omn` weighted with `weight`
"""
@inline function push_contiguous!{T}(
    omn::OnlineMvMean{T}, data::Array,
    start::Integer, weight::Real = one(T)
)
    m = omn.m
    S = omn.S
    C = omn.C

    idxs = Base.OneTo(m)

    dshft = Int(start) - 1

    @assert idxs == indices(S, 1) == indices(C, 1)
    checkbounds(data, idxs + dshft)

    omn.sum_w += Single(weight)
    
    @inbounds @simd for i in idxs
        x = weight * data[i + dshft]
        S[i], C[i] = kbn_add((S[i], C[i]), x)
    end

    omn
end



"""
    OnlineMvCov{T<:AbstractFloat,W} <: AbstractMatrix{T}

Implementation based on variance calculation Algorithms of Welford and West.

`W` must either be `Weights` (no bias correction) or one of `AnalyticWeights`,
`FrequencyWeights` or `ProbabilityWeights` to specify the desired bias
correction method.
"""

mutable struct OnlineMvCov{T<:AbstractFloat,W} <: AbstractMatrix{T}
    m::Int
    n::Int64
    sum_w::Double{T}
    sum_w2::Double{T}
    Mean_X::Vector{T}
    New_Mean_X::Vector{T}
    S::Matrix{T}

    OnlineMvCov{T,W}(m::Integer) where {T<:AbstractFloat,W} =
        new{T,W}(
            m, zero(Int64), zero(Double{T}), zero(Double{T}),
            zeros(T, m), zeros(T, m), zeros(T, m, m)
        )
end

export OnlineMvCov

OnlineMvCov(m::Integer) = OnlineMvCov{Float64, ProbabilityWeights}(m::Integer)

"""
    Base.size(
        ocv::OnlineMvCov)
    )

Return dimensions of moving covariance matrix
"""
@inline Base.size(ocv::OnlineMvCov) = size(ocv.S)

"""
    function Base.getindex{T}(
        ocv::OnlineMvCov{T, Weights},
        idxs::Integer...
    )

Computes covariances at indices `idxs` wiht no bias correction.
"""
@propagate_inbounds function Base.getindex{T}(ocv::OnlineMvCov{T, Weights}, idxs::Integer...)
    sum_w = ocv.sum_w
    ifelse(
        sum_w > 0,
        T(ocv.S[idxs...] / sum_w),
        T(NaN)
    )
end

"""
    function Base.getindex{T}(
        ocv::OnlineMvCov{T, AnalyticWeights},
        idxs::Integer...
    )

Computes covariances at indices `idxs`.
`AnalyticWeights`: ``\\frac{1}{\\sum w - \\sum {w^2} / \\sum w}``
"""
@propagate_inbounds function Base.getindex{T}(ocv::OnlineMvCov{T, AnalyticWeights}, idxs::Integer...)
    sum_w = ocv.sum_w
    sum_w2 = ocv.sum_w2
    d = sum_w - sum_w2 / sum_w
    ifelse(
        sum_w > 0 && d > 0,
        T(ocv.S[idxs...] / d),
        T(NaN)
    )
end

"""
    function Base.getindex{T}(
        ocv::OnlineMvCov{T, FrequencyWeights},
        idxs::Integer...
    )

Computes covariances at indices `idxs`.
`FrequencyWeights`: ``\\frac{1}{\\sum{w} - 1}``
"""
@propagate_inbounds function Base.getindex{T}(ocv::OnlineMvCov{T, FrequencyWeights}, idxs::Integer...)
    sum_w = ocv.sum_w
    ifelse(
        sum_w > 1,
        T(ocv.S[idxs...] / (sum_w - 1)),
        T(NaN)
    )    
end

"""
    function Base.getindex{T}(
        ocv::OnlineMvCov{T, ProbabilityWeights},
        idxs::Integer...
    )

Computes covariances at indices `idxs`.
`ProbabilityWeights`: ``\\frac{n}{(n - 1) \\sum w}`` where ``n`` equals `count(!iszero, w)`
"""
@propagate_inbounds function Base.getindex{T}(ocv::OnlineMvCov{T, ProbabilityWeights}, idxs::Integer...)
    n = ocv.n
    sum_w = ocv.sum_w
    ifelse(
        n > 1 && sum_w > 0,
        T(ocv.S[idxs...] * n / ((n - 1) * sum_w)),
        T(NaN)
    )
end



@inline Base.push!{T, W}(ocv::OnlineMvCov{T, W}, data::Vector, weight::Real = one(T)) =
    push_contiguous!(ocv, data, first(linearindices(data)), weight)



# function Base.merge!(a::OnlineMvCov, b::OnlineMvCov)
#     ...
# end

function Base.merge!{T,W}(target::OnlineMvCov{T,W}, others::OnlineMvCov{T,W}...)
    for x in others
        target.m != x.m && throw(ArgumentError("can't merge OnlineMvCov instances with different size"))
    end
    m = target.m
    dx = Array{T}(m)
    for x in others
        S = target.S
        idxs = Base.OneTo(m)
        @inbounds @simd for i in idxs
            dx[i] = target.Mean_X[i] - x.Mean_X[i]
        end
        scl = (target.sum_w * x.sum_w)/(target.sum_w + x.sum_w)
        @inbounds for j in idxs
            j_offs = sub2ind(S, 0, j)
            @assert sub2ind(S, last(idxs), j) == last(idxs) + j_offs
            @simd for i in idxs
                S[i + j_offs] = muladd(dx[i], scl * dx[j], S[i + j_offs] + x.S[i + j_offs])
            end
        end

        d = inv(T(target.sum_w + x.sum_w))
        @inbounds @simd for i in idxs
            target.Mean_X[i] = T((target.Mean_X[i] * target.sum_w
                                  + x.Mean_X[i]*x.sum_w)*d)
            target.New_Mean_X[i] = target.Mean_X[i]
        end
        
        target.n += x.n
        target.sum_w += x.sum_w
        target.sum_w2 += x.sum_w2
    end
    target
end

Base.merge(x::OnlineMvCov, others::OnlineMvCov...) = merge!(deepcopy(x), others...)


@inline function push_contiguous!{T,W}(
    ocv::OnlineMvCov{T,W}, data::Array,
    start::Integer,
    weight::Real = one(T)
)
    m = ocv.m
    n = ocv.n
    sum_w = ocv.sum_w
    sum_w2 = ocv.sum_w2
    Mean_X = ocv.Mean_X
    New_Mean_X = ocv.New_Mean_X
    S = ocv.S

    idxs = Base.OneTo(m)

    dshft = Int(start) - 1

    @assert idxs == indices(Mean_X, 1) == indices(New_Mean_X, 1) == indices(S, 1) == indices(S, 2)
    checkbounds(data, idxs + dshft)

    n += one(n)
    sum_w += Single(weight)
    sum_w2 += Single(weight^2)

    sum_w_inv = inv(T(sum_w))

    @inbounds @simd for i in idxs
        x = data[i + dshft]
        mean_X = Mean_X[i]
        New_Mean_X[i] = muladd(x - mean_X, weight * sum_w_inv, mean_X)
    end

    @inbounds for j in idxs
        new_dx_j = data[j + dshft] - New_Mean_X[j]
        j_offs = sub2ind(S, 0, j)
        @assert sub2ind(S, last(idxs), j) == last(idxs) + j_offs
        @simd for i in idxs
            dx_i = data[i + dshft] - Mean_X[i]
            S[i + j_offs] = muladd(dx_i, weight * new_dx_j, S[i + j_offs])
        end
    end

    @inbounds @simd for i in idxs
        Mean_X[i] = New_Mean_X[i]
    end

    ocv.n = n
    ocv.sum_w = sum_w
    ocv.sum_w2 = sum_w2

    ocv
end



const OnlineStatistic = Union{OnlineMvMean, OnlineMvCov}


function Base.append!(target::OnlineStatistic, data::Matrix, vardim::Integer = 1)
    if (vardim == 1)
        throw(ArgumentError("vardim == $vardim not supported (yet)"))
    elseif (vardim == 2)
        @assert target.m == size(data, 1)  # TODO: Replace by exception
        @inbounds for i in indices(data, 2)
            push_contiguous!(target, data, sub2ind(data, 1, i))
        end
    else
        throw(ArgumentError("Value of vardim must be 2, not $vardim"))
    end

    target
end


function Base.append!(target::OnlineStatistic, data::Matrix, weights::Vector, vardim::Integer = 1)
    if (vardim == 1)
        throw(ArgumentError("vardim == $vardim not supported (yet)"))
    elseif (vardim == 2)
        @assert target.m == size(data, 1)  # TODO: Replace by exception
        @assert indices(data, 2) == indices(weights, 1)  # TODO: Replace by exception
        @inbounds for i in indices(data, 2)
            push_contiguous!(target, data, sub2ind(data, 1, i), weights[i])
        end
    else
        throw(ArgumentError("Value of vardim must be 2, not $vardim"))
    end

    target
end
