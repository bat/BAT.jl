# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc """
    autocrl(xv::AbstractVector{T}, kv::AbstractVector{Int} = Vector{Int}())

autocorrelation := Σ Cov[x_i,x_(i+k)]/Var[x]

Computes the autocorrelations at various leg k of the
input vector (time series) xv.
The vector kv is the collections of lags to take into account
"""
function autocrl(xv::AbstractVector{T}, kv::AbstractVector{Int} = Vector{Int}()) where T<:Real
       N = size(xv)[1]
       if size(kv)[1] == 0
           kv = 1:(N-1)
       end
       x_avg = sum(xv)/N
       result = Vector{Float64}(zeros(size(kv)))
       for k in kv
           autocrl_num = 0.0
           autocrl_den = 0.0
           for i in 1:N-k
               autocrl_num += (xv[i+k] - x_avg)*(xv[i]-x_avg)
               autocrl_den += (xv[i]-x_avg)^2
           end
           result[k] = autocrl_num/autocrl_den
       end
       return result
end


@doc """
    wgt_ess(w::AbstractVector{T})

Kish's approximation for weighted samples ESS estimation.
Computes the weighting factor for weigthed samples, where w is the vector of
weigths.
"""
function wgt_ess(w::AbstractVector{T}) where T<:Real
    return sum(w)^2/sum(w.^2)
end


@doc """
Effective size estimation for a vector of samples xv. If a weight vector w is provided,
the Kish approximation is applied.

By default computes the autocorrelation up to the square root of the number of entries
in the vector, unless an explicit list of lags is provided (kv).
"""
function ESS(xv::AbstractVector{T1}, w::AbstractVector{T2} = Vector{Float64}(zeros(0)), kv::AbstractVector{Int} = Vector{Int}(1:floor(Int,sqrt(length(xv))))) where T1<:Real where T2<:Number
    atc = StatsBase.autocor(xv,kv)
    result = size(xv)[1]/(1 + 2*sum(atc))
    w_correction = 1.0
    if size(w) == size(xv)
        w_correction = wgt_ess(w)/size(w)[1]
    end
    return min(length(xv),result*w_correction)
end

@doc """
    ESS(params::AbstractArray, weights::AbstractVector; with_weights=true)

Effective size estimation for a (multidimensional) ElasticArray.
By default applies the Kish approximation with the weigths available, but
can be turned off (with_weights=false).
"""
function ESS(params::AbstractArray, weights::AbstractVector; with_weights=true)
        ess = size(params, 2)
        for dim in axes(params, 1)
            tmpview = view(params,dim,:)
            tmp = with_weights ?
                ESS(tmpview, weights) : ESS(tmpview)
            if tmp < ess
                ess = tmp
            end
        end
        return ess
    end

@doc """
    ESS(samples::DensitySampleVector; with_weights=true)

Effective size estimation for a (multidimensional) DensitySampleVector.
By default applies the Kish approximation with the weigths available, but
can be turned off (with_weights=false).
"""
function ESS(samples::DensitySampleVector; with_weights=true)
    return ESS(samples.params, samples.weight, with_weights=with_weights)
end
