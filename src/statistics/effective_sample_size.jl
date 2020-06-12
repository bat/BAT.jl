# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc doc"""
    wgt_effective_sample_size(w::AbstractVector{T})

*BAT-internal, not part of stable public API.*

Kish's approximation for weighted samples effective_sample_size estimation.
Computes the weighting factor for weigthed samples, where w is the vector of
weigths.
"""
function wgt_effective_sample_size(w::AbstractVector{T}) where T<:Real
    return sum(w)^2/sum(w.^2)
end


@doc doc"""
    effective_sample_size(xv, w; ...)

*BAT-internal, not part of stable public API.*

Effective size estimation for a vector of samples xv. If a weight vector w is provided,
the Kish approximation is applied.

By default computes the autocorrelation up to the square root of the number of entries
in the vector, unless an explicit list of lags is provided (kv).
"""
function effective_sample_size(xv::AbstractVector{<:Real}, w::AbstractVector{<:Real} = Vector{Int}())
    #kv = collect(1:floor(Int,sqrt(length(xv))))
    #atc = StatsBase.autocor(xv,kv)
    atc = bat_autocorr(xv).result

    # we need to break the sum when ρ_k + ρ_k+1 < 0
    # (see Thompson2010 - arXiv1011.0175 and Geyer2002)
    sumatc = 0
    for k in 1:(length(atc)-1)
        if atc[k]+atc[k+1] >= 0
            sumatc = sumatc+atc[k]
        else
            break
        end
    end
    result = size(xv)[1]/(1 + 2*sumatc)
    w_correction = 1.0
    if size(w) == size(xv)
        w_correction = wgt_effective_sample_size(w)/length(eachindex(w))
    end
    return min(length(xv),result*w_correction)
end

@doc doc"""
    effective_sample_size(variates::AbstractArray, weights::AbstractVector; with_weights=true)

*BAT-internal, not part of stable public API.*

Effective size estimation for a (multidimensional) ElasticArray.
By default applies the Kish approximation with the weigths available, but
can be turned off (with_weights=false).
"""
function effective_sample_size(variates::AbstractArray, weights::AbstractVector; with_weights=true)
        ess = size(variates, 2)
        for dim in axes(variates, 1)
            tmpview = view(variates,dim,:)
            tmp = with_weights ?
                effective_sample_size(tmpview, weights) : effective_sample_size(tmpview)
            if tmp < ess
                ess = tmp
            end
        end
        return ess
    end

@doc doc"""
    effective_sample_size(samples::DensitySampleVector; with_weights=true)

*BAT-internal, not part of stable public API.*

Effective size estimation for a (multidimensional) DensitySampleVector.
By default applies the Kish approximation with the weigths available, but
can be turned off (with_weights=false).
"""
function effective_sample_size(samples::DensitySampleVector; with_weights=true)
    return effective_sample_size(flatview(unshaped.(samples.v)), samples.weight, with_weights=with_weights)
end
