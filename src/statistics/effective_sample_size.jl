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


function _ess_from_atc(atc::AbstractVector{<:Real})
    # we need to break the sum when ρ_k + ρ_k+1 < 0
    # (see Thompson2010 - arXiv1011.0175 and Geyer2002)
    sumatc = 0
    for k in 1:(length(atc)-1)
        if atc[k] + atc[k+1] >= 0
            sumatc = sumatc + atc[k]
        else
            break
        end
    end
    ess = length(atc)/(1 + 2 * sumatc)
    return min(length(atc), ess)
end


@doc doc"""
    bat_eff_sample_size(v::AbstractVector)
    bat_eff_sample_size(smpls::DensitySampleVector; use_weights=true)

Estimate effective sample size estimation for variate series `v`, resp.
density samples `smpls`, separately for each degree of freedom.

* `use_weights`: Take sample weights into account, using Kish's approximation

Returns a NamedTuple of the shape

```julia
(result = X::AbstractVector{<:Real}, ...)
```

Result properties not listed here are algorithm-specific and are not part
of the stable BAT API.
"""
function bat_eff_sample_size end
export bat_eff_sample_size

function bat_eff_sample_size(v::AbstractVector; use_weights = true)
    atc = bat_autocorr(v).result
    flat_atc = flatview(atc)

    ess = map(axes(flat_atc, 1)) do i
        atcview = view(flat_atc, i, :)
        _ess_from_atc(atcview)
    end

    (result = ess,)
end

function bat_eff_sample_size(smpls::DensitySampleVector; use_weights = true)
    ess = bat_eff_sample_size(samples.v).result

    if use_weights
        W = smpls.weight
        w_correction = wgt_effective_sample_size(W) / length(eachindex(W))
        ess .*= w_correction
    end

    (result = ess,)
end
