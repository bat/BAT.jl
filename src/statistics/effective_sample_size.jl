# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    AutocorLenAlgorithm

Abstract type for integrated autocorrelation length estimation algorithms.
"""
abstract type AutocorLenAlgorithm end
export AutocorLenAlgorithm


"""
    bat_integrated_autocorr_len(
        v::Union{AbstractVector{<:Real},AbstractVectorOfSimilarVectors{<:Real}},
        algorithm::AutocorLenAlgorithm = GeyerAutocorLen()
    )

Estimate the integrated autocorrelation length of variate series `v`,
separately for each degree of freedom.

Returns a NamedTuple of the shape

```julia
(result = integrated_autocorr_len, ...)
```

Result properties not listed here are algorithm-specific and are not part
of the stable BAT API.
"""
function bat_integrated_autocorr_len end
export bat_integrated_autocorr_len

function bat_integrated_autocorr_len(v::Union{AbstractVector{<:Real},AbstractVectorOfSimilarVectors{<:Real}})
    bat_integrated_autocorr_len(v, GeyerAutocorLen())
end


function tau_int_from_atc end


function bat_integrated_autocorr_len(v::AbstractVector{<:Real}, algorithm::AutocorLenAlgorithm)
    atc = fft_autocor(v)
    tau_int_est = tau_int_from_atc(atc, algorithm)
    (result = tau_int_est,)
end

function bat_integrated_autocorr_len(v::AbstractVectorOfSimilarVectors{<:Real}, algorithm::AutocorLenAlgorithm)
    atc = fft_autocor(v)
    flat_atc = flatview(atc)
    tau_int_est = map(axes(flat_atc, 1)) do i
        tau_int_from_atc(view(flat_atc, i, :), algorithm)
    end
    (result = tau_int_est,)
end



"""
    GeyerAutocorLen <: AutocorLenAlgorithm
    
Integrated autocorrelation length estimation based on Geyer’s initial monotone sequence criterion

See [C. J. Geyer, "Praktical Markov Chain Monte Carlo" (1992)](https://projecteuclid.org/download/pdf_1/euclid.ss/1177011137)
and [C. J. Geyer, "Introduction to Markov Chain Monte Carlo" (2011)](https://www.semanticscholar.org/paper/1-Introduction-to-Markov-Chain-Monte-Carlo-Geyer/21a92825dcec23c743e77451ff5b5ee6b1091651).


Constructor:

```julia
GeyerAutocorLen()
```

The same algorithm is used by
[STAN (v2.21)](https://mc-stan.org/docs/2_21/reference-manual/effective-sample-size-section.html#estimation-of-effective-sample-size)
and (MCMCChains.jl (v3.0, function `ess_rhat`))[https://github.com/TuringLang/MCMCChains.jl/blob/v4.0.0/src/ess.jl#L288].
"""
struct GeyerAutocorLen <: AutocorLenAlgorithm
end

export GeyerAutocorLen


function tau_int_from_atc(atc::AbstractVector{<:Real}, algorithm::GeyerAutocorLen)
    s = zero(eltype(atc))
    Γ_min = eltype(atc)(Inf)

    i = firstindex(atc)
    while i < lastindex(atc) - 1
        Γ = min(atc[i] + atc[i+1], Γ_min)
        if Γ >= 0
            s = s + Γ
            Γ_min = Γ
        else
            break
        end
        i = i + 2
    end

    -1 + 2 * s
end



"""
    SokalAutocorLen <: AutocorLenAlgorithm
    
Integrated autocorrelation length estimation based on the automated windowing
procedure descibed in
[A. D. Sokal, "Monte Carlo Methods in Statistical Mechanics" (1996)](https://pdfs.semanticscholar.org/0bfe/9e3db30605fe2d4d26e1a288a5e2997e7225.pdf)

Constructor:

```julia
SokalAutocorLen(;c::Integer = 5)
```

* `c`: Step size for window search.

Same procedure is used by the emcee Python package (v3.0).
"""
@with_kw struct SokalAutocorLen <: AutocorLenAlgorithm
    c::Int = 5  
end

export SokalAutocorLen


function tau_int_from_atc(atc::AbstractVector{<:Real}, algorithm::SokalAutocorLen)
    c = algorithm.c
    idxs = eachindex(atc)
    idx1 = first(idxs)

    tau_int::eltype(atc) = -1
    for M in idxs
        tau_int += 2 * atc[M]
        if M - idx1 >= c * tau_int
            break
        end
    end

    tau_int
end



@doc doc"""
    bat_eff_sample_size(
        v::Union{AbstractVector{<:Real},AbstractVectorOfSimilarVectors{<:Real}},
        algorithm::AutocorLenAlgorithm = GeyerAutocorLen()
    )

    bat_eff_sample_size(
        smpls::DensitySampleVector,
        algorithm::AutocorLenAlgorithm;
        use_weights=true
    )

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


function bat_eff_sample_size(v::Union{AbstractVector{<:Real},AbstractVectorOfSimilarVectors{<:Real}}, algorithm = GeyerAutocorLen())
    tau_int = bat_integrated_autocorr_len(v, algorithm).result
    n = length(eachindex(v))
    ess = min.(n, n./ tau_int)
    (result = ess,)
end


function bat_eff_sample_size(smpls::DensitySampleVector, algorithm = GeyerAutocorLen(); use_weights = true)
    ess = bat_eff_sample_size(smpls.v, algorithm).result

    if use_weights
        W = smpls.weight
        w_correction = wgt_effective_sample_size(W) / length(eachindex(W))
        ess .*= w_correction
    end

    (result = ess,)
end


@doc doc"""
    wgt_effective_sample_size(w::AbstractVector{T})

*BAT-internal, not part of stable public API.*

Kish's approximation for weighted samples effective_sample_size estimation.
Computes the weighting factor for weigthed samples, where w is the vector of
weigths.
"""
function wgt_effective_sample_size(w::AbstractVector{<:Real})
    return sum(w)^2/sum(w.^2)
end
