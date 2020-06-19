# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function tau_int_from_atc end


function bat_integrated_autocorr_len_impl(v::AbstractVector{<:Real}, algorithm::AutocorLenAlgorithm)
    atc = fft_autocor(v)
    tau_int_est = tau_int_from_atc(atc, algorithm)
    (result = tau_int_est,)
end

function bat_integrated_autocorr_len_impl(v::AbstractVectorOfSimilarVectors{<:Real}, algorithm::AutocorLenAlgorithm)
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



function bat_eff_sample_size_impl(v::Union{AbstractVector{<:Real},AbstractVectorOfSimilarVectors{<:Real}}, algorithm::AutocorLenAlgorithm)
    tau_int = bat_integrated_autocorr_len(v, algorithm).result
    n = length(eachindex(v))
    ess = min.(n, n./ tau_int)
    (result = ess,)
end


function bat_eff_sample_size_impl(smpls::DensitySampleVector, algorithm::AutocorLenAlgorithm)
    @argcheck smpls.v isa AbstractVector{<:AbstractVector{<:Real}}

    n = length(eachindex(smpls))

    W = smpls.weight
    w0 = first(W)

    ess = if all(w -> w ≈ w0, W)
        bat_eff_sample_size_impl(smpls.v, algorithm).result
    else
        # If weights not uniform, resample to get unweighted samples. Kish's
        # approximation of ESS for weighted samples is often not good enough.

        # Empirical resampling factor:
        resampling_factor = min(mean(W .^ 2) / mean(W)^2, 10)

        n_resample = round(Int, n * resampling_factor)

        # RNG seed for resampling should be the same for the same samples:
        rng_seed = trunc(UInt64, mean(W) * n)
        rng = Philox4x((0x0, rng_seed))::Philox4x{UInt64,10}

        unweighted_smpls = bat_sample(rng, smpls, n_resample, OrderedResampling()).result
        bat_eff_sample_size_impl(unweighted_smpls.v, algorithm).result
    end
 
    (result = ess,)
end
