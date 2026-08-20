# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function tau_int_from_atc end


function bat_integrated_autocorr_len_impl(v::AbstractVector{<:Real}, algorithm::AutocorLenAlgorithm, ::BATContext)
    atc = fft_autocor(v)
    tau_int_est = tau_int_from_atc(atc, algorithm)
    (result = tau_int_est,)
end

function bat_integrated_autocorr_len_impl(v::AbstractVectorOfSimilarVectors{<:Real}, algorithm::AutocorLenAlgorithm, ::BATContext)
    atc = fft_autocor(v)
    flat_atc = flatview(atc)
    tau_int_est = map(axes(flat_atc, 1)) do i
        tau_int_from_atc(view(flat_atc, i, :), algorithm)
    end
    (result = tau_int_est,)
end



"""
    struct GeyerAutocorLen <: AutocorLenAlgorithm
    
Integrated autocorrelation length estimation based on Geyer’s initial monotone sequence criterion

See [C. J. Geyer, "Practical Markov Chain Monte Carlo" (1992)](https://projecteuclid.org/download/pdf_1/euclid.ss/1177011137)
and [C. J. Geyer, "Introduction to Markov Chain Monte Carlo" (2011)](https://www.semanticscholar.org/paper/1-Introduction-to-Markov-Chain-Monte-Carlo-Geyer/21a92825dcec23c743e77451ff5b5ee6b1091651).

Constructors:

* ```$(FUNCTIONNAME)()```

The same algorithm is used by
[STAN (v2.21)](https://mc-stan.org/docs/2_21/reference-manual/effective-sample-size-section.html#estimation-of-effective-sample-size)
and [MCMCChains.jl (v3.0, function `ess_rhat`)](https://github.com/TuringLang/MCMCChains.jl/blob/v4.0.0/src/ess.jl#L288).
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
    struct SokalAutocorLen <: AutocorLenAlgorithm
    
Integrated autocorrelation length estimation based on the automated windowing
procedure descibed in
[A. D. Sokal, "Monte Carlo Methods in Statistical Mechanics" (1996)](https://www.semanticscholar.org/paper/Monte-Carlo-Methods-in-Statistical-Mechanics%3A-and-Sokal/0bfe9e3db30605fe2d4d26e1a288a5e2997e7225)

Same procedure is used by the emcee Python package (v3.0).

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct SokalAutocorLen <: AutocorLenAlgorithm
    "Step size for window search"
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



"""
    struct EffSampleSizeFromAC <: EffSampleSizeAlgorithm

Effective sample size estimation based on the integrated autocorrelation
length of the samples - a property of the ordered sampling process.

For uniformly weighted samples the stored order is taken as the process
order. Samples carrying MCMC sample ids are decomposed into their exact
per-walker ordered sequences (repetition weights are expanded exactly),
whose independent ESS contributions add. For nonuniformly weighted
samples without process provenance, a resample-then-autocorrelate
heuristic is used - [`KishESS`](@ref) is the provenance-free
alternative (and the default in that case).

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct EffSampleSizeFromAC{AC<:AutocorLenAlgorithm} <: EffSampleSizeAlgorithm
    acalg::AC = GeyerAutocorLen()
end

export EffSampleSizeFromAC



function bat_eff_sample_size_impl(v::Union{AbstractVector{<:Real},AbstractVectorOfSimilarVectors{<:Real}}, algorithm::EffSampleSizeFromAC, context::BATContext)
    tau_int = bat_integrated_autocorr_len_impl(v, algorithm.acalg, context).result
    n = length(eachindex(v))
    ess = min.(n, n./ tau_int)
    (result = ess,)
end


function bat_eff_sample_size_impl(smpls::DensitySampleVector, algorithm::EffSampleSizeFromAC, context::BATContext)
    vs = varshape(smpls)
    unshaped_smpls = unshaped.(smpls)
    n = length(eachindex(unshaped_smpls))
    n > 0 || throw(ArgumentError("Can't compute the effective sample size of an empty sample vector"))

    W = unshaped_smpls.weight
    all(w -> w >= 0, W) && sum(W) > 0 || throw(ArgumentError("Effective sample size requires non-negative sample weights with a positive sum"))
    w0 = first(W)

    # Autocorrelation ESS is a property of an ordered sampling process.
    # For uniform weights, the stored order is the process order; MCMC
    # sample-id provenance reconstructs the per-walker ordered sequences
    # exactly, even after merging; nonuniformly weighted samples without
    # process provenance only support a resampling heuristic (KishESS is
    # the provenance-free alternative, see the algorithm defaults):
    unshaped_ess = if all(w -> w ≈ w0, W)
        bat_eff_sample_size_impl(unshaped_smpls.v, algorithm, context).result
    elseif _has_process_provenance(unshaped_smpls)
        _mcmc_process_ess(unshaped_smpls, algorithm, context)
    else
        _resample_ac_ess(unshaped_smpls, algorithm, context)
    end

    result_vs = replace_const_shapes(s::ConstValueShape -> ConstValueShape(Fill(n, size(s.value)...)), vs)
    ess = result_vs(unshaped_ess)

    (result = ess,)
end


# Per-sample MCMC ids identify the ordered sampling process (unique ids
# only - multinomially resampled samples must not masquerade as ordered
# chains, order-preserving systematic resampling keeps its ids):
function _has_process_provenance(unshaped_smpls::DensitySampleVector)
    info = unshaped_smpls.info
    eltype(info) <: MCMCSampleID || return false
    return allunique((id.chainid, id.walkerid, id.chaincycle, id.stepno) for id in info)
end

# Exact process ESS from MCMC sample-id provenance: reconstruct each
# walker's ordered sequence (chains and walkers may be merged and
# permuted), compute its autocorrelation ESS exactly for repetition
# weights, and sum - independent chains and walkers contribute
# additively:
function _mcmc_process_ess(unshaped_smpls::DensitySampleVector, algorithm::EffSampleSizeFromAC, context::BATContext)
    info = unshaped_smpls.info
    keys = [(id.chainid, id.walkerid) for id in info]
    ess_sum = nothing
    for k in unique(keys)
        idxs = findall(==(k), keys)
        ord = sortperm(view(info, idxs), by = id -> (id.chaincycle, id.stepno))
        walker_smpls = unshaped_smpls[idxs[ord]]
        ess_w = _walker_ordered_ess(walker_smpls, algorithm, context)
        ess_sum = isnothing(ess_sum) ? ess_w : ess_sum .+ ess_w
    end
    return ess_sum
end

# ESS of one ordered walker sequence. Within MCMC provenance, integer
# weights are repetition counts (the only integer-weight MCMC weighting
# scheme), non-integer weights (e.g. from ARP weighting) use the
# resampling heuristic on the ordered sequence:
function _walker_ordered_ess(walker_smpls::DensitySampleVector, algorithm::EffSampleSizeFromAC, context::BATContext)
    W = walker_smpls.weight
    if eltype(W) <: Integer
        return _repetition_exact_ess(walker_smpls, algorithm, context)
    elseif all(w -> w ≈ first(W), W)
        return bat_eff_sample_size_impl(walker_smpls.v, algorithm, context).result
    else
        return _resample_ac_ess(walker_smpls, algorithm, context)
    end
end

# Heuristic ESS for nonuniformly weighted ordered samples: deterministic
# order-preserving resampling to unit weights, then autocorrelation ESS
# on the resampled sequence. This imputes a serial structure that
# nonuniform weights of unknown provenance cannot guarantee - it is a
# heuristic, not an exact process ESS:
function _resample_ac_ess(unshaped_smpls::DensitySampleVector, algorithm::EffSampleSizeFromAC, context::BATContext)
    # Canonical relative weights make everything below invariant under a
    # global rescaling and safe from overflow at extreme weight scales:
    u = _canonical_rel_weights(unshaped_smpls.weight)
    n = length(eachindex(u))

    # Empirical resampling factor:
    resampling_factor = min(mean(u .^ 2) / mean(u)^2, 10)
    n_resample = round(Int, n * resampling_factor)

    # The resampling RNG seed must be the same for the same samples,
    # and invariant under a global rescaling of the weights (which
    # leaves the represented weighted measure unchanged):
    rng_seed = hash(u)
    resample_context = BATContext(rng = Philox4x((0x0, rng_seed))::Philox4x{UInt64,10})

    unweighted_smpls = samplesof(evalmeasure(unshaped_smpls, SystematicResampling(nsamples = n_resample), resample_context))
    resampled_ess = bat_eff_sample_size_impl(unweighted_smpls.v, algorithm, context).result
    return min.(n, resampled_ess)
end


# Exact autocorrelation ESS for samples whose weights are known (by the
# caller) to be Markov-chain repetition counts: run-length decoding
# reconstructs the exact ordered chain (its size is the number of chain
# steps, so this is affordable up to a size guard) and the standard
# autocorrelation machinery runs on it. Only the caller can assert the
# repetition semantics - generic sample vectors deliberately erase weight
# provenance:
function _repetition_exact_ess(smpls::DensitySampleVector, algorithm::EffSampleSizeFromAC, context::BATContext)
    unshaped_smpls = unshaped.(smpls)
    W = unshaped_smpls.weight
    @argcheck all(w -> w >= 0 && isinteger(w), W)
    # Float accumulation can't wrap around like an integer sum would for
    # huge repetition counts - the result is only compared against the
    # decoding size guard, which needs no exactness:
    N_expanded = sum(float, W, init = 0.0)
    N_expanded > 0 || throw(ArgumentError("Can't compute the effective sample size of an empty chain"))
    n_dof = length(first(unshaped_smpls.v))

    if all(isone, W)
        return bat_eff_sample_size_impl(unshaped_smpls.v, algorithm, context).result
    elseif N_expanded * n_dof <= 5 * 10^7
        idxs = inverse_rle(eachindex(W), Int.(W))
        expanded_v = nestedview(flatview(unshaped_smpls.v)[:, idxs])
        return bat_eff_sample_size_impl(expanded_v, algorithm, context).result
    else
        # Chain too large to decode, fall back to the resampling
        # heuristic on the ordered sequence:
        return _resample_ac_ess(unshaped_smpls, algorithm, context)
    end
end



"""
    struct KishESS <: EffSampleSizeAlgorithm
    
Kish's effective sample size estimator, uses only the sample weights.

See L. Kish, "Survey Sampling", John Wiley & Sons (1965), and
[effective sample size of weighted
samples](https://en.wikipedia.org/wiki/Effective_sample_size#Weighted_samples).

Constructors:

* ```$(FUNCTIONNAME)()```

"""
struct KishESS <: EffSampleSizeAlgorithm
end

export KishESS


function bat_eff_sample_size_impl(smpls::DensitySampleVector, algorithm::KishESS, context::BATContext)
    isempty(smpls) && throw(ArgumentError("Can't compute the effective sample size of an empty sample vector"))
    # Canonical relative weights keep the ratio finite for integer weights
    # near typemax and for extreme floating-point weight scales:
    u = _canonical_rel_weights(smpls.weight)
    ess = sum(u)^2 / sum(x -> x^2, u)
    (result = ess,)
end
