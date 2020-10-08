# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    MHProposalDistTuning

Abstract super-type for Metropolis-Hastings tuning strategies for
proposal distributions.
"""
abstract type MHProposalDistTuning <: MCMCTuningAlgorithm end
export MHProposalDistTuning


"""
    MetropolisHastings

Metropolis-Hastings MCMC sampling algorithm.

Constructors:

```julia
MetropolisHastings(
    proposal::ProposalDistSpec
    weighting::AbstractMCMCWeightingScheme
    tuning::TN = MHProposalDistTuning
)
"""
@with_kw struct MetropolisHastings{
    Q<:ProposalDistSpec,
    WS<:AbstractMCMCWeightingScheme,
    TN<:MHProposalDistTuning,
} <: MCMCAlgorithm
    proposal::Q = MvTDistProposal()
    weighting::WS = RepetitionWeighting()
    tuning::TN = AdaptiveMHTuning()
end

export MetropolisHastings


get_mcmc_tuning(algorithm::MetropolisHastings) = algorithm.tuning


mcmc_compatible(::MetropolisHastings, ::AbstractProposalDist, ::NoVarBounds) = true

mcmc_compatible(::MetropolisHastings, proposaldist::AbstractProposalDist, bounds::HyperRectBounds) =
    issymmetric(proposaldist) || all(x -> x == hard_bounds, bounds.bt)

mcmc_compatible(::MetropolisHastings, proposaldist::AbstractProposalDist, bounds::HierarchicalDensityBounds) =
    issymmetric(proposaldist)



mutable struct MHIterator{
    AL<:MetropolisHastings,
    D<:AbstractDensity,
    R<:AbstractRNG,
    PR<:RNGPartition,
    Q<:AbstractProposalDist,
    SV<:DensitySampleVector
} <: MCMCIterator
    algorithm::AL
    density::D
    rng::R
    rngpart_cycle::PR
    info::MCMCIteratorInfo
    proposaldist::Q
    samples::SV
    nsamples::Int64
    stepno::Int64
end


function MHIterator(
    rng::AbstractRNG,
    algorithm::MCMCAlgorithm,
    density::AbstractDensity,
    info::MCMCIteratorInfo,
    x_init::AbstractVector{P},
) where {P<:Real}
    stepno::Int64 = 0

    npar = totalndof(density)

    params_vec = Vector{P}(undef, npar)
    params_vec .= x_init
    !(params_vec in var_bounds(density)) && throw(ArgumentError("Parameter(s) out of bounds"))

    proposaldist = algorithm.proposal(P, npar)

    log_posterior_value = eval_logval(density, params_vec, strict = true)

    T = typeof(log_posterior_value)
    W = sample_weight_type(typeof(algorithm.weighting))

    sample_info = MCMCSampleID(info.id, info.cycle, 1, CURRENT_SAMPLE)
    current_sample = DensitySample(params_vec, log_posterior_value, one(W), sample_info, nothing)
    samples = DensitySampleVector{Vector{P},T,W,MCMCSampleID,Nothing}(undef, 0, npar)
    push!(samples, current_sample)

    nsamples::Int64 = 0

    rngpart_cycle = RNGPartition(rng, 0:(typemax(Int16) - 2))

    chain = MHIterator(
        algorithm,
        density,
        rng,
        rngpart_cycle,
        info,
        proposaldist,
        samples,
        nsamples,
        stepno
    )

    reset_rng_counters!(chain)

    chain
end


function MCMCIterator(
    rng::AbstractRNG,
    algorithm::MetropolisHastings,
    density::AbstractDensity,
    chainid::Integer,
    startpos::AbstractVector{<:Real}
)
    cycle = 0
    tuned = false
    converged = false
    info = MCMCIteratorInfo(chainid, cycle, tuned, converged)
    MHIterator(rng, algorithm, density, info, startpos)
end


@inline _current_sample_idx(chain::MHIterator) = firstindex(chain.samples)
@inline _proposed_sample_idx(chain::MHIterator) = lastindex(chain.samples)


getalgorithm(chain::MHIterator) = chain.algorithm

getdensity(chain::MHIterator) = chain.density

getrng(chain::MHIterator) = chain.rng

mcmc_info(chain::MHIterator) = chain.info

nsteps(chain::MHIterator) = chain.stepno

nsamples(chain::MHIterator) = chain.nsamples

current_sample(chain::MHIterator) = chain.samples[_current_sample_idx(chain)]

sample_type(chain::MHIterator) = eltype(chain.samples)


function reset_rng_counters!(chain::MHIterator)
    set_rng!(chain.rng, chain.rngpart_cycle, chain.info.cycle)
    rngpart_step = RNGPartition(chain.rng, 0:(typemax(Int32) - 2))
    set_rng!(chain.rng, rngpart_step, chain.stepno)
    nothing
end


function samples_available(chain::MHIterator)
    i = _current_sample_idx(chain::MHIterator)
    chain.samples.info.sampletype[i] == ACCEPTED_SAMPLE
end


function get_samples!(appendable, chain::MHIterator, nonzero_weights::Bool)::typeof(appendable)
    if samples_available(chain)
        samples = chain.samples

        for i in eachindex(samples)
            st = samples.info.sampletype[i]
            if (
                (st == ACCEPTED_SAMPLE || st == REJECTED_SAMPLE) &&
                (samples.weight[i] > 0 || !nonzero_weights)
            )
                push!(appendable, samples[i])
            end
        end
    end
    appendable
end


function next_cycle!(chain::MHIterator)
    _cleanup_samples(chain)

    chain.info = MCMCIteratorInfo(chain.info, cycle = chain.info.cycle + 1)
    chain.nsamples = 0
    chain.stepno = 0

    reset_rng_counters!(chain)

    resize!(chain.samples, 1)

    i = _proposed_sample_idx(chain)
    @assert chain.samples.info[i].sampletype == CURRENT_SAMPLE
    chain.samples.weight[i] = 1

    chain.samples.info[i] = MCMCSampleID(chain.info.id, chain.info.cycle, chain.stepno, CURRENT_SAMPLE)

    chain
end


function _cleanup_samples(chain::MHIterator)
    samples = chain.samples
    current = _current_sample_idx(chain)
    proposed = _proposed_sample_idx(chain)
    if (current != proposed) && samples.info.sampletype[proposed] == CURRENT_SAMPLE
        # Proposal was accepted in the last step
        @assert samples.info.sampletype[current] == ACCEPTED_SAMPLE
        samples.v[current] .= samples.v[proposed]
        samples.logd[current] = samples.logd[proposed]
        samples.weight[current] = samples.weight[proposed]
        samples.info[current] = samples.info[proposed]

        resize!(samples, 1)
    end
end


function mcmc_step!(chain::MHIterator)
    _cleanup_samples(chain)

    samples = chain.samples
    algorithm = getalgorithm(chain)

    if !mcmc_compatible(algorithm, chain.proposaldist, var_bounds(getdensity(chain)))
        error("Implementation of algorithm $algorithm does not support current parameter bounds with current proposal distribution")
    end

    chain.stepno += 1
    reset_rng_counters!(chain)

    rng = getrng(chain)
    density = getdensity(chain)

    proposaldist = chain.proposaldist

    # Grow samples vector by one:
    resize!(samples, size(samples, 1) + 1)
    samples.info[lastindex(samples)] = MCMCSampleID(chain.info.id, chain.info.cycle, chain.stepno, PROPOSED_SAMPLE)

    current = _current_sample_idx(chain)
    proposed = _proposed_sample_idx(chain)
    @assert current != proposed

    current_params = samples.v[current]
    proposed_params = samples.v[proposed]

    # Propose new variate:
    samples.weight[proposed] = 0
    proposal_rand!(rng, proposaldist, proposed_params, current_params)
    renormalize_variate!(proposed_params, density, proposed_params)

    current_log_posterior = samples.logd[current]
    T = typeof(current_log_posterior)

    # Evaluate prior and likelihood with proposed variate:
    proposed_log_posterior = eval_logval(density, proposed_params, strict = false)

    samples.logd[proposed] = proposed_log_posterior

    p_accept = if proposed_log_posterior > -Inf
        # log of ratio of forward/reverse transition probability
        log_tpr = if issymmetric(proposaldist)
            T(0)
        else
            log_tp_fwd = distribution_logpdf(proposaldist, proposed_params, current_params)
            log_tp_rev = distribution_logpdf(proposaldist, current_params, proposed_params)
            T(log_tp_fwd - log_tp_rev)
        end

        p_accept_unclamped = exp(proposed_log_posterior - current_log_posterior - log_tpr)
        T(clamp(p_accept_unclamped, 0, 1))
    else
        zero(T)
    end

    @assert p_accept >= 0
    accepted = rand(chain.rng, float(typeof(p_accept))) < p_accept

    if accepted
        samples.info.sampletype[current] = ACCEPTED_SAMPLE
        samples.info.sampletype[proposed] = CURRENT_SAMPLE
        chain.nsamples += 1
    else
        samples.info.sampletype[proposed] = REJECTED_SAMPLE
    end

    delta_w_current, w_proposed = _mh_weights(algorithm, p_accept, accepted)
    samples.weight[current] += delta_w_current
    samples.weight[proposed] = w_proposed

    nothing
end


function _mh_weights(
    algorithm::MetropolisHastings{Q,<:RepetitionWeighting},
    p_accept::Real,
    accepted::Bool
) where {Q,W}
    if accepted
        (0, 1)
    else
        (1, 0)
    end
end


function _mh_weights(
    algorithm::MetropolisHastings{Q,<:ARPWeighting},
    p_accept::Real,
    accepted::Bool
) where {Q,W}
    T = typeof(p_accept)
    if p_accept ≈ 1
        (zero(T), one(T))
    elseif p_accept ≈ 0
        (one(T), zero(T))
    else
        (T(1 - p_accept), p_accept)
    end
end


eff_acceptance_ratio(chain::MHIterator) = nsamples(chain) / nsteps(chain)
