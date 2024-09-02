# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type MHProposalDistTuning

Abstract type for Metropolis-Hastings tuning strategies for
proposal distributions.
"""
abstract type MHProposalDistTuning <: MCMCTuningAlgorithm end
export MHProposalDistTuning


"""
    struct MetropolisHastings <: MCMCAlgorithm

Metropolis-Hastings MCMC sampling algorithm.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MetropolisHastings{
    Q<:ContinuousDistribution,
    WS<:AbstractMCMCWeightingScheme,
    TN<:MHProposalDistTuning,
} <: MCMCAlgorithm
    proposal::Q = TDist(1.0)
    weighting::WS = RepetitionWeighting()
    tuning::TN = AdaptiveMHTuning()
end

export MetropolisHastings


bat_default(::Type{MCMCSampling}, ::Val{:trafo}, mcalg::MetropolisHastings) = PriorToGaussian()

bat_default(::Type{MCMCSampling}, ::Val{:nsteps}, mcalg::MetropolisHastings, trafo::AbstractTransformTarget, nchains::Integer) = 10^5

bat_default(::Type{MCMCSampling}, ::Val{:init}, mcalg::MetropolisHastings, trafo::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCChainPoolInit(nsteps_init = max(div(nsteps, 100), 250))

bat_default(::Type{MCMCSampling}, ::Val{:burnin}, mcalg::MetropolisHastings, trafo::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCMultiCycleBurnin(nsteps_per_cycle = max(div(nsteps, 10), 2500))


get_mcmc_tuning(algorithm::MetropolisHastings) = algorithm.tuning



mutable struct MHState{
    AL<:MetropolisHastings,
    D<:BATMeasure,
    PR<:RNGPartition,
    Q<:Distribution{Multivariate,Continuous},
    SV<:DensitySampleVector,
    CTX<:BATContext
} <: MCMCState
    algorithm::AL
    target::D
    rngpart_cycle::PR
    info::MCMCStateInfo
    proposaldist::Q
    samples::SV
    nsamples::Int64
    stepno::Int64
    context::CTX
end


function MHState(
    algorithm::MCMCAlgorithm,
    target::BATMeasure,
    info::MCMCStateInfo,
    x_init::AbstractVector{P},
    context::BATContext
) where {P<:Real}
    rng = get_rng(context)
    stepno::Int64 = 0

    npar = getdof(target)

    params_vec = Vector{P}(undef, npar)
    params_vec .= x_init

    proposaldist = mv_proposaldist(P, algorithm.proposal, npar)

    log_posterior_value = logdensityof(target, params_vec)

    T = typeof(log_posterior_value)
    W = sample_weight_type(typeof(algorithm.weighting))

    sample_info = MCMCSampleID(info.id, info.cycle, 1, CURRENT_SAMPLE)
    current_sample = DensitySample(params_vec, log_posterior_value, one(W), sample_info, nothing)
    samples = DensitySampleVector{Vector{P},T,W,MCMCSampleID,Nothing}(undef, 0, npar)
    push!(samples, current_sample)

    nsamples::Int64 = 0

    rngpart_cycle = RNGPartition(rng, 0:(typemax(Int16) - 2))

    chain = MHState(
        algorithm,
        target,
        rngpart_cycle,
        info,
        proposaldist,
        samples,
        nsamples,
        stepno,
        context
    )

    reset_rng_counters!(chain)

    chain
end


function MCMCState(
    algorithm::MetropolisHastings,
    target::BATMeasure,
    chainid::Integer,
    startpos::AbstractVector{<:Real},
    context::BATContext
)
    cycle = 0
    tuned = false
    converged = false
    info = MCMCStateInfo(chainid, cycle, tuned, converged)
    MHState(algorithm, target, info, startpos, context)
end


@inline _current_sample_idx(state::MCMCState) = firstindex(state.samples)
@inline _proposed_sample_idx(state::MCMCState) = lastindex(state.samples)


current_sample(state::MCMCState, proposal::MHProposal) = state.samples[_current_sample_idx(state)]


function reset_rng_counters!(chain::MHState)
    rng = get_rng(get_context(chain))
    set_rng!(rng, chain.rngpart_cycle, chain.info.cycle)
    rngpart_step = RNGPartition(rng, 0:(typemax(Int32) - 2))
    set_rng!(rng, rngpart_step, chain.stepno)
    nothing
end


function samples_available(chain::MHState)
    i = _current_sample_idx(chain::MHState)
    chain.samples.info.sampletype[i] == ACCEPTED_SAMPLE
end


function get_samples!(appendable, chain::MHState, nonzero_weights::Bool)::typeof(appendable)
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


function next_cycle!(chain::MHState)
    _cleanup_samples(chain)

    chain.info = MCMCStateInfo(chain.info, cycle = chain.info.cycle + 1)
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


function _cleanup_samples(chain::MHState)
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


function mcmc_step!(chain::MHState)
    rng = get_rng(get_context(chain))

    _cleanup_samples(chain)

    samples = chain.samples
    algorithm = getalgorithm(chain)

    chain.stepno += 1
    reset_rng_counters!(chain)

    rng = get_rng(get_context(chain))
    target = mcmc_target(chain)

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

    current_log_posterior = samples.logd[current]
    T = typeof(current_log_posterior)

    # Evaluate prior and likelihood with proposed variate:
    proposed_log_posterior = checked_logdensityof(target, proposed_params)

    samples.logd[proposed] = proposed_log_posterior

    p_accept = if proposed_log_posterior > -Inf
        # log of ratio of forward/reverse transition probability
        log_tpr = if issymmetric_around_origin(proposaldist)
            T(0)
        else
            log_tp_fwd = proposaldist_logpdf(proposaldist, proposed_params, current_params)
            log_tp_rev = proposaldist_logpdf(proposaldist, current_params, proposed_params)
            T(log_tp_fwd - log_tp_rev)
        end

        p_accept_unclamped = exp(proposed_log_posterior - current_log_posterior - log_tpr)
        T(clamp(p_accept_unclamped, 0, 1))
    else
        zero(T)
    end

    @assert p_accept >= 0
    accepted = rand(rng, float(typeof(p_accept))) < p_accept

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
) where Q
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
) where Q
    T = typeof(p_accept)
    if p_accept ≈ 1
        (zero(T), one(T))
    elseif p_accept ≈ 0
        (one(T), zero(T))
    else
        (T(1 - p_accept), p_accept)
    end
end


eff_acceptance_ratio(chain::MHState) = nsamples(chain) / nsteps(chain)
