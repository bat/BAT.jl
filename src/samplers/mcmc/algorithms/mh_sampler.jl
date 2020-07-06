# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    MetropolisHastings

Metropolis-Hastings MCMC sampling algorithm.

Constructors:

    MetropolisHastings()
    MetropolisHastings(weighting::AbstractWeightingScheme)
"""
struct MetropolisHastings{
    Q<:ProposalDistSpec,
    W<:Real,
    WS<:AbstractWeightingScheme{W}
} <: MCMCAlgorithm
    proposalspec::Q
    weighting::WS

    MetropolisHastings(proposalspec::Q,weighting::WS) where {
        Q<:ProposalDistSpec, W<:Real, WS<:AbstractWeightingScheme{W}} =
        new{Q,W,WS}(proposalspec, weighting)
end

export MetropolisHastings


MetropolisHastings(proposalspec::ProposalDistSpec = MvTDistProposal()) =
    MetropolisHastings(proposalspec, RepetitionWeighting())

MetropolisHastings(weighting::AbstractWeightingScheme) =
    MetropolisHastings(MvTDistProposal(), weighting)


mcmc_compatible(::MetropolisHastings, ::AbstractProposalDist, ::NoVarBounds) = true

mcmc_compatible(::MetropolisHastings, proposaldist::AbstractProposalDist, bounds::HyperRectBounds) =
    issymmetric(proposaldist) || all(x -> x == hard_bounds, bounds.bt)

mcmc_compatible(::MetropolisHastings, proposaldist::AbstractProposalDist, bounds::HierarchicalDensityBounds) =
    issymmetric(proposaldist)


_sample_weight_type(::Type{MetropolisHastings{Q,W,WS}}) where {Q,W,WS} = W



mutable struct MHIterator{
    SP<:MCMCSpec,
    R<:AbstractRNG,
    PR<:RNGPartition,
    Q<:AbstractProposalDist,
    SV<:DensitySampleVector
} <: MCMCIterator
    spec::SP
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
    spec::MCMCSpec,
    info::MCMCIteratorInfo,
    x_init::AbstractVector{P},
) where {P<:Real}
    stepno::Int64 = 0

    postr = spec.posterior
    npar = totalndof(postr)
    alg = spec.algorithm

    params_vec = Vector{P}(undef, npar)
    if isempty(x_init)
        mcmc_startval!(params_vec, rng, postr, alg)
    else
        params_vec .= x_init
    end
    !(params_vec in var_bounds(postr)) && throw(ArgumentError("Parameter(s) out of bounds"))

    proposaldist = alg.proposalspec(P, npar)

    # ToDo: Make numeric type configurable:

    log_posterior_value = logvalof(postr, params_vec, strict = true)

    T = typeof(log_posterior_value)
    W = _sample_weight_type(typeof(alg))

    sample_info = MCMCSampleID(info.id, info.cycle, 1, CURRENT_SAMPLE)
    current_sample = DensitySample(params_vec, log_posterior_value, one(W), sample_info, nothing)
    samples = DensitySampleVector{Vector{P},T,W,MCMCSampleID,Nothing}(undef, 0, npar)
    push!(samples, current_sample)

    nsamples::Int64 = 0

    rngpart_cycle = RNGPartition(rng, 0:(typemax(Int16) - 2))

    chain = MHIterator(
        spec,
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


function reset_rng_counters!(chain::MHIterator)
    set_rng!(chain.rng, chain.rngpart_cycle, chain.info.cycle)
    rngpart_step = RNGPartition(chain.rng, 0:(typemax(Int32) - 2))
    set_rng!(chain.rng, rngpart_step, chain.stepno)
    nothing
end


function (spec::MCMCSpec{<:MetropolisHastings})(
    rng::AbstractRNG,
    chainid::Integer
)
    P = float(eltype(var_bounds(spec.posterior)))

    cycle = 0
    tuned = false
    converged = false
    info = MCMCIteratorInfo(chainid, cycle, tuned, converged)

    MHIterator(rng, spec, info, Vector{P}())
end


@inline _current_sample_idx(chain::MHIterator) = firstindex(chain.samples)
@inline _proposed_sample_idx(chain::MHIterator) = lastindex(chain.samples)

function _available_samples_idxs(chain::MHIterator)
    sampletype = chain.samples.info.sampletype
    from = firstindex(chain.samples)

    to = if samples_available(chain)
        lastidx = lastindex(chain.samples)
        @assert sampletype[from] == ACCEPTED_SAMPLE
        @assert sampletype[lastidx] == CURRENT_SAMPLE
        lastidx - 1
    else
        from - 1
    end

    r = from:to
    @assert all(x -> x > INVALID_SAMPLE, view(sampletype, r))
    r
end


mcmc_spec(chain::MHIterator) = chain.spec

getrng(chain::MHIterator) = chain.rng

mcmc_info(chain::MHIterator) = chain.info

nsteps(chain::MHIterator) = chain.stepno

nsamples(chain::MHIterator) = chain.nsamples

current_sample(chain::MHIterator) = chain.samples[_current_sample_idx(chain)]

sample_type(chain::MHIterator) = eltype(chain.samples)


function samples_available(chain::MHIterator)
    i = _current_sample_idx(chain::MHIterator)
    chain.samples.info.sampletype[i] == ACCEPTED_SAMPLE
end


function get_samples!(appendable, chain::MHIterator, nonzero_weights::Bool)::typeof(appendable)
    if samples_available(chain)
        idxs = _available_samples_idxs(chain)
        samples = chain.samples

        # if nonzero_weights
            for i in idxs
                if !nonzero_weights || samples.weight[i] > 0
                    push!(appendable, samples[i])
                end
            end
        # else
        #     append!(appendable, view(samples, idxs))
        # end
    end
    appendable
end


function next_cycle!(chain::MHIterator)
    chain.info = MCMCIteratorInfo(chain.info, cycle = chain.info.cycle + 1)
    chain.nsamples = 0
    chain.stepno = 0

    reset_rng_counters!(chain)

    resize!(chain.samples, 1)

    i = _current_sample_idx(chain)
    @assert chain.samples.info[i].sampletype == CURRENT_SAMPLE

    chain.samples.weight[i] = 1
    chain.samples.info[i] = MCMCSampleID(chain.info.id, chain.info.cycle, chain.stepno, CURRENT_SAMPLE)

    chain
end


function mcmc_step!(
    callback::AbstractMCMCCallback,
    chain::MHIterator
)
    alg = algorithm(chain)

    if !mcmc_compatible(alg, chain.proposaldist, var_bounds(getposterior(chain)))
        error("Implementation of algorithm $alg does not support current parameter bounds with current proposal distribution")
    end

    chain.stepno += 1
    reset_rng_counters!(chain)

    rng = getrng(chain)
    pstr = getposterior(chain)

    proposaldist = chain.proposaldist
    samples = chain.samples

    # Grow samples vector by one:
    resize!(samples, size(samples, 1) + 1)
    samples.info[lastindex(samples)] = MCMCSampleID(chain.info.id, chain.info.cycle, chain.stepno, PROPOSED_SAMPLE)

    current = _current_sample_idx(chain)
    proposed = _proposed_sample_idx(chain)
    @assert current != proposed

    accepted = let
        current_params = samples.v[current]
        proposed_params = samples.v[proposed]

        # Propose new variate:
        samples.weight[proposed] = 0
        proposal_rand!(rng, proposaldist, proposed_params, current_params)
        renormalize_variate!(proposed_params, pstr, proposed_params)

        current_log_posterior = samples.logd[current]
        T = typeof(current_log_posterior)

        # Evaluate prior and likelihood with proposed variate:
        proposed_log_posterior = logvalof(pstr, proposed_params, strict = false)

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

        delta_w_current, w_proposed = _mh_weights(alg, p_accept, accepted)
        samples.weight[current] += delta_w_current
        samples.weight[proposed] = w_proposed

        callback(1, chain)

        if accepted
            current_params .= proposed_params
            samples.logd[current] = samples.logd[proposed]
            samples.weight[current] = samples.weight[proposed]
            samples.info[current] = samples.info[proposed]
        end

        accepted
    end

    if accepted
        resize!(samples, 1)
    end

    chain
end


function _mh_weights(
    algorithm::MetropolisHastings{Q,W,<:RepetitionWeighting},
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
    algorithm::MetropolisHastings{Q,W,<:ARPWeighting},
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
