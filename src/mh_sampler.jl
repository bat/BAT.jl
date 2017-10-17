# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct MetropolisHastings{
    Q<:ProposalDistSpec
} <: MCMCAlgorithm{AcceptRejectState}
    q::Q
end

export MetropolisHastings

MetropolisHastings() = MetropolisHastings(MvTDistProposalSpec())


mcmc_compatible(::MetropolisHastings, ::AbstractProposalDist, ::UnboundedParams) = true

mcmc_compatible(::MetropolisHastings, pdist::AbstractProposalDist, bounds::HyperRectBounds) =
    issymmetric(pdist) || all(x -> x == hard_bounds, bounds.bt)


function AcceptRejectState(
    algorithm::MetropolisHastings,
    target::AbstractTargetSubject,
    current_sample::DensitySample{P,T,W}
) where {P,T,W}
    AcceptRejectState(
        algorithm.q(P, nparams(current_sample)),
        current_sample
    )
end



function mcmc_propose_accept_reject!(
    callback::Function,
    chain::MCMCIterator{<:MetropolisHastings},
    exec_context::ExecContext
)
    state = chain.state
    rng = chain.rng
    target = chain.target
    pdist = state.pdist

    current_sample = state.current_sample
    proposed_sample = state.proposed_sample

    current_params = current_sample.params
    proposed_params = proposed_sample.params

    current_log_value = current_sample.log_value
    T = typeof(current_log_value)

    # Propose new parameters:
    proposal_rand!(rng, pdist, proposed_params, current_params)
    apply_bounds!(proposed_params, target.bounds, false)

    accepted = if proposed_params in target.bounds
        # Evaluate target density at new parameters:
        proposed_log_value = T(target_logval(target.tdensity, proposed_params, exec_context))

        # log of ratio of forward/reverse transition probability
        log_tpr = if issymmetric(pdist)
            T(0)
        else
            log_tp_fwd = proposal_logpdf(pdist, proposed_params, current_params)
            log_tp_rev = proposal_logpdf(pdist, current_params, proposed_params)
            T(log_tp_fwd - log_tp_rev)
        end

        current_log_value = current_sample.log_value
        proposed_sample.log_value = proposed_log_value

        # Metropolis-Hastings accept/reject:
        if rand(rng) < exp(proposed_log_value - current_log_value - log_tpr)
            current_sample.weight = state.current_nreject + 1
            true
        else
            false
        end
    else
        # Reject:
        proposed_sample.log_value = -Inf
        false
    end

    if accepted
        state.proposal_accepted = true
        state.nsamples += 1
        callback(1, chain)
    else
        state.current_nreject += 1
        callback(2, chain)
    end

    nothing
end
