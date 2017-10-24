# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct MetropolisHastings{
    W<:Real,
    Q<:ProposalDistSpec
} <: MCMCAlgorithm{AcceptRejectState}
    q::Q

    MetropolisHastings{W}(q::Q = MvTDistProposalSpec()) where {W<:Real, Q<:ProposalDistSpec} = new{W,Q}(q)
end

export MetropolisHastings

MetropolisHastings(q::ProposalDistSpec = MvTDistProposalSpec()) = MetropolisHastings{Int}()


mcmc_compatible(::MetropolisHastings, ::AbstractProposalDist, ::UnboundedParams) = true

mcmc_compatible(::MetropolisHastings, pdist::AbstractProposalDist, bounds::HyperRectBounds) =
    issymmetric(pdist) || all(x -> x == hard_bounds, bounds.bt)

sample_weight_type(::Type{MetropolisHastings{W,Q}}) where {Q,W} = W


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
    callback::AbstractMCMCCallback,
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
    proposed_sample.weight = 0
    proposal_rand!(rng, pdist, proposed_params, current_params)
    apply_bounds!(proposed_params, target.bounds, false)

    p_accept = if proposed_params in target.bounds
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

        p_accept = if proposed_log_value > -Inf
            clamp(T(exp(proposed_log_value - current_log_value - log_tpr)), zero(T), one(T))
        else
            0
        end
    else
        p_accept = zero(T)
        proposed_sample.log_value = -Inf

        zero(T)
    end

    mh_acc_rej!(callback, chain, p_accept)

    chain
end


function mh_acc_rej!(callback::AbstractMCMCCallback, chain::MCMCIterator{<:MetropolisHastings{<:Integer}}, p_accept::Real)
    state = chain.state
    state.current_sample.weight += 1
    if rand(chain.rng, float(typeof(p_accept))) < p_accept
        state.proposal_accepted = true
        state.nsamples += 1
        callback(1, chain)
    else
        state.current_nreject += 1
        callback(2, chain)
    end
    chain
end


function mh_acc_rej!(callback::AbstractMCMCCallback, chain::MCMCIterator{<:MetropolisHastings{<:AbstractFloat}}, p_accept::Real)
    if p_accept ≈ 1
        p_accept = one(p_accept)
    elseif p_accept ≈ 0
        p_accept = zero(p_accept)
    end

    state = chain.state
    state.current_sample.weight += (1 - p_accept)
    state.proposed_sample.weight = p_accept
    if rand(chain.rng, float(typeof(p_accept))) < p_accept
        state.proposal_accepted = true
        state.nsamples += 1
        callback(1, chain)
    else
        state.current_nreject += 1
        if p_accept ≈ 0
            callback(2, chain)
        else
            callback(1, chain)
        end
    end
    chain
end
