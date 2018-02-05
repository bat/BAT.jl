# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type MHWeightingScheme{T<:Real} end
export MHWeightingScheme

struct MHMultiplicityWeights{T<:Real} <: MHWeightingScheme{T} end
export MHMultiplicityWeights
MHMultiplicityWeights() = MHMultiplicityWeights{Int}()

struct MHAccRejProbWeights{T<:AbstractFloat} <: MHWeightingScheme{T} end
export MHAccRejProbWeights
MHAccRejProbWeights() = MHAccRejProbWeights{Float64}()

struct MHPosteriorFractionWeights{T<:AbstractFloat} <: MHWeightingScheme{T} end
export MHPosteriorFractionWeights
MHPosteriorFractionWeights() = MHPosteriorFractionWeights{Float64}()


struct MetropolisHastings{
    Q<:ProposalDistSpec,
    W<:Real,
    WS<:MHWeightingScheme{W}
} <: MCMCAlgorithm{AcceptRejectState}
    q::Q
    weighting_scheme::WS
end

export MetropolisHastings

MetropolisHastings(
    q::Q,
    weighting_scheme::WS
) where {
    Q<:ProposalDistSpec,
    W<:Real,
    WS<:MHWeightingScheme{W}
} = MetropolisHastings{Q,W,WS}(q, weighting_scheme)

MetropolisHastings(q::ProposalDistSpec = MvTDistProposalSpec()) = MetropolisHastings(q, MHMultiplicityWeights())
MetropolisHastings(weighting_scheme::MHWeightingScheme) = MetropolisHastings(MvTDistProposalSpec(), weighting_scheme)


mcmc_compatible(::MetropolisHastings, ::AbstractProposalDist, ::NoParamBounds) = true

mcmc_compatible(::MetropolisHastings, pdist::AbstractProposalDist, bounds::HyperRectBounds) =
    issymmetric(pdist) || all(x -> x == hard_bounds, bounds.bt)

sample_weight_type(::Type{MetropolisHastings{Q,W,WS}}) where {Q,W,WS} = W


function AcceptRejectState(
    algorithm::MetropolisHastings,
    target::AbstractDensity,
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
    apply_bounds!(proposed_params, param_bounds(target), false)

    p_accept = if proposed_params in param_bounds(target)
        # Evaluate target density at new parameters:
        proposed_log_value = T(density_logval(target, proposed_params, exec_context))

        # log of ratio of forward/reverse transition probability
        log_tpr = if issymmetric(pdist)
            T(0)
        else
            log_tp_fwd = distribution_logpdf(pdist, proposed_params, current_params)
            log_tp_rev = distribution_logpdf(pdist, current_params, proposed_params)
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


function mh_acc_rej!(
    callback::AbstractMCMCCallback,
    chain::MCMCIterator{<:MetropolisHastings{Q,W,WS}},
    p_accept::Real
) where {Q, W, WS <: MHMultiplicityWeights}
    @assert p_accept >= 0
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


function mh_acc_rej!(
    callback::AbstractMCMCCallback,
    chain::MCMCIterator{<:MetropolisHastings{Q,W,WS}},
    p_accept::Real
) where {Q, W, WS <: MHAccRejProbWeights}
    @assert p_accept >= 0
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


function mh_acc_rej!(
    callback::AbstractMCMCCallback,
    chain::MCMCIterator{<:MetropolisHastings{Q,W,WS}},
    p_accept::Real
) where {Q, W, WS <: MHPosteriorFractionWeights}
    @assert p_accept >= 0

    state = chain.state

    r = rand(chain.rng, float(typeof(p_accept)))

    if p_accept ≈ 0
        state.current_sample.weight += 1
        state.proposed_sample.weight = 0
        callback(2, chain)
    else
        # Renormalize posterior values:
        logval_1 = state.current_sample.log_value
        logval_2 = state.proposed_sample.log_value
        max_lv = max(logval_1, logval_2)
        v_1 = exp(logval_1 - max_lv)
        v_2 = exp(logval_2 - max_lv)
        v_sum = v_1 + v_2

        state.current_sample.weight += v_1 / v_sum
        state.proposed_sample.weight = v_2 / v_sum
        if r < p_accept
            state.proposal_accepted = true
            state.nsamples += 1
        end
        callback(1, chain)
    end
    chain
end
