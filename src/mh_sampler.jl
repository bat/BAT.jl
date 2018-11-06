# This file is a part of BAT.jl, licensed under the MIT License (MIT).

mutable struct MHState{
    Q<:AbstractProposalDist,
    S<:DensitySample
} <: AbstractMCMCState
    pdist::Q
    current_sample::S
    proposed_sample::S
    proposal_accepted::Bool
    current_nreject::Int64
    nsamples::Int64
    nsteps::Int64
end

function MHState(
    pdist::AbstractProposalDist,
    current_sample::DensitySample
)
    proposed_sample = DensitySample(
        similar(current_sample.params),
        convert(typeof(current_sample.log_value), NaN),
        zero(current_sample.weight)
    )

    MHState(
        pdist,
        current_sample,
        proposed_sample,
        false,
        zero(Int64),
        zero(Int64),
        one(Int64)
    )
end


nparams(state::MHState) = nparams(state.pdist)

nsteps(state::MHState) = state.nsteps

nsamples(state::MHState) = state.nsamples

eff_acceptance_ratio(state::MHState) = nsamples(state) / nsteps(state)


function next_cycle!(state::MHState)
    state.current_sample.weight = one(state.current_sample.weight)
    state.nsamples = zero(Int64)
    state.nsteps = zero(Int64)
    state
end


density_sample_type(state::MHState{Q,S}) where {Q,S} = S


_accrej_current_sample(state::MHState) =
    ifelse(state.proposal_accepted, state.current_sample, state.proposed_sample)


function nsamples_available(chain::MCMCIterator{<:MCMCAlgorithm{MHState}}, nonzero_weights::Bool = false)
    if nonzero_weights
        (_accrej_current_sample(chain.state).weight > 0) ? 1 : 0
    else
        1
    end
end


function get_samples!(appendable, chain::MCMCIterator{<:MCMCAlgorithm{MHState}}, nonzero_weights::Bool)::typeof(appendable)
    sample = _accrej_current_sample(chain.state)
    if (!nonzero_weights || sample.weight > 0)
        push!(appendable, sample)
    end
    appendable
end


function get_sample_ids!(appendable, chain::MCMCIterator{<:MCMCAlgorithm{MHState}}, nonzero_weights::Bool)::typeof(appendable)
    state = chain.state
    sample = _accrej_current_sample(state)
    if (!nonzero_weights || sample.weight > 0)
        sampletype = state.proposal_accepted ? 2 : 1
        sampleid = MCMCSampleID(chain.id, chain.cycle, state.nsteps, sampletype)
        push!(appendable, sampleid)
    end
    appendable
end


function MCMCIterator(
    algorithm::MCMCAlgorithm{MHState},
    likelihood::AbstractDensity,
    prior::AbstractDensity,
    id::Int64,
    rng::AbstractRNG,
    initial_params::AbstractVector{P} = Vector{P}(),
    exec_context::ExecContext = ExecContext(),
) where {P<:Real}
    target = likelihood * prior

    cycle = zero(Int)
    reset_rng_counters!(rng, id, cycle, 0)

    params_vec = Vector{P}(nparams(target))
    if isempty(initial_params)
        rand_initial_params!(rng, algorithm, prior, params_vec)
    else
        params_vec .= initial_params
    end

    !(params_vec in param_bounds(target)) && throw(ArgumentError("Parameter(s) out of bounds"))

    m = length(params_vec)

    log_value = density_logval(target, params_vec, exec_context)
    L = typeof(log_value)
    W = sample_weight_type(typeof(algorithm))

    current_sample = DensitySample(
        params_vec,
        log_value,
        one(W)
    )

    proposed_sample = DensitySample(
        similar(current_sample.params),
        convert(typeof(current_sample.log_value), NaN),
        zero(W)
    )

    state = MHState(
        algorithm,
        target,
        current_sample
    )

    chain = MCMCIterator(
        algorithm,
        likelihood,
        prior,
        target,
        state,
        rng,
        id,
        cycle,
        false,
        false
    )

    chain
end


exec_capabilities(mcmc_step!, callback::AbstractMCMCCallback, chain::MCMCIterator{<:MCMCAlgorithm{MHState}}) =
    exec_capabilities(density_logval, chain.target, chain.state.proposed_sample.params)


function mcmc_step!(
    callback::AbstractMCMCCallback,
    chain::MCMCIterator{<:MCMCAlgorithm{MHState}},
    exec_context::ExecContext,
    ll::LogLevel
)
    state = chain.state
    algorithm = chain.algorithm

    if !mcmc_compatible(algorithm, chain.state.pdist, param_bounds(chain.target))
        error("Implementation of algorithm $algorithm does not support current parameter bounds with current proposal distribution")
    end

    current_sample = state.current_sample
    proposed_sample = state.proposed_sample

    state.nsteps += 1
    reset_rng_counters!(chain)

    mcmc_propose_accept_reject!(callback, chain, exec_context)

    if state.proposal_accepted
        copy!(current_sample, proposed_sample)
        state.current_nreject = 0
        state.proposal_accepted = false
    end

    chain
end

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
} <: MCMCAlgorithm{MHState}
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


function MHState(
    algorithm::MetropolisHastings,
    target::AbstractDensity,
    current_sample::DensitySample{P,T,W}
) where {P,T,W}
    MHState(
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
            zero(T)
        end
    else
        p_accept = zero(T)
        proposed_sample.log_value = -Inf

        zero(T)
    end

    mh_acc_rej!(chain, p_accept)

    callback(1, chain)

    chain
end


function mh_acc_rej!(
    chain::MCMCIterator{<:MetropolisHastings{Q,W,WS}},
    p_accept::Real
) where {Q, W, WS <: MHMultiplicityWeights}
    @assert p_accept >= 0
    state = chain.state
    state.current_sample.weight += 1
    if rand(chain.rng, float(typeof(p_accept))) < p_accept
        state.proposal_accepted = true
        state.nsamples += 1
    else
        state.current_nreject += 1
    end
    chain
end


function mh_acc_rej!(
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
    else
        state.current_nreject += 1
    end
    chain
end


function mh_acc_rej!(
    chain::MCMCIterator{<:MetropolisHastings{Q,W,WS}},
    p_accept::Real
) where {Q, W, WS <: MHPosteriorFractionWeights}
    @assert p_accept >= 0

    state = chain.state

    r = rand(chain.rng, float(typeof(p_accept)))

    if p_accept ≈ 0
        state.current_sample.weight += 1
        state.proposed_sample.weight = 0
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
    end
    chain
end

