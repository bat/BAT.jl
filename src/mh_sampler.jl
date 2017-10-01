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
        0,
        0,
        1
    )
end


nparams(state::MHState) = nparams(state.pdist)

function next_cycle!(state::MHState)
    state.nsamples = 0
    state.nsteps = 0
    state
end


acceptance_ratio(state::MHState) = state.nsamples / state.nsteps


function MCMCBasicStats(state::MHState)
    s = state.current_sample
    L = promote_type(typeof(s.log_value), Float64)
    P = promote_type(eltype(s.params), Float64)
    m = length(s.params)
    MCMCBasicStats{L, P}(m)
end


sample_available(state::MHState, ::Val{:complete}) = state.proposal_accepted

function current_sample(state::MHState, ::Val{:complete})
    !state.proposal_accepted && error("No complete sample available")
    state.current_sample
end


sample_available(state::MHState, ::Val{:rejected}) = !state.proposal_accepted

function current_sample(state::MHState, ::Val{:rejected})
    state.proposal_accepted && error("No rejected sample available")
    state.proposed_sample
end


sample_available(state::MHState, ::Val{:any}) = true

function current_sample(state::MHState, ::Val{:any})
    ifelse(state.proposal_accepted, state.current_sample, state.proposed_sample)
end

function current_sampleno(state::MHState)
    state.nsamples + 1
end




struct MetropolisHastings <: MCMCAlgorithm{MHState} end
export MetropolisHastings



function MCMCChain(
    algorithm::MetropolisHastings,
    target::AbstractTargetSubject,
    pdist::AbstractProposalDist,
    id::Integer,
    rng::AbstractRNG,
    initial_params::AbstractVector{P},
    exec_context::ExecContext = ExecContext(),
) where {P<:Real}
    cycle = 0

    reset_rng_counters!(rng, MCMCSampleID(id, cycle, 0))

    params_vec = if isempty(initial_params)
        convert(Vector{P}, rand_initial_params(rng, target))
    else
        convert(Vector{P}, initial_params)
    end

    !(params_vec in target.bounds) && throw(ArgumentError("Parameter(s) out of bounds"))

    reset_rng_counters!(rng, MCMCSampleID(id, cycle, 1))

    m = length(params_vec)

    log_value = target_logval(target.tdensity, params_vec, exec_context)
    L = typeof(log_value)

    current_sample = DensitySample(
        params_vec,
        log_value,
        zero(Int)
    )

    proposed_sample = DensitySample(
        similar(current_sample.params),
        convert(typeof(current_sample.log_value), NaN),
        zero(Int)
    )

    state = MHState(
        pdist,
        current_sample
    )

    chain = MCMCChain(
        algorithm,
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


mcmc_compatible(::MetropolisHastings, pdist::AbstractProposalDist, bounds::UnboundedParams) = true

mcmc_compatible(::MetropolisHastings, pdist::AbstractProposalDist, bounds::HyperRectBounds) =
    issymmetric(pdist) || all(x -> x == hard_bounds, bounds.bt)


function mcmc_iterate!(
    callback,
    chain::MCMCChain{<:MetropolisHastings},
    exec_context::ExecContext = ExecContext();
    max_nsamples::Int64 = Int64(1),
    max_nsteps::Int = 1000,
    max_time::Float64 = Inf,
    ll::LogLevel = LOG_NONE
)
    algorithm = chain.algorithm
    cbfunc = mcmc_callback(callback)

    start_time = time()
    nsteps = 0
    nsamples = 0

    target = chain.target
    state = chain.state
    rng = chain.rng

    tdensity = target.tdensity
    bounds = target.bounds

    pdist = state.pdist
    current_sample = state.current_sample
    proposed_sample = state.proposed_sample

    T = typeof(current_sample.log_value)

    if !mcmc_compatible(algorithm, pdist, bounds)
        error("Implementation of algorithm $algorithm does not support current parameter bounds with current proposal distribution")
    end

    current_params = current_sample.params
    proposed_params = proposed_sample.params

    while nsamples < max_nsamples && nsteps < max_nsteps && (time() - start_time) < max_time
        if state.proposal_accepted
            reset_rng_counters!(rng, MCMCSampleID(chain))
            copy!(current_sample, proposed_sample)
            state.current_nreject = 0
            state.proposal_accepted = false
        end

        current_log_value = current_sample.log_value

        # Propose new parameters:
        proposal_rand!(rng, pdist, proposed_params, current_params)
        apply_bounds!(proposed_params, bounds, false)

        # log of ratio of forward/reverse transition probability
        log_tpr = if issymmetric(pdist)
            T(0)
        else
            log_tp_fwd = proposal_logpdf(pdist, proposed_params, current_params)
            log_tp_rev = proposal_logpdf(pdist, current_params, proposed_params)
            T(log_tp_fwd - log_tp_rev)
        end

        # Evaluate target density at new parameters:
        proposed_log_value = if proposed_params in bounds
            T(target_logval(tdensity, proposed_params, exec_context))
        else
            T(-Inf)
        end

        proposed_sample.log_value = proposed_log_value

        # Metropolis-Hastings accept/reject:
        accepted = rand(rng) < exp(proposed_log_value - current_log_value - log_tpr)

        nsteps += 1
        state.nsteps += 1

        if accepted
            current_sample.weight = state.current_nreject + 1
            state.proposal_accepted = true
            nsamples += 1
            state.nsamples += 1
        else
            state.current_nreject += 1
        end

        if accepted
            cbfunc(1, chain)
        else
            cbfunc(2, chain)
        end
    end
    chain
end

export mcmc_iterate!
