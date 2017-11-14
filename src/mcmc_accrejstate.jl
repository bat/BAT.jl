# This file is a part of BAT.jl, licensed under the MIT License (MIT).


mutable struct AcceptRejectState{
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

function AcceptRejectState(
    pdist::AbstractProposalDist,
    current_sample::DensitySample
)
    proposed_sample = DensitySample(
        similar(current_sample.params),
        convert(typeof(current_sample.log_value), NaN),
        zero(current_sample.weight)
    )

    AcceptRejectState(
        pdist,
        current_sample,
        proposed_sample,
        false,
        zero(Int64),
        zero(Int64),
        one(Int64)
    )
end


nparams(state::AcceptRejectState) = nparams(state.pdist)

nsteps(state::AcceptRejectState) = state.nsteps

nsamples(state::AcceptRejectState) = state.nsamples

acceptance_ratio(state::AcceptRejectState) = nsamples(state) / nsteps(state)


function next_cycle!(state::AcceptRejectState)
    state.current_sample.weight = one(state.current_sample.weight)
    state.nsamples = zero(Int64)
    state.nsteps = zero(Int64)
    state
end


function MCMCBasicStats(state::AcceptRejectState)
    s = state.current_sample
    L = promote_type(typeof(s.log_value), Float64)
    P = promote_type(eltype(s.params), Float64)
    m = length(s.params)
    MCMCBasicStats{L, P}(m)
end


sample_available(state::AcceptRejectState, ::Val{:complete}) =
    state.proposal_accepted || state.proposed_sample.weight > 0

function current_sample(state::AcceptRejectState, ::Val{:complete})
    if state.proposal_accepted
        @assert state.current_sample.weight > 0
        state.current_sample
    elseif state.proposed_sample.weight > 0
        state.proposed_sample
    else
        error("No complete sample available")
    end
end


sample_available(state::AcceptRejectState, ::Val{:rejected}) =
    !state.proposal_accepted && state.proposed_sample.weight <= 0

function current_sample(state::AcceptRejectState, ::Val{:rejected})
    if !state.proposal_accepted && state.proposed_sample.weight <= 0
        state.proposed_sample
    else
        error("No rejected sample available")
    end
end


sample_available(state::AcceptRejectState, ::Val{:any}) = true

function current_sample(state::AcceptRejectState, ::Val{:any})
    ifelse(state.proposal_accepted, state.current_sample, state.proposed_sample)
end

function current_sampleno(state::AcceptRejectState)
    state.nsamples + 1
end


function MCMCIterator(
    algorithm::MCMCAlgorithm{AcceptRejectState},
    likelihood::AbstractDensity,
    prior::AbstractDensity,
    id::Int64,
    rng::AbstractRNG,
    initial_params::AbstractVector{P} = Vector{P}(),
    exec_context::ExecContext = ExecContext(),
) where {P<:Real}
    target = likelihood * prior

    cycle = zero(Int)
    reset_rng_counters!(rng, MCMCSampleID(id, cycle, 0))

    params_vec = Vector{P}(nparams(target))
    if isempty(initial_params)
        rand_initial_params!(rng, algorithm, prior, params_vec)
    else
        params_vec .= initial_params
    end

    !(params_vec in param_bounds(target)) && throw(ArgumentError("Parameter(s) out of bounds"))

    reset_rng_counters!(rng, MCMCSampleID(id, cycle, 1))

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

    state = AcceptRejectState(
        algorithm,
        target,
        current_sample
    )

    chain = MCMCIterator(
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


function mcmc_step!(
    callback::AbstractMCMCCallback,
    chain::MCMCIterator{<:MCMCAlgorithm{AcceptRejectState}},
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

    if state.proposal_accepted
        reset_rng_counters!(chain.rng, MCMCSampleID(chain))
        copy!(current_sample, proposed_sample)
        state.current_nreject = 0
        state.proposal_accepted = false
    end

    mcmc_propose_accept_reject!(callback, chain, exec_context)

    state.nsteps += 1

    chain
end
