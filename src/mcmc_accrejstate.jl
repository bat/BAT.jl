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


nparams(state::AcceptRejectState) = nparams(state.pdist)

function next_cycle!(state::AcceptRejectState)
    state.nsamples = 0
    state.nsteps = 0
    state
end


acceptance_ratio(state::AcceptRejectState) = state.nsamples / state.nsteps


function MCMCBasicStats(state::AcceptRejectState)
    s = state.current_sample
    L = promote_type(typeof(s.log_value), Float64)
    P = promote_type(eltype(s.params), Float64)
    m = length(s.params)
    MCMCBasicStats{L, P}(m)
end


sample_available(state::AcceptRejectState, ::Val{:complete}) = state.proposal_accepted

function current_sample(state::AcceptRejectState, ::Val{:complete})
    !state.proposal_accepted && error("No complete sample available")
    state.current_sample
end


sample_available(state::AcceptRejectState, ::Val{:rejected}) = !state.proposal_accepted

function current_sample(state::AcceptRejectState, ::Val{:rejected})
    state.proposal_accepted && error("No rejected sample available")
    state.proposed_sample
end


sample_available(state::AcceptRejectState, ::Val{:any}) = true

function current_sample(state::AcceptRejectState, ::Val{:any})
    ifelse(state.proposal_accepted, state.current_sample, state.proposed_sample)
end

function current_sampleno(state::AcceptRejectState)
    state.nsamples + 1
end


function MCMCChain(
    algorithm::MCMCAlgorithm{AcceptRejectState},
    target::AbstractTargetSubject,
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

    state = AcceptRejectState(
        algorithm,
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
