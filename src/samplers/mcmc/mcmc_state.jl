# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# TODO: MD, adjust docstring to new typestructure
mutable struct MCMCState{
    M<:BATMeasure,
    PR<:RNGPartition,
    FT<:Function,
    P<:MCMCProposalState,
    SVX<:DensitySampleVector,
    SVZ<:DensitySampleVector,
    CTX<:BATContext
} <: MCMCIterator
    target::M
    proposal::P
    f_transform::FT
    samples::SVX
    sample_z::SVZ
    info::MCMCStateInfo
    rngpart_cycle::PR
    nsamples::Int64
    stepno::Int64
    context::CTX
end
export MCMCState

function MCMCState(
    sampling::MCMCSampling,
    target::BATMeasure,
    id::Integer,
    v_init::AbstractVector{P},
    context::BATContext
    ) where {P<:Real}
    
    rngpart_cycle = RNGPartition(get_rng(context), 0:(typemax(Int16) - 2))
    rng = get_rng(context)
    n_dims = getdof(target)
    
    #Create Proposal state. Necessary in particular for AHMC proposal
    proposal = _create_proposal_state(sampling.proposal, target, context, v_init, rng)
    stepno::Int64 = 0

    cycle::Int32 = 0
    nsamples::Int64 = 0

    g = init_adaptive_transform(sampling.adaptive_transform, target, context)

    logd_x = logdensityof(target, v_init)
    inverse_g = inverse(g)
    z = inverse_g(v_init)
    logd_z = logdensityof(MeasureBase.pullback(g, target), z)

    W = _weight_type(proposal.weighting)
    T = typeof(logd_x)

    info, sample_id_type = _get_sample_id(proposal, Int32(id), cycle, 1, CURRENT_SAMPLE)
    sample_x = DensitySample(v_init, logd_x, one(W), info, nothing)

    samples = DensitySampleVector{Vector{P}, T, W, sample_id_type, Nothing}(undef, 0, n_dims)
    push!(samples, sample_x)

    sample_z = DensitySampleVector{Vector{P}, T, W, sample_id_type, Nothing}(undef, 0, n_dims) 
    sample_z_current = DensitySample(z, logd_z, one(W), info, nothing)
    sample_z_proposed = DensitySample(z, logd_z, one(W), _get_sample_id(proposal, Int32(id), cycle, 1, PROPOSED_SAMPLE)[1], nothing)
    push!(sample_z, sample_z_current, sample_z_proposed)

    state = MCMCState(
        target,
        proposal,
        g,
        samples,
        sample_z,
        MCMCStateInfo(id, cycle, false, false),
        rngpart_cycle,
        nsamples,
        stepno,
        context
    )

    # TODO: MD, resetting the counters necessary/desired? 
    reset_rng_counters!(state)

    state
end

@inline _current_sample_idx(mc_state::MCMCState) = firstindex(mc_state.samples)
@inline _proposed_sample_idx(mc_state::MCMCState) = lastindex(mc_state.samples)

@inline _current_sample_z_idx(mc_state::MCMCState) = firstindex(mc_state.sample_z)
@inline _proposed_sample_z_idx(mc_state::MCMCState) = lastindex(mc_state.sample_z)


get_proposal(state::MCMCState) = state.proposal

mcmc_target(state::MCMCState) = state.target

get_context(state::MCMCState) = state.context

mcmc_info(state::MCMCState) = state.info

nsteps(state::MCMCState) = state.stepno

nsamples(state::MCMCState) = state.nsamples

current_sample(state::MCMCState) = state.samples[_current_sample_idx(state)]

proposed_sample(state::MCMCState) = state.samples[_proposed_sample_idx(state)]

current_sample_z(state::MCMCState) = state.sample_z[_current_sample_z_idx(state)]

proposed_sample_z(state::MCMCState) = state.sample_z[_proposed_sample_z_idx(state)]

sample_type(state::MCMCState) = eltype(state.samples)


function DensitySampleVector(mc_state::MCMCState)
    DensitySampleVector(sample_type(mc_state), totalndof(varshape(mcmc_target(mc_state))))
end


function mcmc_step!(mc_state::MCMCState, tuner::Union{AbstractMCMCTunerInstance, Nothing}, temperer::Union{AbstractMCMCTemperingInstance, Nothing})
    # TODO: MD, include sample_z in _cleanup_samples()
    _cleanup_samples(mc_state)
    reset_rng_counters!(mc_state)

    @unpack target, proposal, f_transform, samples, sample_z, nsamples, stepno, context = mc_state
    rng = get_rng(context)
    
    mc_state.stepno += 1
    
    resize!(samples, size(samples, 1) + 1)

    samples.info[lastindex(samples)] = _get_sample_id(proposal, mc_state.info.id, mc_state.info.cycle, mc_state.stepno, PROPOSED_SAMPLE)[1]

    mc_state, accepted, p_accept = mcmc_propose!!(mc_state)

    tuner_new, f_transform_tuned = mcmc_tune_transform!!(mc_state, tuner, p_accept)

    # TODO: MD, Discuss updating of 'sample_z' due to possibly changed 'f_transform' during transfom tuning_callback

    current = _current_sample_idx(mc_state)
    proposed = _proposed_sample_idx(mc_state)

    _accept_reject!(mc_state, accepted, p_accept, current, proposed)

    nothing
end


function reset_rng_counters!(mc_state::MCMCState)
    rng = get_rng(get_context(mc_state))
    set_rng!(rng, mc_state.rngpart_cycle, mc_state.info.cycle)
    rngpart_step = RNGPartition(rng, 0:(typemax(Int32) - 2))
    set_rng!(rng, rngpart_step, mc_state.stepno)
    nothing
end

function _cleanup_samples(mc_state::MCMCState)
    samples = mc_state.samples
    current = _current_sample_idx(mc_state)
    proposed = _proposed_sample_idx(mc_state)
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

function next_cycle!(mc_state::MCMCState)
    _cleanup_samples(mc_state)

    mc_state.info = MCMCStateInfo(mc_state.info, cycle = mc_state.info.cycle + 1)
    mc_state.nsamples = 0
    mc_state.stepno = 0

    reset_rng_counters!(mc_state)

    resize!(mc_state.samples, 1)

    i = _proposed_sample_idx(mc_state)
    @assert mc_state.samples.info[i].sampletype == CURRENT_SAMPLE
    mc_state.samples.weight[i] = 1

    mc_state.samples.info[i] = MCMCSampleID(mc_state.info.id, mc_state.info.cycle, mc_state.stepno, CURRENT_SAMPLE)

    mc_state
end


function get_samples!(appendable, mc_state::MCMCState, nonzero_weights::Bool)::typeof(appendable)
    if samples_available(mc_state)
        samples = mc_state.samples

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

function samples_available(mc_state::MCMCState)
    i = _current_sample_idx(mc_state)
    mc_state.samples.info.sampletype[i] == ACCEPTED_SAMPLE
end
