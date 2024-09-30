# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# TODO: MD, adjust docstring to new typestructure
# TODO: MD, use Accessors.jl to make immutable 
mutable struct MCMCChainState{
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
    info::MCMCChainStateInfo
    rngpart_cycle::PR
    nsamples::Int64
    stepno::Int64
    context::CTX
end
export MCMCChainState

function MCMCChainState(
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

    state = MCMCChainState(
        target,
        proposal,
        g,
        samples,
        sample_z,
        MCMCChainStateInfo(id, cycle, false, false),
        rngpart_cycle,
        nsamples,
        stepno,
        context
    )

    # TODO: MD, resetting the counters necessary/desired? 
    reset_rng_counters!(state)

    state
end

@inline _current_sample_idx(chain_state::MCMCChainState) = firstindex(chain_state.samples)
@inline _proposed_sample_idx(chain_state::MCMCChainState) = lastindex(chain_state.samples)

@inline _current_sample_z_idx(chain_state::MCMCChainState) = firstindex(chain_state.sample_z)
@inline _proposed_sample_z_idx(chain_state::MCMCChainState) = lastindex(chain_state.sample_z)

@inline _current_sample_idx(mcmc_state::MCMCState) = firstindex(mcmc_state.chain_state.samples)
@inline _proposed_sample_idx(mcmc_state::MCMCState) = lastindex(mcmc_state.chain_state.samples)

@inline _current_sample_z_idx(mcmc_state::MCMCState) = firstindex(mcmc_state.chain_state.sample_z)
@inline _proposed_sample_z_idx(mcmc_state::MCMCState) = lastindex(mcmc_state.chain_state.sample_z)


get_proposal(state::MCMCChainState) = state.proposal

mcmc_target(state::MCMCChainState) = state.target

get_context(state::MCMCChainState) = state.context

mcmc_info(state::MCMCChainState) = state.info

nsteps(state::MCMCChainState) = state.stepno

nsamples(state::MCMCChainState) = state.nsamples

current_sample(state::MCMCChainState) = state.samples[_current_sample_idx(state)]

proposed_sample(state::MCMCChainState) = state.samples[_proposed_sample_idx(state)]

current_sample_z(state::MCMCChainState) = state.sample_z[_current_sample_z_idx(state)]

proposed_sample_z(state::MCMCChainState) = state.sample_z[_proposed_sample_z_idx(state)]

sample_type(state::MCMCChainState) = eltype(state.samples)


mcmc_target(state::MCMCState) = mcmc_target(state.chain_state)

nsamples(state::MCMCState) = nsamples(state.chain_state)

nsteps(state::MCMCState) = nsteps(state.chain_state)


function DensitySampleVector(states::MCMCState)
    DensitySampleVector(sample_type(states.chain_state), totalndof(varshape(mcmc_target(states))))
end

function DensitySampleVector(chain_state::MCMCChainState)
    DensitySampleVector(sample_type(chain_state), totalndof(varshape(mcmc_target(chain_state))))
end

# TODO: MD, make into !!
function mcmc_step!!(mcmc_state::MCMCState)
    global g_state_step = mcmc_state


    # TODO: MD, include sample_z in _cleanup_samples()
    _cleanup_samples(mcmc_state)
    reset_rng_counters!(mcmc_state)

    chain_state = mcmc_state.chain_state

    @unpack target, proposal, f_transform, samples, sample_z, nsamples, context = chain_state
    
    chain_state.stepno += 1
    
    resize!(samples, size(samples, 1) + 1)

    samples.info[lastindex(samples)] = _get_sample_id(proposal, chain_state.info.id, chain_state.info.cycle, chain_state.stepno, PROPOSED_SAMPLE)[1]

    chain_state, accepted, p_accept = mcmc_propose!!(chain_state)

    mcmc_state_new = mcmc_tune_post_step!!(mcmc_state, p_accept)

    chain_state = mcmc_state_new.chain_state

    current = _current_sample_idx(chain_state)
    proposed = _proposed_sample_idx(chain_state)

    _accept_reject!(chain_state, accepted, p_accept, current, proposed)

    mcmc_state_final = @set mcmc_state_new.chain_state = chain_state

    return mcmc_state_final
end


function reset_rng_counters!(chain_state::MCMCChainState)
    rng = get_rng(get_context(chain_state))
    set_rng!(rng, chain_state.rngpart_cycle, chain_state.info.cycle)
    rngpart_step = RNGPartition(rng, 0:(typemax(Int32) - 2))
    set_rng!(rng, rngpart_step, chain_state.stepno)
    nothing
end

function reset_rng_counters!(mcmc_state::MCMCState)
    reset_rng_counters!(mcmc_state.chain_state)
end

function _cleanup_samples(chain_state::MCMCChainState)
    samples = chain_state.samples
    current = _current_sample_idx(chain_state)
    proposed = _proposed_sample_idx(chain_state)
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

function _cleanup_samples(mcmc_state::MCMCState)
    _cleanup_samples(mcmc_state.chain_state)
end

function next_cycle!(chain_state::MCMCChainState)
    _cleanup_samples(chain_state)

    chain_state.info = MCMCChainStateInfo(chain_state.info, cycle = chain_state.info.cycle + 1)
    chain_state.nsamples = 0
    chain_state.stepno = 0

    reset_rng_counters!(chain_state)

    resize!(chain_state.samples, 1)

    i = _proposed_sample_idx(chain_state)
    @assert chain_state.samples.info[i].sampletype == CURRENT_SAMPLE
    chain_state.samples.weight[i] = 1

    chain_state.samples.info[i] = MCMCSampleID(chain_state.info.id, chain_state.info.cycle, chain_state.stepno, CURRENT_SAMPLE)

    chain_state
end

function next_cycle!(state::MCMCState)
    next_cycle!(state.chain_state)
end


function get_samples!(appendable, chain_state::MCMCChainState, nonzero_weights::Bool)::typeof(appendable)
    if samples_available(chain_state)
        samples = chain_state.samples

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

function get_samples!(appendable, mcmc_state::MCMCState, nonzero_weights::Bool)::typeof(appendable)
    get_samples!(appendable, mcmc_state.chain_state, nonzero_weights)
end


function samples_available(chain_state::MCMCChainState)
    i = _current_sample_idx(chain_state)
    chain_state.samples.info.sampletype[i] == ACCEPTED_SAMPLE
end

function samples_available(mcmc_state::MCMCState)
    samples_available(mcmc_state.chain_state)
end

function mcmc_update_z_position!!(mc_state::MCMCChainState)

    proposed_sample_x = proposed_sample(mc_state)

    x_proposed, logd_x_proposed = proposed_sample_x.v, proposed_sample_x.logd 

    z_new, ladj_inv = with_logabsdet_jacobian(inverse(mc_state.f_transform), vec(x_proposed))

    logd_z_new = logd_x_proposed - ladj_inv

    mc_state_tmp = @set mc_state.sample_z.v[2] = vec(z_new)
    mc_state_new = @set mc_state_tmp.sample_z.logd[2] = logd_z_new

    return mc_state_new
end

# TODO: MD, Discuss: 
# When using different Tuners for proposal and transformation, which should be applied first? 
# And if the z-position changes during the transformation tuning, should the proposal Tuner run on the updated z-position?
function mcmc_tuning_init!!(state::MCMCState, max_nsteps::Integer)
    mcmc_tuning_init!!(state.trafo_tuner_state, state.chain_state, max_nsteps)
    mcmc_tuning_init!!(state.proposal_tuner_state, state.chain_state, max_nsteps)
end

function mcmc_tuning_reinit!!(state::MCMCState, max_nsteps::Integer)
    mcmc_tuning_reinit!!(state.trafo_tuner_state, state.chain_state, max_nsteps)
    mcmc_tuning_reinit!!(state.proposal_tuner_state, state.chain_state, max_nsteps)
end

function mcmc_tuning_postinit!!(state::MCMCState, samples::DensitySampleVector)
    mcmc_tuning_postinit!!(state.trafo_tuner_state, state.chain_state, samples)
    mcmc_tuning_postinit!!(state.proposal_tuner_state, state.chain_state, samples)
end

# TODO: MD, when should the z-position be updated? Before or after the proposal tuning?
function mcmc_tune_post_cycle!!(state::MCMCState, samples::DensitySampleVector)
    chain_state_tmp, trafo_tuner_state_new, trafo_changed = mcmc_tune_post_cycle!!(state.trafo_tuner_state, state.chain_state, samples)
    chain_state_new, proposal_tuner_state_new, _ = mcmc_tune_post_cycle!!(state.proposal_tuner_state, chain_state_tmp, samples)

    if trafo_changed
        chain_state_new = mcmc_update_z_position!!(chain_state_new)
    end

    mcmc_state_cs = @set state.chain_state = chain_state_new
    mcmc_state_tt = @set mcmc_state_cs.trafo_tuner_state = trafo_tuner_state_new
    mcmc_state_pt = @set mcmc_state_tt.proposal_tuner_state = proposal_tuner_state_new

    return mcmc_state_pt
end

function mcmc_tune_post_step!!(state::MCMCState, p_accept::Real)
    chain_state_tmp, trafo_tuner_state_new, trafo_changed = mcmc_tune_post_step!!(state.trafo_tuner_state, state.chain_state, p_accept)
    chain_state_new, proposal_tuner_state_new, _ = mcmc_tune_post_step!!(state.proposal_tuner_state, chain_state_tmp, p_accept)

    if trafo_changed
        chain_state_new = mcmc_update_z_position!!(chain_state_new)
    end

    # TODO: MD, inelegant, use AccessorsExtra.jl to set several fields at once? https://github.com/JuliaAPlavin/AccessorsExtra.jl
    mcmc_state_cs = @set state.chain_state = chain_state_new
    mcmc_state_tt = @set mcmc_state_cs.trafo_tuner_state = trafo_tuner_state_new
    mcmc_state_pt = @set mcmc_state_tt.proposal_tuner_state = proposal_tuner_state_new
    
    return mcmc_state_pt
end

function mcmc_tuning_finalize!!(state::MCMCState)
    mcmc_tuning_finalize!!(state.trafo_tuner_state, state.chain_state)
    mcmc_tuning_finalize!!(state.proposal_tuner_state, state.chain_state)
end 
