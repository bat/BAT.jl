# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# TODO: MD, use Accessors.jl to make immutable
"""
    MCMCChainState

State of a MCMC chain.
"""
mutable struct MCMCChainState{
    M<:BATMeasure,
    PR<:RNGPartition,
    FT<:Function,
    P<:MCMCProposalState,
    WS<:AbstractMCMCWeightingScheme,
    SVX<:DensitySampleVector,
    SVZ<:DensitySampleVector,
    CTX<:BATContext
} <: MCMCIterator
    target::M
    proposal::P
    f_transform::FT
    weighting::WS
    current::@NamedTuple{x::SVX, z::SVZ}
    proposed::@NamedTuple{x::SVX, z::SVZ}
    output::SVX
    accepted::Vector{Bool}
    info::MCMCChainStateInfo
    rngpart_cycle::PR
    nsamples::Vector{Int64}
    stepno::Int64
    context::CTX
end
export MCMCChainState

function MCMCChainState(
    samplingalg::TransformedMCMC,
    target::BATMeasure,
    chainid::Integer,
    x_init::AbstractVector{PV},
    context::BATContext
) where {P<:Real, PV<:AbstractVector{P}}
    n_walkers = length(x_init)
    target_unevaluated = unevaluated(target)

    rngpart_cycle = RNGPartition(get_rng(context), 0:(typemax(Int16) - 2))
    rng = get_rng(context)

    f = init_adaptive_transform(samplingalg.adaptive_transform, target, context)
    f_inv = inverse(f)
    proposal = _create_proposal_state(samplingalg.proposal, target, context, x_init, f, rng)

    logd_x_init = logdensityof.(target_unevaluated, x_init)
    z_init = f_inv.(x_init) 
    logd_z_init = logdensityof.(MeasureBase.pullback(f, target_unevaluated), z_init)

    W = mcmc_weight_type(samplingalg.sample_weighting)

    sample_weights_curr = zeros(W, n_walkers)
    sample_info_curr = MCMCSampleID[MCMCSampleID(Int32(chainid), Int32(i), one(Int32), zero(Int64), get_current_proposal_idx(proposal), true) for i in 1:n_walkers]
    sample_aux_curr = fill(nothing, n_walkers)

    current_x_init = DensitySampleVector(
        x_init, 
        logd_x_init;
        weight = sample_weights_curr, 
        info = sample_info_curr,
        aux = sample_aux_curr
    )
    current_z_init = DensitySampleVector(
        z_init, 
        logd_z_init;
        weight = deepcopy(sample_weights_curr), 
        info = deepcopy(sample_info_curr),
        aux = deepcopy(sample_aux_curr)
    )

    prop_locs_init = deepcopy(x_init)
    prop_logds_init = deepcopy(logd_x_init)
    sample_weights_prop = zeros(W, n_walkers)
    sample_info_prop = deepcopy(sample_info_curr) 
    sample_aux_prop = fill(nothing, n_walkers)

    proposed_init = DensitySampleVector(
        prop_locs_init,
        prop_logds_init;
        weight = sample_weights_prop,
        info = sample_info_prop,
        aux = sample_aux_prop 
    )

    current = (x = current_x_init, z = current_z_init)
    proposed = (x = deepcopy(proposed_init), z = deepcopy(proposed_init))
    output = deepcopy(current_x_init)
    accepted = fill(false, n_walkers)

    stepno::Int64 = 0
    cycle::Int32 = 0

    n_proposals = proposal isa MultiProposalState ? length(proposal.proposal_states) : 1
    nsamples::Vector{Int64} = zeros(n_proposals)

    state = MCMCChainState(
        target,
        proposal,
        f,
        samplingalg.sample_weighting,
        current,
        proposed, 
        output, 
        accepted,
        MCMCChainStateInfo(chainid, cycle, false, false),
        rngpart_cycle,
        nsamples,
        stepno,
        context
    )

    state
end

get_proposal(state::MCMCChainState) = state.proposal

mcmc_target(state::MCMCChainState) = state.target

get_context(state::MCMCChainState) = state.context

mcmc_info(state::MCMCChainState) = state.info

nsamples(state::MCMCChainState) = sum(state.nsamples)

detailed_nsamples(state::MCMCChainState) = state.nsamples

nsteps(state::MCMCChainState) = state.stepno

nwalkers(state::MCMCChainState) = length(state.current.x.v)

current_sample(state::MCMCChainState) = state.current.x

proposed_sample(state::MCMCChainState) = state.proposed.x

current_sample_z(state::MCMCChainState) = state.current.z

proposed_sample_z(state::MCMCChainState) = state.proposed.z

sample_type(state::MCMCChainState) = eltype(state.current.x)


mcmc_target(state::MCMCState) = mcmc_target(state.chain_state)

nsamples(state::MCMCState) = nsamples(state.chain_state)

nsteps(state::MCMCState) = nsteps(state.chain_state)

nwalkers(state::MCMCState) = nwalkers(state.chain_state)


_empty_DensitySampleVector(state::MCMCState) =  _empty_DensitySampleVector(state.chain_state)

function _empty_DensitySampleVector(chain_state::MCMCChainState)
    return DensitySampleVector(sample_type(chain_state), totalndof(varshape(mcmc_target(chain_state))))
end


_empty_chain_outputs(state::MCMCState) = _empty_chain_outputs(state.chain_state)

function _empty_chain_outputs(chain_state::MCMCChainState)
    return fill(_empty_DensitySampleVector(chain_state), nwalkers(chain_state))
end

function eff_acceptance_ratio(chain_state::MCMCChainState)
    current_proposal = get_current_proposal(chain_state.proposal)
    return nsamples(chain_state) / (nsteps(chain_state) * nwalkers(chain_state))
end

function detailed_eff_acceptance_ratio(chain_state::MCMCChainState)
    current_proposal = get_current_proposal(chain_state.proposal)
    return detailed_nsamples(chain_state) ./ (nsteps(chain_state) * nwalkers(chain_state))
end


function mcmc_step!!(mcmc_state::MCMCState)
    reset_rng_counters!(mcmc_state)

    chain_state = mcmc_state.chain_state
    chain_state.stepno += 1

    (;proposal, stepno, context) = chain_state

    rng = get_rng(context)

    chain_state.proposal = set_current_proposal!!(proposal, stepno, rng)

    current_proposal = get_current_proposal(chain_state.proposal)

    chain_state, p_accept = mcmc_propose!!(chain_state, current_proposal)

    mcmc_state_new = mcmc_tune_post_step!!(mcmc_state, current_proposal, p_accept)

    chain_state = mcmc_state_new.chain_state

    (;proposal, current, proposed, accepted, output) = chain_state

    curr_prop_idx = get_current_proposal_idx(proposal)
    chain_state.nsamples[curr_prop_idx] += sum(accepted)

    # Set weights according to acceptance
    delta_w_current, w_proposed = mcmc_weight_values(chain_state.weighting, p_accept, accepted)

    current.x.weight .+= delta_w_current
    current.z.weight .+= delta_w_current

    proposed.x.weight .= w_proposed
    proposed.z.weight .= w_proposed

    idxs_acc = findall(accepted)
    idxs_rej = findall(!, accepted)

    # For each walker, mark proposed samples as accepted or rejected
    for i in eachindex(proposed.x)
        old_info = proposed.x.info[i]

        sample_type = accepted[i]
        new_info = MCMCSampleID(
            old_info.chainid, 
            Int32(i), 
            old_info.chaincycle, 
            chain_state.stepno, 
            get_current_proposal_idx(proposal), 
            sample_type
        )

        proposed.x.info[i] = new_info
        proposed.z.info[i] = new_info
    end

    # Save current points to output if they will be overwritten, and save rejected proposed points
    output[idxs_acc] = @view current.x[idxs_acc]
    output[idxs_rej] = @view proposed.x[idxs_rej]

    # Overwrite current points with accepted proposed points
    current.x[idxs_acc] = @view proposed.x[idxs_acc]
    current.z[idxs_acc] = @view proposed.z[idxs_acc]

    chain_state = mcmc_state_new.chain_state
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

function next_cycle!(chain_state::MCMCChainState)
    n_walkers = nwalkers(chain_state)

    chain_state.info = MCMCChainStateInfo(chain_state.info.id,
                                          chain_state.info.cycle + 1,
                                          chain_state.info.tuned,
                                          chain_state.info.converged
                                          )
    chain_state.nsamples .= 0
    chain_state.stepno = 0

    proposal = chain_state.proposal
    info = chain_state.info

    new_current_info_vec = [MCMCSampleID(
        info.id,
        Int32(i),
        info.cycle,
        zero(Int64),
        get_current_proposal_idx(proposal),
        true
        ) for i in 1:n_walkers]
    chain_state.current.x.info .= new_current_info_vec
    chain_state.current.z.info .= new_current_info_vec

    new_proposed_info_vec = [MCMCSampleID(
        info.id,
        Int32(i),
        info.cycle,
        zero(Int64),
        get_current_proposal_idx(proposal),
        false
        ) for i in 1:n_walkers]
    chain_state.proposed.x.info .= new_proposed_info_vec
    chain_state.proposed.z.info .= new_proposed_info_vec

    reset_rng_counters!(chain_state)

    chain_state
end

function next_cycle!(state::MCMCState)
    next_cycle!(state.chain_state)
end

# This assumes 'appendable' to be a vector of appendables that respectively hold the samples for each walker
function get_samples!(appendable, chain_state::MCMCChainState, nonzero_weights::Bool)::typeof(appendable)
    chain_output = chain_state.output
    viable_samples = nonzero_weights ? findall(chain_output.weight .> 0) : eachindex(chain_output)

    for i in viable_samples
        # If last sample in appendable[i] is equal to the new sample increment its weight, otherwise append new sample
        checked_push!(appendable[i], chain_output[i])
    end

    appendable
end

function get_samples!(appendable, mcmc_state::MCMCState, nonzero_weights::Bool)::typeof(appendable)
    get_samples!(appendable, mcmc_state.chain_state, nonzero_weights)
end


# TDOD: MD, make properly !!
function flush_samples!!(chain_state::MCMCChainState)
    (;current, output) = chain_state

    output[:] = @view current.x[:]
    current.x.weight .= 0

    return chain_state
end

function flush_samples!!(mcmc_state::MCMCState)
    new_mcmc_state = @set mcmc_state.chain_state = flush_samples!!(mcmc_state.chain_state)
    return new_mcmc_state
end


function mcmc_update_z_position!!(mcmc_state::MCMCState)
    chain_state_new = mcmc_update_z_position!!(mcmc_state.chain_state)

    mcmc_state_new = @set mcmc_state.chain_state = chain_state_new
    return mcmc_state_new
end

function mcmc_update_z_position!!(mc_state::MCMCChainState)
    f_inv = inverse(mc_state.f_transform)

    current_z_new::typeof(mc_state.current.z) = transform_samples(f_inv, mc_state.current.x)
    proposed_z_new::typeof(mc_state.proposed.z) = transform_samples(f_inv, mc_state.proposed.x)

    mc_state_new::typeof(mc_state) = @set mc_state.current.z = current_z_new
    mc_state_new = @set mc_state_new.proposed.z = proposed_z_new

    return mc_state_new
end

# TODO: MD, Discuss:
# When using different Tuners for proposal and transformation, which should be applied first? 
# And if the z-position changes during the transformation tuning, should the proposal Tuner run on the updated z-position?
function mcmc_tuning_init!!(state::MCMCState, max_nsteps::Integer)
    # TODO: mcmc_tuning_init!! should support immutable tuners and states and return the new objects
    mcmc_tuning_init!!(state.trafo_tuner_state, state.chain_state, max_nsteps)
    mcmc_tuning_init!!(state.proposal_tuner_state, state.chain_state, max_nsteps)
end

function mcmc_tuning_reinit!!(state::MCMCState, max_nsteps::Integer)
    # TODO: mcmc_tuning_reinit!! should support immutable tuners and states and return the new objects
    mcmc_tuning_reinit!!(state.trafo_tuner_state, state.chain_state, max_nsteps)
    mcmc_tuning_reinit!!(state.proposal_tuner_state, state.chain_state, max_nsteps)
end

function mcmc_tuning_postinit!!(state::MCMCState, samples::AbstractVector{<:DensitySampleVector})
    # TODO: mcmc_tuning_postinit!! should support immutable tuners and states and return the new objects
    mcmc_tuning_postinit!!(state.trafo_tuner_state, state.chain_state, samples)
    mcmc_tuning_postinit!!(state.proposal_tuner_state, state.chain_state, samples)
end

# TODO: MD, when should the z-position be updated? Before or after the proposal tuning?
function mcmc_tune_post_cycle!!(state::MCMCState, samples::AbstractVector{<:DensitySampleVector})
    f_transform_tuned, trafo_tuner_state_new, chain_state_trafo_tuned = mcmc_tune_post_cycle!!(
        state.chain_state.f_transform,
        state.trafo_tuner_state,
        state.chain_state,
        state.chain_state.proposal,
        samples
    )

    chain_state_trafo_tuned = @set chain_state_trafo_tuned.f_transform = f_transform_tuned
    proposal = chain_state_trafo_tuned.proposal
    proposal = set_proposal_transform!!(proposal, chain_state_trafo_tuned)
    chain_state_trafo_tuned = mcmc_update_z_position!!(chain_state_trafo_tuned)

    proposal_state_new, proposal_tuner_state_new, chain_state_new = mcmc_tune_post_cycle!!(
        proposal,
        state.proposal_tuner_state,
        chain_state_trafo_tuned,
        samples
    )

    chain_state_final = @set chain_state_new.proposal = proposal_state_new

    mcmc_state_cs = @set state.chain_state = chain_state_final
    mcmc_state_tt = @set mcmc_state_cs.trafo_tuner_state = trafo_tuner_state_new
    mcmc_state_pt = @set mcmc_state_tt.proposal_tuner_state = proposal_tuner_state_new

    return mcmc_state_pt
end

function mcmc_tune_post_step!!(state::MCMCState, proposal::MCMCProposalState, p_accept::AbstractVector{<:Real})
    f_transform_tuned, trafo_tuner_state_new, chain_state_trafo_tuned = mcmc_tune_post_step!!(
        state.chain_state.f_transform,
        state.trafo_tuner_state,
        state.chain_state,
        proposal,
        state.chain_state.current,
        state.chain_state.proposed,
        p_accept
    )

    chain_state_trafo_tuned = @set chain_state_trafo_tuned.f_transform = f_transform_tuned
    proposal = chain_state_trafo_tuned.proposal
    proposal = set_proposal_transform!!(proposal, chain_state_trafo_tuned)
    chain_state_trafo_tuned = mcmc_update_z_position!!(chain_state_trafo_tuned)

    proposal_state_new, proposal_tuner_state_new, chain_state_new = mcmc_tune_post_step!!(
        proposal,
        state.proposal_tuner_state,
        chain_state_trafo_tuned,
        p_accept
    )

    chain_state_final = @set chain_state_new.proposal = proposal_state_new

    # TODO: MD, inelegant, use AccessorsExtra.jl to set several fields at once? https://github.com/JuliaAPlavin/AccessorsExtra.jl
    mcmc_state_cs = @set state.chain_state = chain_state_final
    mcmc_state_tt = @set mcmc_state_cs.trafo_tuner_state = trafo_tuner_state_new
    mcmc_state_pt = @set mcmc_state_tt.proposal_tuner_state = proposal_tuner_state_new

    return mcmc_state_pt
end

function mcmc_tuning_finalize!!(state::MCMCState)
    mcmc_tuning_finalize!!(state.chain_state.f_transform, state.trafo_tuner_state, state.chain_state)
    mcmc_tuning_finalize!!(state.chain_state.proposal, state.proposal_tuner_state, state.chain_state)
end


function _construct_mcmc_state(
    samplingalg::TransformedMCMC,
    target::BATMeasure,
    rngpart::RNGPartition,
    id::Integer,
    initval_alg::InitvalAlgorithm,
    parent_context::BATContext
)
    new_context = set_rng(parent_context, AbstractRNG(rngpart, id))
    v_init = bat_ensemble_initvals(target, initval_alg, samplingalg.nwalkers, new_context)
    return MCMCState(samplingalg, target, Int32(id), v_init, new_context)
end

_gen_mcmc_states(
    samplingalg::TransformedMCMC,
    target::BATMeasure,
    rngpart::RNGPartition,
    ids::AbstractRange{<:Integer},
    initval_alg::InitvalAlgorithm,
    context::BATContext
) = [_construct_mcmc_state(samplingalg, target, rngpart, id, initval_alg, context) for id in ids]
