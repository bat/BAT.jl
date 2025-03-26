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
    nsamples::Int64
    stepno::Int64
    context::CTX
end
export MCMCChainState

function MCMCChainState(
    samplingalg::TransformedMCMC,
    target::BATMeasure,
    id::Integer,
    x_init::AbstractVector{PV},
    context::BATContext
) where {P<:Real, PV<:AbstractVector{P}}
    n_walkers = length(x_init)
    target_unevaluated = unevaluated(target)
    
    rngpart_cycle = RNGPartition(get_rng(context), 0:(typemax(Int16) - 2))
    rng = get_rng(context)
    n_dims = getdof(target)
     
    f = init_adaptive_transform(samplingalg.adaptive_transform, target, context)
    f_inv = inverse(f)
    proposal = _create_proposal_state(samplingalg.proposal, target, context, x_init, f, rng)
    
    logd_x_init = logdensityof.(target_unevaluated, x_init)
    z_init = f_inv.(x_init) 
    logd_z_init = logdensityof.(MeasureBase.pullback(f, target_unevaluated), z_init)
    
    W = mcmc_weight_type(samplingalg.sample_weighting)
    T = eltype(logd_x_init)
    info_curr, IT = _get_sample_id(proposal, Int32(id), one(Int32), 1, CURRENT_SAMPLE)
    
    current_x_init = DensitySampleVector(x_init, 
                                         logd_x_init;
                                         weight = fill(one(W), n_walkers), 
                                         info = fill(info_curr, n_walkers),
                                         aux = fill(nothing, n_walkers)
                                        )
    current_z_init = DensitySampleVector(z_init, 
                                         logd_z_init;
                                         weight = fill(one(W), n_walkers), 
                                         info = fill(info_curr, n_walkers),
                                         aux = fill(nothing, n_walkers)
                                        )
    dsv_init = similar(current_x_init)

    global gs_cs_init = (current_x_init, current_z_init, dsv_init)

    current = (x = current_x_init, z = current_z_init)
    proposed = (x = deepcopy(dsv_init), z = deepcopy(dsv_init))
    output = deepcopy(dsv_init)
    accepted = fill(false, n_walkers)
    
    stepno::Int64 = 0
    cycle::Int32 = 0
    nsamples::Int64 = 0

    state = MCMCChainState(
        target,
        proposal,
        f,
        samplingalg.sample_weighting,
        current,
        proposed, 
        output, 
        accepted,
        MCMCChainStateInfo(id, cycle, false, false),
        rngpart_cycle,
        nsamples,
        stepno,
        context
    )

    state
end

# @inline _current_sample_idx(chain_state::MCMCChainState) = firstindex(chain_state.samples)
# @inline _proposed_sample_idx(chain_state::MCMCChainState) = lastindex(chain_state.samples)

# @inline _current_sample_z_idx(chain_state::MCMCChainState) = firstindex(chain_state.sample_z)
# @inline _proposed_sample_z_idx(chain_state::MCMCChainState) = lastindex(chain_state.sample_z)

# @inline _current_sample_idx(mcmc_state::MCMCState) = firstindex(mcmc_state.chain_state.samples)
# @inline _proposed_sample_idx(mcmc_state::MCMCState) = lastindex(mcmc_state.chain_state.samples)

# @inline _current_sample_z_idx(mcmc_state::MCMCState) = firstindex(mcmc_state.chain_state.sample_z)
# @inline _proposed_sample_z_idx(mcmc_state::MCMCState) = lastindex(mcmc_state.chain_state.sample_z)


get_proposal(state::MCMCChainState) = state.proposal

mcmc_target(state::MCMCChainState) = state.target

get_context(state::MCMCChainState) = state.context

mcmc_info(state::MCMCChainState) = state.info

nsteps(state::MCMCChainState) = state.stepno

nsamples(state::MCMCChainState) = state.nsamples

current_sample(state::MCMCChainState) = state.current.x

proposed_sample(state::MCMCChainState) = state.proposed.x

current_sample_z(state::MCMCChainState) = state.current.z

proposed_sample_z(state::MCMCChainState) = state.proposed.z

sample_type(state::MCMCChainState) = eltype(state.current.x)


mcmc_target(state::MCMCState) = mcmc_target(state.chain_state)

nsamples(state::MCMCState) = nsamples(state.chain_state)

nsteps(state::MCMCState) = nsteps(state.chain_state)


function DensitySampleVector(state::MCMCState)
    return fill(DensitySampleVector(sample_type(state.chain_state), totalndof(varshape(mcmc_target(state)))), length(state.chain_state.current.x))
end

function DensitySampleVector(chain_state::MCMCChainState)
    return fill(DensitySampleVector(sample_type(chain_state), totalndof(varshape(mcmc_target(chain_state)))), length(chain_state.current.x))
end


function mcmc_step!!(mcmc_state::MCMCState)
    _cleanup_samples(mcmc_state)
    
    reset_rng_counters!(mcmc_state)

    chain_state = mcmc_state.chain_state

    @unpack target, proposal, f_transform, current, proposed, nsamples, context = chain_state
    
    chain_state.stepno += 1

    new_proposed_info, _ = _get_sample_id(proposal, chain_state.info.id, chain_state.info.cycle, chain_state.stepno, PROPOSED_SAMPLE)
    new_proposed_info_vec = fill(new_proposed_info, length(chain_state.current.x.v))

    chain_state.proposed.x.info .= new_proposed_info_vec
    chain_state.proposed.z.info .= new_proposed_info_vec

    chain_state, p_accept = mcmc_propose!!(chain_state)

    # This does not change `sample_z` in the chain_state, that happens in the next mcmc step in `_cleanup_samples()`.
    _accept_reject!(chain_state, p_accept)

    mcmc_state_new = mcmc_tune_post_step!!(mcmc_state, p_accept)

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

# TODO, MD: Think about merging this and accept_reject so that the weight update and sample cleanup happen in the same place. 
function _cleanup_samples(chain_state::MCMCChainState)# 
    #ToChange:
    # * Mark walker points in current and proposed as accepted/rejected, depending
    # on accepted vector
    # * For accepted points, copy current to output and overwrite current
    # * For rejected points, increase weight of current (here or elsewhere) and
    # copy proposed point to output.
    idxs_acc = findall(chain_state.accepted)
    idxs_rej = findall(!, chain_state.accepted)

    # Save the current points to output if the walker accepted their proposed point
    chain_state.output[idxs_acc] = chain_state.current.x[idxs_acc]

    # Copy the proposed points that were accepted to current and mark them as accepted
    chain_state.current.x[idxs_acc] = chain_state.proposed.x[idxs_acc]
    chain_state.current.z[idxs_acc] = chain_state.proposed.z[idxs_acc]
    
    # TODO, MD: Discuss the resetting of the MCMCSampleID. @reset would be elegant, but 
    #           broadcasting it does not seem to be supported; and also @reset messes up 
    #           the types of the other fields. 
    for i in idxs_acc
        old_info = chain_state.current.x.info[i]
        # Set sampletype to ACCEPTED_SAMPLE, leave the rest as is.
        new_info, _ = _get_sample_id(chain_state.proposal, old_info.chainid, old_info.chaincycle, old_info.stepno, ACCEPTED_SAMPLE)

        chain_state.current.x.info[i] = new_info
        chain_state.current.z.info[i] = new_info
    end

    # Save the proposed points that were rejected to output
    chain_state.output[idxs_rej] = chain_state.proposed.x[idxs_rej]

    for i in idxs_rej
        old_info = chain_state.output.info[i]
        new_info, _ = _get_sample_id(chain_state.proposal, old_info.chainid, old_info.chaincycle, old_info.stepno, REJECTED_SAMPLE)

        chain_state.output.info[i] = new_info
    end

    # samples = chain_state.samples
    # current = _current_sample_idx(chain_state)
    # proposed = _proposed_sample_idx(chain_state)
    # sample_z = chain_state.sample_z
    # current_z = _current_sample_z_idx(chain_state)
    # proposed_z = _proposed_sample_z_idx(chain_state)
    # if (current != proposed) && samples.info.sampletype[proposed] == CURRENT_SAMPLE
    #     # Proposal was accepted in the last step
    #     @assert samples.info.sampletype[current] == ACCEPTED_SAMPLE
    #     samples.v[current] .= samples.v[proposed]
    #     samples.logd[current] = samples.logd[proposed]
    #     samples.weight[current] = samples.weight[proposed]
    #     samples.info[current] = samples.info[proposed]

    #     resize!(samples, 1)
    #     # TODO: MD, discuss the usage of sample_z, and if it stays, clean it up and use proper info
    #     sample_z.v[current_z] .= sample_z.v[proposed_z]
    #     sample_z.logd[current_z] = sample_z.logd[proposed_z]
    # end
end

function _cleanup_samples(mcmc_state::MCMCState)
    _cleanup_samples(mcmc_state.chain_state)
end

function next_cycle!(chain_state::MCMCChainState)
    _cleanup_samples(chain_state)
    n_walkers = length(chain_state.current.x)

    chain_state.info = MCMCChainStateInfo(chain_state.info.id, 
                                          chain_state.info.cycle + 1, 
                                          chain_state.info.tuned, 
                                          chain_state.info.converged
                                         )
    chain_state.nsamples = 0
    chain_state.stepno = 0

    new_current_info, _ = _get_sample_id(chain_state.proposal, chain_state.info.id, chain_state.info.cycle, 0, CURRENT_SAMPLE)
    new_current_info_vec = fill(new_current_info, n_walkers)
    chain_state.current.x.info .= new_current_info_vec
    chain_state.current.z.info .= new_current_info_vec
    
    new_proposed_info, _ = _get_sample_id(chain_state.proposal, chain_state.info.id, chain_state.info.cycle, 0, PROPOSED_SAMPLE)
    new_proposed_info_vec = fill(new_proposed_info, n_walkers)
    chain_state.proposed.x.info .= new_proposed_info_vec
    chain_state.proposed.z.info .= new_proposed_info_vec

    reset_rng_counters!(chain_state)

    # i = _proposed_sample_idx(chain_state)
    # @assert chain_state.samples.info[i].sampletype == CURRENT_SAMPLE
    # chain_state.samples.weight[i] = 1

    # chain_state.samples.info[i] = MCMCSampleID(chain_state.info.id, chain_state.info.cycle, chain_state.stepno, CURRENT_SAMPLE)

    chain_state
end

function next_cycle!(state::MCMCState)
    next_cycle!(state.chain_state)
end

# This assumes 'appendable' to be a vector of appendables that respectively hold the samples for each walker
function get_samples!(appendable, chain_state::MCMCChainState, nonzero_weights::Bool)::typeof(appendable)
    if samples_available(chain_state)
        chain_output = chain_state.output

        sample_types = getfield.(chain_output.info, :sampletype)
        for i in eachindex(chain_output)
            st = sample_types[i]
            if (
                (st == ACCEPTED_SAMPLE || st == REJECTED_SAMPLE) &&
                (chain_output.weight[i] > 0 || !nonzero_weights)
            )
                push!(appendable[i], chain_output[i])
            end
        end
    end
    appendable
end

function get_samples!(appendable, mcmc_state::MCMCState, nonzero_weights::Bool)::typeof(appendable)
    get_samples!(appendable, mcmc_state.chain_state, nonzero_weights)
end


function samples_available(chain_state::MCMCChainState)
    return any(getfield.(chain_state.output.info, :sampletype) .== ACCEPTED_SAMPLE)
end

function samples_available(mcmc_state::MCMCState)
    samples_available(mcmc_state.chain_state)
end

function mcmc_update_z_position!!(mcmc_state::MCMCState)
    chain_state_new = mcmc_update_z_position!!(mcmc_state.chain_state)

    mcmc_state_new = @set mcmc_state.chain_state = chain_state_new
    return mcmc_state_new
end

function mcmc_update_z_position!!(mc_state::MCMCChainState)
    f_transform = mc_state.f_transform

    x_current = mc_state.current.x.v
    logd_x_current = mc_state.current.x.logd

    x_proposed = mc_state.proposed.x.v
    logd_x_proposed = mc_state.proposed.x.logd

    trafo_current = with_logabsdet_jacobian.(inverse(f_transform), x_current)
    trafo_proposed = with_logabsdet_jacobian.(inverse(f_transform), x_proposed)

    logd_z_current_new = logd_x_current - getfield.(trafo_current, 2) 
    logd_z_proposed_new = logd_x_proposed - getfield.(trafo_proposed, 2)

    mc_state_new = deepcopy(mc_state)

    mc_state_new.current.z.v .= getfield.(trafo_current, 1)
    mc_state_new.proposed.z.v .= getfield.(trafo_proposed, 1)
    
    mc_state_new.current.z.logd .= logd_z_current_new
    mc_state_new.proposed.z.logd .= logd_z_proposed_new
    
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
    chain_state_tmp, trafo_tuner_state_new = mcmc_tune_post_cycle!!(state.trafo_tuner_state, state.chain_state, samples)
    chain_state_new, proposal_tuner_state_new = mcmc_tune_post_cycle!!(state.proposal_tuner_state, chain_state_tmp, samples)

    mcmc_state_cs = @set state.chain_state = chain_state_new
    mcmc_state_tt = @set mcmc_state_cs.trafo_tuner_state = trafo_tuner_state_new
    mcmc_state_pt = @set mcmc_state_tt.proposal_tuner_state = proposal_tuner_state_new

    return mcmc_state_pt
end

function mcmc_tune_post_step!!(state::MCMCState, p_accept::AbstractVector{<:Real})
    chain_state_tmp, trafo_tuner_state_new = mcmc_tune_post_step!!(state.trafo_tuner_state, state.chain_state, p_accept)
    chain_state_new, proposal_tuner_state_new = mcmc_tune_post_step!!(state.proposal_tuner_state, chain_state_tmp, p_accept)

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
