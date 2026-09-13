# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# TODO: MD, use Accessors.jl to make immutable
"""
    MCMCChainState

*Experimental feature, not part of stable public API.*

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
    GC<:AbstractVector{<:GenContext},
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
    walker_order::Vector{Int}
    walker_genctxs::GC
    info::MCMCChainStateInfo
    rngpart_cycle::PR
    nattempts::Vector{Int64}
    nsamples::Vector{Int64}
    stepno::Int64
    context::CTX
end
export MCMCChainState

_contains_hamiltonian_mc(proposal::MCMCProposal) =
    proposal isa HamiltonianMC ||
    (proposal isa MCMCMultiProposal && any(_contains_hamiltonian_mc, proposal.proposals))

function MCMCChainState(
    samplingalg::TransformedMCMC,
    target::BATMeasure,
    chainid::Integer,
    x_init::AbstractVector{PV},
    context::BATContext
) where {P<:Real, PV<:AbstractVector{P}}
    _validate_mcmc_proposal_configuration(samplingalg.proposal, samplingalg.proposal_tuning)

    n_walkers = length(x_init)
    target_unevaluated = unevaluated(target)

    if _contains_hamiltonian_mc(samplingalg.proposal) && samplingalg.sample_weighting isa ARPWeighting
        throw(ArgumentError(
            "ARPWeighting is not valid for HamiltonianMC: the NUTS acceptance statistic is a trajectory average, not the selection probability of the returned state"
        ))
    end

    rngpart_cycle = RNGPartition(get_rng(context), 0:(typemax(Int16) - 2))
    rng = get_rng(context)

    f = init_adaptive_transform(samplingalg.adaptive_transform, target, x_init, context)
    f_inv = inverse(f)
    proposal = _create_proposal_state(samplingalg.proposal, target_unevaluated, context, x_init, f, rng)

    logd_x_init = BAT.checked_logdensityof.(target_unevaluated, x_init)
    z_init = f_inv.(x_init)
    ladj_c = _transform_ladj(f)
    logd_z_init = isnothing(ladj_c) ?
        logdensityof.(MeasureBase.pullback(f, target_unevaluated), z_init) :
        logd_x_init .+ ladj_c

    W = mcmc_weight_type(samplingalg.sample_weighting)

    sample_weights_curr = zeros(W, n_walkers)
    sample_info_curr = MCMCSampleID[MCMCSampleID(Int32(chainid), Int32(i), one(Int32), zero(Int64), get_active_proposal_idx(proposal), true) for i in 1:n_walkers]
    sample_aux_curr = fill(nothing, n_walkers)

    current_x_init = DensitySampleVector(
        v = x_init,
        logd = logd_x_init,
        weight = sample_weights_curr,
        info = sample_info_curr,
        aux = sample_aux_curr
    )
    current_z_init = DensitySampleVector(
        v = z_init,
        logd = logd_z_init,
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
        v = prop_locs_init,
        logd = prop_logds_init,
        weight = sample_weights_prop,
        info = sample_info_prop,
        aux = sample_aux_prop
    )

    current = (x = current_x_init, z = current_z_init)
    proposed = (x = deepcopy(proposed_init), z = deepcopy(proposed_init))
    output = deepcopy(current_x_init)
    accepted = fill(false, n_walkers)
    walker_order = sortperm(
        eachindex(sample_info_curr); by = i -> sample_info_curr[i].walkerid,
    )
    walker_genctxs = map(1:n_walkers) do _
        get_gencontext(set_rng(context, rngpart_createrng(typeof(rng))))
    end

    stepno::Int64 = 0
    cycle::Int32 = 0

    n_proposals = proposal isa MultiProposalState ? length(proposal.proposal_states) : 1
    nattempts = zeros(Int64, n_proposals)
    nsamples::Vector{Int64} = zeros(n_proposals)

    # The stored target is evaluated in the sampling hot loop, so it must be
    # a bare measure. Knowledge attached to the given target (samples,
    # approximations, see EvaluatedMeasure) is consumed upstream by the
    # initial-value generation; the adaptive transform initialization
    # currently sees only the initial positions:
    state = MCMCChainState(
        target_unevaluated,
        proposal,
        f,
        samplingalg.sample_weighting,
        current,
        proposed, 
        output, 
        accepted,
        walker_order,
        walker_genctxs,
        MCMCChainStateInfo(chainid, cycle, false, false),
        rngpart_cycle,
        nattempts,
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
    # One independent output vector per walker - fill would alias a single object:
    return [_empty_DensitySampleVector(chain_state) for _ in 1:nwalkers(chain_state)]
end

function eff_acceptance_ratio(chain_state::MCMCChainState)
    active_proposal = get_active_proposal(chain_state.proposal)
    return nsamples(chain_state) / (nsteps(chain_state) * nwalkers(chain_state))
end

detailed_eff_acceptance_ratio(state::MCMCChainState) = state.nsamples ./ state.nattempts


function mcmc_step!!(mcmc_state::MCMCState)
    reset_rng_counters!(mcmc_state)

    chain_state = mcmc_state.chain_state
    chain_state.stepno += 1

    (;proposal, stepno, context) = chain_state

    rng = get_rng(context)
    n_rng_streams = _MCMC_N_RNG_PURPOSES * _MCMC_PROPOSALS_PER_PURPOSE
    step_rngpart = RNGPartition(rng, Base.OneTo(n_rng_streams))
    selection_idx = _mcmc_rng_stream_idx(_MCMC_PROPOSAL_SELECTION_PURPOSE, 1)
    proposal_selection_rng = AbstractRNG(step_rngpart, selection_idx)

    chain_state.proposal, active_proposal = next_proposal!!(
        proposal_selection_rng, proposal, stepno,
    )

    proposal_idx = get_active_proposal_idx(chain_state.proposal)
    chain_state, active_proposal_new, step_info = mcmc_propose!!(
        chain_state, active_proposal, step_rngpart, proposal_idx)

    chain_state.proposal = update_active_proposal!!(chain_state.proposal, active_proposal_new)

    mcmc_state_new = mcmc_tune_post_step!!(mcmc_state, active_proposal, step_info)

    chain_state = mcmc_state_new.chain_state

    (;proposal, current, proposed, accepted, output) = chain_state

    active_prop_idx = get_active_proposal_idx(proposal)
    chain_state.nattempts[active_prop_idx] += length(accepted)
    chain_state.nsamples[active_prop_idx] += sum(accepted)

    # Set weights according to acceptance
    delta_w_current, w_proposed = mcmc_weight_values(chain_state.weighting, step_info.p_accept, accepted)

    current.x.weight .+= delta_w_current
    current.z.weight .+= delta_w_current

    proposed.x.weight .= w_proposed
    proposed.z.weight .= w_proposed

    idxs_acc = findall(accepted)
    idxs_rej = findall(!, accepted)

    # For each walker, mark proposed samples as accepted or rejected
    for i in eachindex(proposed.x)
        old_info = current.x.info[i]

        sample_type = accepted[i]
        new_info = MCMCSampleID(
            old_info.chainid, 
            old_info.walkerid,
            old_info.chaincycle, 
            chain_state.stepno, 
            get_active_proposal_idx(proposal), 
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

# Log-abs-det Jacobian of a space transformation, if it is constant
# (nothing otherwise). `false` is the type-neutral additive zero:
_transform_ladj(::Any) = nothing
_transform_ladj(::typeof(identity)) = false
_transform_ladj(f::MulAdd) = _mul_factor_ladj(f.A)

# Only for matrix-shaped factors is logabsdet the per-variate ladj; scalar
# factors act on every dimension, their ladj depends on the variate length,
# so they take the generic with_logabsdet_jacobian path:
_mul_factor_ladj(A) = ndims(A) == 2 ? first(logabsdet(A)) : nothing
_mul_factor_ladj(::Real) = nothing
_mul_factor_ladj(::UniformScaling) = nothing

# Batched transform application together with per-element LADJs, with a
# single logabsdet evaluation for transforms with constant Jacobian:
function _transform_with_ladj(f, zs::AbstractVector)
    c = _transform_ladj(f)
    if isnothing(c)
        ys_ladjs = with_logabsdet_jacobian.(f, zs)
        return first.(ys_ladjs), getsecond.(ys_ladjs)
    else
        return f.(zs), fill(c, length(eachindex(zs)))
    end
end


function mcmc_propose!!(chain_state::MCMCChainState, proposal::SMP,
    step_rngpart::RNGPartition, proposal_idx::Integer) where {SMP<:SimpleMCMCProposalState}
    (; target, f_transform, current) = chain_state

    current_z = current.z.v
    logd_z_current = current.z.logd

    walker_info = current.x.info
    proposal_rngpart = _mcmc_walker_rngpart(
        step_rngpart, _MCMC_PROPOSAL_TRANSITION_PURPOSE, proposal_idx)
    genctxs = chain_state.walker_genctxs
    foreach(genctxs, walker_info) do genctx, info
        set_rng!(get_rng(genctx), proposal_rngpart, info.walkerid)
    end

    # TODO: MD; Make this function ! because it alters genctx?
    z_proposed, hastings_correction = mcmc_propose_transition(current_z, proposal, genctxs)

    x_proposed, ladj = _transform_with_ladj(f_transform, z_proposed)

    logd_x_proposed = BAT.checked_logdensityof.(target, x_proposed)
    logd_z_proposed::typeof(logd_x_proposed) = logd_x_proposed .+ ladj

    chain_state.proposed.x.v .= x_proposed
    chain_state.proposed.z.v .= z_proposed

    chain_state.proposed.x.logd .= logd_x_proposed
    chain_state.proposed.z.logd .= logd_z_proposed

    log_accept_ratio = logd_z_proposed - logd_z_current + hastings_correction
    p_accept = @. ifelse(isnan(log_accept_ratio), zero(log_accept_ratio), clamp(exp(log_accept_ratio), 0, 1))
    acceptance_rngpart = _mcmc_walker_rngpart(
        step_rngpart, _MCMC_ACCEPTANCE_PURPOSE, proposal_idx)
    accepted = map(eachindex(p_accept)) do i
        rng = get_rng(genctxs[i])
        set_rng!(rng, acceptance_rngpart, walker_info[i].walkerid)
        rand(rng) < p_accept[i]
    end

    chain_state.accepted .= accepted

    step_info = MCMCStepInfo(
        p_accept, _selected_z_grads(proposal, accepted), nothing, nothing, nothing,
        chain_state.walker_order,
    )
    return chain_state, proposal, step_info
end

const _MCMC_PROPOSAL_SELECTION_PURPOSE = 1
const _MCMC_PROPOSAL_TRANSITION_PURPOSE = 2
const _MCMC_ACCEPTANCE_PURPOSE = 3
const _MCMC_N_RNG_PURPOSES = 3
# Purpose blocks are fixed-width so their stream indices cannot overlap;
# multi-proposal construction and the index helper both enforce this limit.
const _MCMC_PROPOSALS_PER_PURPOSE = typemax(Int16) - 2

@inline function _mcmc_rng_stream_idx(purpose::Integer, proposal_idx::Integer)
    1 <= purpose <= _MCMC_N_RNG_PURPOSES || throw(ArgumentError(
        "MCMC RNG purpose must be between 1 and $_MCMC_N_RNG_PURPOSES, got $purpose"))
    1 <= proposal_idx <= _MCMC_PROPOSALS_PER_PURPOSE || throw(ArgumentError(
        "MCMC proposal index must be between 1 and $_MCMC_PROPOSALS_PER_PURPOSE, got $proposal_idx"))
    return (purpose - 1) * _MCMC_PROPOSALS_PER_PURPOSE + proposal_idx
end

function _mcmc_walker_rngpart(
    step_rngpart::RNGPartition, purpose::Integer, proposal_idx::Integer)
    stream_rng = AbstractRNG(step_rngpart, _mcmc_rng_stream_idx(purpose, proposal_idx))
    return RNGPartition(stream_rng, Base.OneTo(typemax(Int32) - 2))
end

_logical_walker_order(chain_state::MCMCChainState) = chain_state.walker_order

function _ordered_walker_sum(values, walker_order)
    result = zero(eltype(values))
    # Fixed order preserves reproducible sums.
    @inbounds for i in walker_order
        result += values[i]
    end
    return result
end

function _ordered_walker_sum_and_sqsum(values, walker_order)
    value_sum = zero(eltype(values))
    value_sqsum = zero(eltype(values))
    # Fixed order preserves reproducible sums.
    @inbounds for i in walker_order
        value = values[i]
        value_sum += value
        value_sqsum += abs2(value)
    end
    return value_sum, value_sqsum
end

# Proposals that compute z-space log-density gradients report the
# gradients at the selected (post-accept/reject) states, others nothing:
_selected_z_grads(::MCMCProposalState, ::AbstractVector{Bool}) = nothing

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
    walker_info = chain_state.current.x.info

    chain_state.info = MCMCChainStateInfo(chain_state.info.id,
                                          chain_state.info.cycle + 1,
                                          chain_state.info.tuned,
                                          chain_state.info.converged
                                          )
    chain_state.nattempts .= 0
    chain_state.nsamples .= 0
    chain_state.stepno = 0

    proposal = chain_state.proposal
    info = chain_state.info

    new_current_info_vec = [MCMCSampleID(
        info.id,
        walker_info[i].walkerid,
        info.cycle,
        zero(Int64),
        get_active_proposal_idx(proposal),
        true
        ) for i in 1:n_walkers]
    chain_state.current.x.info .= new_current_info_vec
    chain_state.current.z.info .= new_current_info_vec

    new_proposed_info_vec = [MCMCSampleID(
        info.id,
        walker_info[i].walkerid,
        info.cycle,
        zero(Int64),
        get_active_proposal_idx(proposal),
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
    # x- and z-side weights must stay in sync:
    current.x.weight .= 0
    current.z.weight .= 0

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

    current_z_new = _transform_dsv!!(f_inv, mc_state.current.z, mc_state.current.x)
    proposed_z_new = _transform_dsv!!(f_inv, mc_state.proposed.z, mc_state.proposed.x)

    mc_state_new::typeof(mc_state) = @set mc_state.current.z = current_z_new
    mc_state_new = @set mc_state_new.proposed.z = proposed_z_new

    return mc_state_new
end

# TODO: MD, Discuss:
# When using different Tuners for proposal and transformation, which should be applied first? 
# And if the z-position changes during the transformation tuning, should the proposal Tuner run on the updated z-position?
function mcmc_tuning_init!!(state::MCMCState, max_nsteps::Integer)
    # TODO: mcmc_tuning_init!! should support immutable tuners and states and return the new objects
    mcmc_trafo_tuning_init!!(state.trafo_tuner_state, state.chain_state, max_nsteps)
    mcmc_proposal_tuning_init!!(state.proposal_tuner_state, state.chain_state, max_nsteps)
end

function mcmc_tuning_reinit!!(state::MCMCState, max_nsteps::Integer)
    # TODO: mcmc_tuning_reinit!! should support immutable tuners and states and return the new objects
    mcmc_trafo_tuning_reinit!!(state.trafo_tuner_state, state.chain_state, max_nsteps)
    mcmc_proposal_tuning_reinit!!(state.proposal_tuner_state, state.chain_state, max_nsteps)
end

function mcmc_tuning_postinit!!(state::MCMCState, samples::AbstractVector{<:DensitySampleVector})
    # TODO: mcmc_tuning_postinit!! should support immutable tuners and states and return the new objects
    mcmc_trafo_tuning_postinit!!(state.trafo_tuner_state, state.chain_state, samples)
    mcmc_proposal_tuning_postinit!!(state.proposal_tuner_state, state.chain_state, samples)
end

# TODO: MD, when should the z-position be updated? Before or after the proposal tuning?
function mcmc_tune_post_cycle!!(state::MCMCState, samples::AbstractVector{<:DensitySampleVector})
    f_transform_tuned, trafo_tuner_state_new, chain_state_trafo_tuned = mcmc_tune_trafo_post_cycle!!(
        state.chain_state.f_transform,
        state.trafo_tuner_state,
        state.chain_state,
        state.chain_state.proposal,
        samples
    )

    # Only an actual transform change requires the (comparatively expensive)
    # proposal-target rebuild and z-position remap; transform tuners signal
    # "no change" by returning the identical transform object:
    if f_transform_tuned !== chain_state_trafo_tuned.f_transform
        chain_state_trafo_tuned = @set chain_state_trafo_tuned.f_transform = f_transform_tuned
        proposal = chain_state_trafo_tuned.proposal
        proposal = set_proposal_transform!!(proposal, chain_state_trafo_tuned)
        chain_state_trafo_tuned = mcmc_update_z_position!!(chain_state_trafo_tuned)
        if transform_change_restarts_stepsize(trafo_tuner_state_new)
            proposal, _, chain_state_trafo_tuned = mcmc_proposal_transform_committed!!(
                proposal, state.proposal_tuner_state, chain_state_trafo_tuned
            )
        end
    else
        proposal = chain_state_trafo_tuned.proposal
    end

    proposal_state_new, proposal_tuner_state_new, chain_state_new = mcmc_tune_proposal_post_cycle!!(
        proposal,
        state.proposal_tuner_state,
        chain_state_trafo_tuned,
        samples
    )

    α = eff_acceptance_ratio(chain_state_new)

    logds = [walker_smpls.logd for walker_smpls in samples]
    max_log_posterior = maximum(maximum(logds))

    tuning_success = get_tuning_success(chain_state_new, proposal_state_new, proposal_tuner_state_new)

    if tuning_success
        chain_state_new.info = MCMCChainStateInfo(chain_state_new.info, tuned = true)
        @debug "MCMC chain $(chain_state_new.info.id) tuned, acceptance ratio = $(Float32(α)), max. log posterior = $(Float32(max_log_posterior))"
    else
        chain_state_new.info = MCMCChainStateInfo(chain_state_new.info, tuned = false)
        @debug "MCMC chain $(chain_state_new.info.id) *not* tuned, acceptance ratio = $(Float32(α)), max. log posterior = $(Float32(max_log_posterior))"
    end

    chain_state_final = @set chain_state_new.proposal = proposal_state_new

    mcmc_state_cs = @set state.chain_state = chain_state_final
    mcmc_state_tt = @set mcmc_state_cs.trafo_tuner_state = trafo_tuner_state_new
    mcmc_state_pt = @set mcmc_state_tt.proposal_tuner_state = proposal_tuner_state_new

    return mcmc_state_pt
end

function mcmc_tune_post_step!!(state::MCMCState, proposal::MCMCProposalState, step_info::MCMCStepInfo)
    proposal_tuning_was_paused =
        transform_tuning_pauses_proposal(state.trafo_tuner_state)
    f_transform_tuned, trafo_tuner_state_new, chain_state_trafo_tuned = mcmc_tune_trafo_post_step!!(
        state.chain_state.f_transform,
        state.trafo_tuner_state,
        state.chain_state,
        proposal,
        state.chain_state.current,
        state.chain_state.proposed,
        step_info
    )

    # Only an actual transform change requires the (comparatively expensive)
    # proposal-target rebuild and z-position remap; transform tuners signal
    # "no change" by returning the identical transform object:
    stepsize_restart = false
    if f_transform_tuned !== chain_state_trafo_tuned.f_transform
        chain_state_trafo_tuned = @set chain_state_trafo_tuned.f_transform = f_transform_tuned
        proposal = chain_state_trafo_tuned.proposal
        proposal = set_proposal_transform!!(proposal, chain_state_trafo_tuned)
        chain_state_trafo_tuned = mcmc_update_z_position!!(chain_state_trafo_tuned)
        stepsize_restart = transform_change_restarts_stepsize(trafo_tuner_state_new)
    else
        proposal = chain_state_trafo_tuned.proposal
    end

    proposal_state_new, proposal_tuner_state_new, chain_state_new = if stepsize_restart
        # The step statistic was generated under the old geometry, discard it:
        mcmc_proposal_transform_committed!!(
            proposal,
            state.proposal_tuner_state,
            chain_state_trafo_tuned,
            trafo_tuner_state_new,
        )
    elseif proposal_tuning_was_paused ||
            transform_tuning_pauses_proposal(trafo_tuner_state_new)
        proposal, state.proposal_tuner_state, chain_state_trafo_tuned
    else
        mcmc_tune_proposal_post_step!!(
            proposal,
            state.proposal_tuner_state,
            chain_state_trafo_tuned,
            step_info
        )
    end

    chain_state_final = @set chain_state_new.proposal = proposal_state_new

    # TODO: MD, inelegant, use AccessorsExtra.jl to set several fields at once? https://github.com/JuliaAPlavin/AccessorsExtra.jl
    mcmc_state_cs = @set state.chain_state = chain_state_final
    mcmc_state_tt = @set mcmc_state_cs.trafo_tuner_state = trafo_tuner_state_new
    mcmc_state_pt = @set mcmc_state_tt.proposal_tuner_state = proposal_tuner_state_new

    return mcmc_state_pt
end

function mcmc_tuning_finalize!!(mcmc_state::MCMCState)
    f_old = mcmc_state.chain_state.f_transform
    f_new, trafo_tuner_state_new, chain_state_new = mcmc_trafo_tuning_finalize!!(
        f_old,
        mcmc_state.trafo_tuner_state,
        mcmc_state.chain_state
    )
    # A finalizer may rebuild the transform object, but it must represent
    # the same map of the same type: finalization runs none of the
    # transform-commit synchronization (no z-remap, no proposal-target
    # rebind, no step-size restart), so geometry changes are only valid
    # through the commit path during tuning:
    typeof(f_new) === typeof(f_old) || throw(ErrorException(
        "Transform tuner finalization must not change the transform type ($(typeof(f_old)) -> $(typeof(f_new))), geometry changes must go through the transform-commit path during tuning"
    ))
    @reset chain_state_new.f_transform = f_new

    proposal_new, proposal_tuner_state_new, chain_state_final = mcmc_proposal_tuning_finalize!!(
        chain_state_new.proposal,
        mcmc_state.proposal_tuner_state,
        chain_state_new
    )

    @reset chain_state_final.proposal = proposal_new

    # Tuning ends here for good: freezing the tuner states makes
    # post-tuning stabilization and retained sampling run a fixed kernel,
    # the transform and proposal parameters no longer adapt:
    @reset mcmc_state.trafo_tuner_state = FrozenMCMCTransformTunerState()
    @reset mcmc_state.proposal_tuner_state = FrozenMCMCProposalTunerState()
    @reset mcmc_state.chain_state = chain_state_final

    return mcmc_state
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
