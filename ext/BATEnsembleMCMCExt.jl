# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATEnsembleMCMCExt

using MeasureBase: pullback
using Random: AbstractRNG
import BAT
import EnsembleMCMC

BAT.pkgext(::Val{:EnsembleMCMC}) = BAT.PackageExtension{:EnsembleMCMC}()

mutable struct EnsembleProposalState{S,H} <: BAT.AbstractEnsembleMove
    ensemble::S
    dirty::Bool
    transitions::H
end

function BAT._create_proposal_state(
    proposal::BAT.EnsembleProposal, target::BAT.BATMeasure, context::BAT.BATContext,
    v_init::AbstractVector, z_init::AbstractVector, f_transform::Function, rng::AbstractRNG,
)
    return BAT._create_proposal_state(
        proposal, target, context, v_init, z_init, nothing, f_transform, rng,
    )
end

function BAT._create_proposal_state(
    proposal::BAT.EnsembleProposal, target::BAT.BATMeasure, context::BAT.BATContext,
    v_init::AbstractVector, z_init::AbstractVector,
    logd_z_init::Union{Nothing,AbstractVector}, f_transform::Function, rng::AbstractRNG,
)
    density = pullback(f_transform, target)
    executor = isnothing(proposal.executor) ? EnsembleMCMC.SerialExecutor() : proposal.executor
    scalar = z -> BAT.checked_logdensityof(density, z)
    logdensity = isnothing(proposal.batch_logdensity!) ? scalar :
        EnsembleMCMC.BatchedLogDensity(scalar,
            (values, positions) -> proposal.batch_logdensity!(values, density, positions))
    ensemble = EnsembleMCMC.initialize(rng, logdensity, z_init;
        move=proposal.move, executor, logdensities=logd_z_init)
    state = EnsembleMCMC.current_state(ensemble)
    transitions = proposal.record_transitions ?
        typeof(_transition_record(Int32(0), Int64(0), state))[] : nothing
    return EnsembleProposalState(ensemble, false, transitions)
end

_transition_record(cycle, step, state) = (; cycle, step, move_index=state.move_index,
    accepted=copy(state.accepted), acceptance_probabilities=copy(state.acceptance_probabilities))
_record_transition!(::Nothing, chain_state, transition) = nothing
function _record_transition!(records::Vector, chain_state, transition)
    push!(records, _transition_record(chain_state.info.cycle, chain_state.stepno, transition))
    return nothing
end

function BAT._invalidate_position_cache!!(proposal::EnsembleProposalState)
    proposal.dirty = true
    return nothing
end

function BAT._validate_mcmc_ensemble_invariants(
    proposal::EnsembleProposalState, ::BAT.BATMeasure, z_init::AbstractVector,
)
    # Package initialization already validates rank. BAT owns retry rerolls.
    proposal.dirty || return nothing
    return EnsembleMCMC.validate_positions(z_init)
end

function BAT.mcmc_propose!!(
    chain_state::BAT.MCMCChainState, proposal::EnsembleProposalState,
    ::BAT.RNGPartition, proposal_index::Integer,
)
    if proposal.dirty
        EnsembleMCMC.synchronize!(proposal.ensemble, chain_state.current.z.v, chain_state.current.z.logd)
        proposal.dirty = false
    end
    # Reconstruct the cycle/step address before BAT reserved its purpose streams.
    rng = AbstractRNG(chain_state.rngpart_cycle, chain_state.info.cycle)
    steps = BAT.RNGPartition(rng, 0:(typemax(Int32) - 2))
    BAT.set_rng!(rng, steps, chain_state.stepno - 1)
    EnsembleMCMC.step!(proposal.ensemble, rng; proposal_index)
    transition = EnsembleMCMC.current_state(proposal.ensemble)
    proposed = chain_state.proposed
    for i in eachindex(transition.candidates)
        copyto!(proposed.z.v[i], transition.candidates[i])
    end
    proposed.z.logd .= transition.candidate_logdensities
    x, ladj = BAT._transform_with_ladj(chain_state.f_transform, proposed.z.v)
    proposed.x.v .= x
    proposed.x.logd .= proposed.z.logd .- ladj
    chain_state.accepted .= transition.accepted
    step_info = BAT.MCMCStepInfo(transition.acceptance_probabilities,
        nothing, nothing, nothing, nothing, chain_state.walker_order)
    _record_transition!(proposal.transitions, chain_state, transition)
    return chain_state, proposal, step_info
end

function BAT._proposal_diagnostics(proposal::EnsembleProposalState)
    state = EnsembleMCMC.current_state(proposal.ensemble)
    counts = (; cumulative_move_attempts=copy(state.attempts),
        cumulative_move_acceptances=copy(state.acceptances))
    isnothing(proposal.transitions) && return counts
    return (; counts..., walker_ids=copy(state.walker_ids), transitions=deepcopy(proposal.transitions))
end

function BAT._proposal_diagnostics(proposal::EnsembleProposalState, chain_state::BAT.MCMCChainState)
    attempts = only(chain_state.nattempts)
    accepted = only(chain_state.nsamples)
    return (; cycle_n_attempts=attempts, cycle_n_accepted=accepted,
        cycle_acceptance_rate=iszero(attempts) ? NaN : accepted / attempts,
        BAT._proposal_diagnostics(proposal)...)
end

end # module BATEnsembleMCMCExt
