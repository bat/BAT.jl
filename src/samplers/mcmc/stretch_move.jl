# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct StretchMove <: MCMCProposal

Experimental affine-invariant ensemble proposal using the Goodman--Weare
stretch move. `nwalkers` must be set explicitly on `TransformedMCMC`.

Constructors:

* ```$(FUNCTIONNAME)(; scale = 2, executor = BAT.SequentialExec())```
"""
struct StretchMove{S<:Real,E<:BATExecutor} <: AbstractEnsembleProposal
    scale::S
    executor::E

    function StretchMove(scale::S, executor::E) where {S<:Real,E<:BATExecutor}
        isfinite(scale) && scale > one(scale) || throw(ArgumentError(
            "StretchMove scale must be finite and greater than 1",
        ))
        _validate_ensemble_executor(executor)
        new{S,E}(scale, executor)
    end
end
StretchMove(scale::Real) = StretchMove(scale, SequentialExec())
StretchMove(; scale::Real = 2, executor::BATExecutor = SequentialExec()) =
    StretchMove(scale, executor)
export StretchMove


struct StretchMoveProposalState{S<:Real,E<:BATExecutor} <: AbstractEnsembleMove
    scale::S
    executor::E
end

_ensemble_group_count(::StretchMoveProposalState) = 2
_ensemble_minimum_walkers(::StretchMoveProposalState, n_dims::Integer) = 2 * n_dims

function _create_proposal_state(
    proposal::StretchMove,
    ::BATMeasure,
    ::BATContext,
    ::AbstractVector,
    z_init::AbstractVector{PV},
    ::Function,
    ::AbstractRNG,
) where {P<:Real, PV<:AbstractVector{P}}
    scale_type = float(eltype(first(z_init)))
    scale = convert(scale_type, proposal.scale)
    isfinite(scale) && scale > one(scale) || throw(ArgumentError(
        "StretchMove scale must remain finite and greater than 1 after conversion to $(scale_type)",
    ))

    return StretchMoveProposalState(scale, proposal.executor)
end


function _stretch_scale(scale::Real, u::Real)
    b = (scale - one(scale)) * u + one(scale)
    return b * (b / scale)
end

function _stretch_candidate!!(candidate, current, companion, scale::Real)
    @. candidate = companion + scale * (current - companion)
    return candidate
end

_stretch_log_acceptance(
    n_dims::Integer, scale::Real, proposed_logd::Real, current_logd::Real,
) = (n_dims - 1) * log(scale) + proposed_logd - current_logd


@inline function _propose_ensemble_candidate!!(
    candidate,
    proposal::StretchMoveProposalState,
    current,
    walker_idx::Integer,
    complement_groups,
    rng::AbstractRNG,
    step_rngpart::RNGPartition,
    proposal_idx::Integer,
    walkerid::Integer,
)
    companion_idxs = only(complement_groups)
    companion_rngpart = _mcmc_walker_rngpart(
        step_rngpart, _MCMC_COMPANION_SELECTION_PURPOSE, proposal_idx,
    )
    stretch_rngpart = _mcmc_walker_rngpart(
        step_rngpart, _MCMC_STRETCH_DRAW_PURPOSE, proposal_idx,
    )
    return _propose_ensemble_candidate!!(
        candidate, proposal, current, walker_idx, companion_idxs, rng,
        companion_rngpart, stretch_rngpart, walkerid,
    )
end

@inline function _propose_ensemble_candidate!!(
    candidate,
    proposal::StretchMoveProposalState,
    current,
    walker_idx::Integer,
    companion_idxs::AbstractVector{<:Integer},
    rng::AbstractRNG,
    companion_rngpart::RNGPartition,
    stretch_rngpart::RNGPartition,
    walkerid::Integer,
)
    set_rng!(rng, companion_rngpart, walkerid)
    companion_idx = rand(rng, companion_idxs)

    set_rng!(rng, stretch_rngpart, walkerid)
    T = float(eltype(current[walker_idx]))
    stretch = _stretch_scale(proposal.scale, rand(rng, T))
    _stretch_candidate!!(
        candidate, current[walker_idx], current[companion_idx], stretch,
    )
    return stretch
end

function _propose_ensemble_group!!(
    chain_state::MCMCChainState,
    proposal::StretchMoveProposalState{<:Any,<:SequentialExec},
    step_rngpart::RNGPartition,
    proposal_idx::Integer,
    active_group::AbstractVector{<:Integer},
    complement_groups,
    p_accept::AbstractVector{<:Real},
    constant_ladj,
    acceptance_rngpart::RNGPartition,
)
    current = chain_state.current
    companion_idxs = only(complement_groups)
    companion_rngpart = _mcmc_walker_rngpart(
        step_rngpart, _MCMC_COMPANION_SELECTION_PURPOSE, proposal_idx,
    )
    stretch_rngpart = _mcmc_walker_rngpart(
        step_rngpart, _MCMC_STRETCH_DRAW_PURPOSE, proposal_idx,
    )

    for walker_idx in active_group
        walkerid = current.x.info[walker_idx].walkerid
        rng = get_rng(chain_state.walker_genctxs[walker_idx])
        current_z = current.z.v[walker_idx]
        proposed_z = chain_state.proposed.z.v[walker_idx]
        proposal_aux = _propose_ensemble_candidate!!(
            proposed_z, proposal, current.z.v,
            walker_idx, companion_idxs, rng,
            companion_rngpart, stretch_rngpart, walkerid,
        )
        _evaluate_ensemble_walker!!(
            chain_state, proposal, walker_idx, current_z, proposed_z, proposal_aux, p_accept,
            constant_ladj, acceptance_rngpart, rng, walkerid,
        )
    end

    return chain_state
end

function _ensemble_log_hastings(
    ::StretchMoveProposalState,
    current,
    proposed,
    stretch::Real,
)
    return _stretch_log_acceptance(length(proposed), stretch, false, false)
end
