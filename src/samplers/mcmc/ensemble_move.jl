# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractEnsembleMove <: MCMCProposalState end

_contains_ensemble_move(::AbstractEnsembleMove) = true


_validate_ensemble_executor(::Union{SequentialExec,MultiThreadedExec}) = nothing

function _validate_ensemble_executor(executor::BATExecutor)
    throw(ArgumentError(
        "Ensemble move executor must be SequentialExec or MultiThreadedExec, got $(nameof(typeof(executor)))",
    ))
end


function _ensemble_move_groups(
    proposal::AbstractEnsembleMove,
    step_rngpart::RNGPartition,
    proposal_idx::Integer,
    walker_order::AbstractVector{<:Integer},
    ::Val{N},
) where {N}
    split_rng = AbstractRNG(
        step_rngpart,
        _mcmc_rng_stream_idx(_MCMC_ENSEMBLE_SPLIT_PURPOSE, proposal_idx),
    )
    permutation = randperm(split_rng, length(walker_order))
    group_size, n_larger_groups = divrem(length(walker_order), N)
    groups = Vector{Vector{eltype(walker_order)}}(undef, N)
    start = firstindex(permutation)
    for group_idx in eachindex(groups)
        stop = start + group_size - 1 + (group_idx <= n_larger_groups)
        groups[group_idx] = walker_order[permutation[start:stop]]
        start = stop + 1
    end
    shuffle!(split_rng, groups)
    return groups
end

function _ensemble_move_groups(
    ::AbstractEnsembleMove,
    step_rngpart::RNGPartition,
    proposal_idx::Integer,
    walker_order::AbstractVector{<:Integer},
    ::Val{2},
)
    split_rng = AbstractRNG(
        step_rngpart,
        _mcmc_rng_stream_idx(_MCMC_ENSEMBLE_SPLIT_PURPOSE, proposal_idx),
    )
    permutation = randperm(split_rng, length(walker_order))
    split_at = fld(length(walker_order), 2)
    left = walker_order[permutation[begin:split_at]]
    right = walker_order[permutation[(split_at + 1):end]]
    return rand(split_rng, Bool) ? (left, right) : (right, left)
end

_ensemble_move_groups(
    proposal::AbstractEnsembleMove,
    step_rngpart::RNGPartition,
    proposal_idx::Integer,
    walker_order::AbstractVector{<:Integer},
) = _ensemble_move_groups(
    proposal, step_rngpart, proposal_idx, walker_order,
    Val(_ensemble_group_count(proposal)),
)

_ensemble_complement_groups(groups::Tuple, active_idx::Integer) =
    active_idx == 1 ? (groups[2],) : (groups[1],)

_ensemble_complement_groups(groups::AbstractVector, active_idx::Integer) =
    groups[eachindex(groups) .!= active_idx]


_ensemble_proposal_is_valid(::AbstractEnsembleMove, proposal_aux) = true


function _evaluate_ensemble_walker!!(
    chain_state::MCMCChainState,
    proposal::AbstractEnsembleMove,
    step_rngpart::RNGPartition,
    proposal_idx::Integer,
    walker_idx::Integer,
    complement_groups,
    p_accept::AbstractVector{<:Real},
    constant_ladj,
    acceptance_rngpart::RNGPartition,
)
    (;target, f_transform, current, proposed) = chain_state
    walkerid = current.x.info[walker_idx].walkerid
    rng = get_rng(chain_state.walker_genctxs[walker_idx])
    proposal_aux = _propose_ensemble_candidate!!(
        proposed.z.v[walker_idx], proposal, current.z.v, walker_idx,
        complement_groups, rng, step_rngpart, proposal_idx, walkerid,
    )

    if !_ensemble_proposal_is_valid(proposal, proposal_aux)
        proposed.x[walker_idx] = current.x[walker_idx]
        proposed.z[walker_idx] = current.z[walker_idx]
        p_accept[walker_idx] = zero(eltype(p_accept))
        chain_state.accepted[walker_idx] = false
        return nothing
    end

    x_proposed, ladj = if isnothing(constant_ladj)
        with_logabsdet_jacobian(f_transform, proposed.z.v[walker_idx])
    else
        f_transform(proposed.z.v[walker_idx]), constant_ladj
    end
    logd_x_proposed = BAT.checked_logdensityof(target, x_proposed)
    logd_z_proposed = logd_x_proposed + ladj

    proposed.x.v[walker_idx] .= x_proposed
    proposed.x.logd[walker_idx] = logd_x_proposed
    proposed.z.logd[walker_idx] = logd_z_proposed

    T = float(eltype(proposed.z.v[walker_idx]))
    log_hastings = _ensemble_log_hastings(
        proposal, current.z.v[walker_idx], proposed.z.v[walker_idx], proposal_aux,
    )
    log_acceptance = log_hastings + logd_z_proposed - current.z.logd[walker_idx]
    p_accept[walker_idx] = _mcmc_acceptance_probability(convert(T, log_acceptance))

    set_rng!(rng, acceptance_rngpart, walkerid)
    chain_state.accepted[walker_idx] = rand(rng, T) < p_accept[walker_idx]
    return nothing
end


function _propose_ensemble_group!!(
    chain_state::MCMCChainState,
    proposal::AbstractEnsembleMove,
    step_rngpart::RNGPartition,
    proposal_idx::Integer,
    active_group::AbstractVector{<:Integer},
    complement_groups,
    p_accept::AbstractVector{<:Real},
    constant_ladj,
    acceptance_rngpart::RNGPartition,
)
    results = Vector{Nothing}(undef, length(active_group))
    exec_map!(proposal.executor, results, active_group) do walker_idx
        _evaluate_ensemble_walker!!(
            chain_state, proposal, step_rngpart, proposal_idx, walker_idx,
            complement_groups, p_accept, constant_ladj, acceptance_rngpart,
        )
    end
    return chain_state
end


function _mcmc_step_transition!!(
    mcmc_state::MCMCState,
    active_proposal::AbstractEnsembleMove,
    step_rngpart::RNGPartition,
    proposal_idx::Integer,
    walker_order::AbstractVector{<:Integer},
)
    chain_state = mcmc_state.chain_state
    groups = _ensemble_move_groups(
        active_proposal, step_rngpart, proposal_idx, walker_order,
    )
    T = float(eltype(first(chain_state.current.z.v)))
    p_accept = Vector{T}(undef, length(walker_order))
    step_info = MCMCStepInfo(
        p_accept, nothing, nothing, nothing, nothing, walker_order,
    )
    constant_ladj = _transform_ladj(chain_state.f_transform)
    acceptance_rngpart = _mcmc_walker_rngpart(
        step_rngpart, _MCMC_ACCEPTANCE_PURPOSE, proposal_idx,
    )

    for active_idx in eachindex(groups)
        active_group = groups[active_idx]
        complement_groups = _ensemble_complement_groups(groups, active_idx)
        _propose_ensemble_group!!(
            chain_state, active_proposal, step_rngpart, proposal_idx,
            active_group, complement_groups, p_accept,
            constant_ladj, acceptance_rngpart,
        )
        _apply_mcmc_subset!!(chain_state, step_info, active_group)
    end

    return _finalize_mcmc_step!!(
        mcmc_state, active_proposal, active_proposal, step_info,
    )
end
