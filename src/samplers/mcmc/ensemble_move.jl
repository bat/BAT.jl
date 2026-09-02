# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractEnsembleProposal <: MCMCProposal end

abstract type AbstractEnsembleMove <: MCMCProposalState end

_contains_ensemble_move(::AbstractEnsembleProposal) = true
_contains_ensemble_move(::AbstractEnsembleMove) = true


_validate_ensemble_executor(::Union{SequentialExec,MultiThreadedExec}) = nothing

function _validate_ensemble_executor(executor::BATExecutor)
    throw(ArgumentError(
        "Ensemble move executor must be SequentialExec or MultiThreadedExec, got $(nameof(typeof(executor)))",
    ))
end


bat_default(
    ::Type{TransformedMCMC},
    ::Val{:proposal_tuning},
    ::AbstractEnsembleProposal,
) = NoMCMCProposalTuning()

bat_default(
    ::Type{TransformedMCMC},
    ::Val{:adaptive_transform},
    ::AbstractEnsembleProposal,
) = NoAdaptiveTransform()

bat_default(
    ::Type{TransformedMCMC},
    ::Val{:transform_tuning},
    ::AbstractEnsembleProposal,
    ::NoAdaptiveTransform,
) = NoMCMCTransformTuning()

bat_default(
    ::Type{TransformedMCMC},
    ::Val{:tempering},
    ::AbstractEnsembleProposal,
) = NoMCMCTempering()

function bat_default(
    ::Type{TransformedMCMC},
    ::Val{:nwalkers},
    proposal::AbstractEnsembleProposal,
    ::TransformIntent,
    ::MCMCTransformTuning,
    ::Integer,
)
    throw(ArgumentError(
        "$(nameof(typeof(proposal))) requires an explicit nwalkers setting on TransformedMCMC",
    ))
end

bat_default(
    ::Type{TransformedMCMC},
    ::Val{:init},
    ::AbstractEnsembleProposal,
    ::TransformIntent,
    ::MCMCTransformTuning,
    ::Integer,
    ::Integer,
    ::Integer,
) = MCMCRetryInit()


_validate_mcmc_proposal_configuration(
    ::AbstractEnsembleProposal,
    ::NoMCMCProposalTuning,
) = nothing

function _validate_mcmc_proposal_configuration(
    proposal::AbstractEnsembleProposal,
    tuning::MCMCProposalTuning,
)
    throw(ArgumentError(
        "$(nameof(typeof(proposal))) requires NoMCMCProposalTuning, got $(nameof(typeof(tuning)))",
    ))
end

_validate_mcmc_transform_tuning_configuration(
    ::AbstractEnsembleProposal,
    ::NoMCMCTransformTuning,
) = nothing

function _validate_mcmc_transform_tuning_configuration(
    proposal::AbstractEnsembleProposal,
    tuning::MCMCTransformTuning,
)
    throw(ArgumentError(
        "$(nameof(typeof(proposal))) requires NoMCMCTransformTuning, got $(nameof(typeof(tuning)))",
    ))
end

_validate_mcmc_adaptive_transform_configuration(
    ::AbstractEnsembleProposal,
    ::NoAdaptiveTransform,
) = nothing

function _validate_mcmc_adaptive_transform_configuration(
    proposal::AbstractEnsembleProposal,
    adaptive_transform::AbstractAdaptiveTransform,
)
    throw(ArgumentError(
        "$(nameof(typeof(proposal))) requires NoAdaptiveTransform, got $(nameof(typeof(adaptive_transform)))",
    ))
end

_validate_mcmc_weighting_configuration(
    ::AbstractEnsembleProposal,
    ::RepetitionWeighting,
) = nothing

function _validate_mcmc_weighting_configuration(
    proposal::AbstractEnsembleProposal,
    weighting::AbstractMCMCWeightingScheme,
)
    throw(ArgumentError(
        "$(nameof(typeof(proposal))) supports RepetitionWeighting only, got $(nameof(typeof(weighting)))",
    ))
end


function _mcmc_ess(
    chain_outputs::AbstractVector{<:AbstractVector{<:DensitySampleVector}},
    merged_output::DensitySampleVector,
    ::AbstractEnsembleProposal,
    ::AbstractMCMCWeightingScheme,
    store_burnin::Bool,
    context::BATContext,
)
    store_burnin && return nothing
    return _pooled_ensemble_ess(chain_outputs, merged_output, context)
end


function _create_proposal_state(
    proposal::AbstractEnsembleProposal,
    target::BATMeasure,
    context::BATContext,
    v_init::AbstractVector,
    f_transform::Function,
    rng::AbstractRNG,
)
    z_init = inverse(f_transform).(v_init)
    proposal_state = _create_proposal_state(
        proposal, target, context, v_init, z_init, f_transform, rng,
    )
    _validate_mcmc_ensemble_invariants(proposal_state, target, z_init)
    return proposal_state
end

function _validate_mcmc_ensemble_invariants(
    proposal::AbstractEnsembleMove,
    target::BATMeasure,
    z_init::AbstractVector,
)
    proposal_name = replace(string(nameof(typeof(proposal))), "ProposalState" => "")
    n_walkers = length(z_init)
    n_dims = totalndof(varshape(target))
    minimum_walkers = _ensemble_minimum_walkers(proposal, n_dims)
    n_walkers >= minimum_walkers || throw(ArgumentError(
        "$proposal_name requires at least $minimum_walkers walkers for dimension $n_dims; got $n_walkers",
    ))
    all(z -> all(isfinite, z), z_init) || throw(ArgumentError(
        "$proposal_name requires finite transformed coordinates during initialization",
    ))

    centered_z = reduce(hcat, map(z -> z .- first(z_init), z_init))
    # `rank` compares singular values with `rtol * σ₁`; scaling rtol by the
    # matrix dimensions and scalar precision therefore preserves the affine
    # rank decision under a common coordinate rescaling.
    rank_rtol = max(size(centered_z)...) * eps(float(real(eltype(centered_z))))
    observed_rank = rank(centered_z; rtol = rank_rtol)
    observed_rank == n_dims || throw(ArgumentError(
        "$proposal_name initialization has $n_walkers walkers in dimension $n_dims with affine rank $observed_rank; expected affine rank $n_dims",
    ))
    return nothing
end


_mcmc_n_rng_purposes(::AbstractEnsembleMove) = _MCMC_N_RNG_PURPOSES

function _proposal_diagnostics(
    ::AbstractEnsembleMove,
    chain_state::MCMCChainState,
)
    n_attempts = only(chain_state.nattempts)
    n_accepted = only(chain_state.nsamples)
    acceptance_rate = iszero(n_attempts) ? NaN : n_accepted / n_attempts
    return (
        cycle_n_attempts = n_attempts,
        cycle_n_accepted = n_accepted,
        cycle_acceptance_rate = acceptance_rate,
    )
end

get_tuning_success(
    ::MCMCChainState,
    ::AbstractEnsembleMove,
    ::NoMCMCProposalTunerState,
) = true


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


@inline function _evaluate_ensemble_walker!!(
    chain_state::MCMCChainState,
    proposal::AbstractEnsembleMove,
    walker_idx::Integer,
    current_z,
    proposed_z,
    proposal_aux,
    p_accept::AbstractVector{<:Real},
    constant_ladj,
    acceptance_rngpart::RNGPartition,
    rng::AbstractRNG,
    walkerid::Integer,
)
    (;target, f_transform, current, proposed) = chain_state

    if !_ensemble_proposal_is_valid(proposal, proposal_aux)
        proposed.x[walker_idx] = current.x[walker_idx]
        proposed.z[walker_idx] = current.z[walker_idx]
        p_accept[walker_idx] = zero(eltype(p_accept))
        chain_state.accepted[walker_idx] = false
        return nothing
    end

    x_proposed, ladj = if isnothing(constant_ladj)
        with_logabsdet_jacobian(f_transform, proposed_z)
    else
        f_transform(proposed_z), constant_ladj
    end
    logd_x_proposed = BAT.checked_logdensityof(target, x_proposed)
    logd_z_proposed = logd_x_proposed + ladj

    proposed.x.v[walker_idx] .= x_proposed
    proposed.x.logd[walker_idx] = logd_x_proposed
    proposed.z.logd[walker_idx] = logd_z_proposed

    T = float(eltype(proposed_z))
    log_hastings = _ensemble_log_hastings(
        proposal, current_z, proposed_z, proposal_aux,
    )
    log_acceptance = log_hastings + logd_z_proposed - current.z.logd[walker_idx]
    p_accept[walker_idx] = _mcmc_acceptance_probability(convert(T, log_acceptance))

    set_rng!(rng, acceptance_rngpart, walkerid)
    chain_state.accepted[walker_idx] = rand(rng, T) < p_accept[walker_idx]
    return nothing
end


function _propose_and_evaluate_ensemble_walker!!(
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
    (;current, proposed) = chain_state
    walkerid = current.x.info[walker_idx].walkerid
    rng = get_rng(chain_state.walker_genctxs[walker_idx])
    current_z = current.z.v[walker_idx]
    proposed_z = proposed.z.v[walker_idx]
    proposal_aux = _propose_ensemble_candidate!!(
        proposed_z, proposal, current.z.v, walker_idx,
        complement_groups, rng, step_rngpart, proposal_idx, walkerid,
    )

    return _evaluate_ensemble_walker!!(
        chain_state, proposal, walker_idx, current_z, proposed_z, proposal_aux, p_accept,
        constant_ladj, acceptance_rngpart, rng, walkerid,
    )
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
        _propose_and_evaluate_ensemble_walker!!(
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
