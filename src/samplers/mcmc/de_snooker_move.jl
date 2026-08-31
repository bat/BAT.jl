# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct DESnookerMove <: MCMCProposal

Experimental four-group differential-evolution snooker proposal. `nwalkers`
must be set explicitly on `TransformedMCMC`.

Constructors:

* ```$(FUNCTIONNAME)(; scale = 1.7, executor = BAT.SequentialExec())```
"""
struct DESnookerMove{S<:Real,E<:BATExecutor} <: MCMCProposal
    scale::S
    executor::E

    function DESnookerMove(scale::S, executor::E) where {S<:Real,E<:BATExecutor}
        isfinite(scale) && scale > zero(scale) || throw(ArgumentError(
            "DESnookerMove scale must be finite and positive",
        ))
        _validate_ensemble_executor(executor)
        new{S,E}(scale, executor)
    end
end

DESnookerMove(;
    scale::Real = 1.7,
    executor::BATExecutor = SequentialExec(),
) = DESnookerMove(scale, executor)
export DESnookerMove


struct DESnookerMoveProposalState{S<:Real,E<:BATExecutor} <: AbstractEnsembleMove
    scale::S
    executor::E
end

_mcmc_n_rng_purposes(::DESnookerMoveProposalState) = _MCMC_N_RNG_PURPOSES
_ensemble_group_count(::DESnookerMoveProposalState) = 4
_ensemble_minimum_walkers(::DESnookerMoveProposalState, n_dims::Integer) = max(2 * n_dims, 4)

function _proposal_diagnostics(
    ::DESnookerMoveProposalState,
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

function _mcmc_ess(
    chain_outputs::AbstractVector{<:AbstractVector{<:DensitySampleVector}},
    merged_output::DensitySampleVector,
    proposal::DESnookerMove,
    weighting::AbstractMCMCWeightingScheme,
    store_burnin::Bool,
    context::BATContext,
)
    _validate_mcmc_weighting_configuration(proposal, weighting)
    store_burnin && return nothing
    return _pooled_ensemble_ess(chain_outputs, merged_output, context)
end


bat_default(::Type{TransformedMCMC}, ::Val{:proposal_tuning}, ::DESnookerMove) =
    NoMCMCProposalTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:adaptive_transform}, ::DESnookerMove) =
    NoAdaptiveTransform()

bat_default(
    ::Type{TransformedMCMC}, ::Val{:transform_tuning}, ::DESnookerMove, ::NoAdaptiveTransform,
) = NoMCMCTransformTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:tempering}, ::DESnookerMove) =
    NoMCMCTempering()

get_tuning_success(
    ::MCMCChainState,
    ::DESnookerMoveProposalState,
    ::NoMCMCProposalTunerState,
) = true

function bat_default(
    ::Type{TransformedMCMC},
    ::Val{:nwalkers},
    ::DESnookerMove,
    ::TransformIntent,
    ::MCMCTransformTuning,
    ::Integer,
)
    throw(ArgumentError(
        "DESnookerMove requires an explicit nwalkers setting on TransformedMCMC",
    ))
end

bat_default(
    ::Type{TransformedMCMC},
    ::Val{:init},
    ::DESnookerMove,
    ::TransformIntent,
    ::MCMCTransformTuning,
    ::Integer,
    ::Integer,
    ::Integer,
) = MCMCRetryInit()


_validate_mcmc_proposal_configuration(
    ::DESnookerMove,
    ::NoMCMCProposalTuning,
) = nothing

function _validate_mcmc_proposal_configuration(
    ::DESnookerMove,
    tuning::MCMCProposalTuning,
)
    throw(ArgumentError(
        "DESnookerMove requires NoMCMCProposalTuning, got $(nameof(typeof(tuning)))",
    ))
end

_validate_mcmc_transform_tuning_configuration(
    ::DESnookerMove,
    ::NoMCMCTransformTuning,
) = nothing

function _validate_mcmc_transform_tuning_configuration(
    ::DESnookerMove,
    tuning::MCMCTransformTuning,
)
    throw(ArgumentError(
        "DESnookerMove requires NoMCMCTransformTuning, got $(nameof(typeof(tuning)))",
    ))
end

_validate_mcmc_adaptive_transform_configuration(
    ::DESnookerMove,
    ::NoAdaptiveTransform,
) = nothing

function _validate_mcmc_adaptive_transform_configuration(
    ::DESnookerMove,
    adaptive_transform::AbstractAdaptiveTransform,
)
    throw(ArgumentError(
        "DESnookerMove requires NoAdaptiveTransform, got $(nameof(typeof(adaptive_transform)))",
    ))
end

function _validate_mcmc_weighting_configuration(
    ::DESnookerMove,
    ::RepetitionWeighting,
)
    return nothing
end

function _validate_mcmc_weighting_configuration(
    ::DESnookerMove,
    weighting::AbstractMCMCWeightingScheme,
)
    throw(ArgumentError(
        "DESnookerMove supports RepetitionWeighting only, got $(nameof(typeof(weighting)))",
    ))
end


function _create_proposal_state(
    proposal::DESnookerMove,
    ::BATMeasure,
    ::BATContext,
    ::AbstractVector,
    z_init::AbstractVector{PV},
    ::Function,
    ::AbstractRNG,
) where {P<:Real,PV<:AbstractVector{P}}
    T = float(eltype(first(z_init)))
    scale = convert(T, proposal.scale)
    isfinite(scale) && scale > zero(scale) || throw(ArgumentError(
        "DESnookerMove scale must remain finite and positive after conversion to $(T)",
    ))
    return DESnookerMoveProposalState(scale, proposal.executor)
end

function _create_proposal_state(
    proposal::DESnookerMove,
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
    proposal::DESnookerMoveProposalState,
    target::BATMeasure,
    z_init::AbstractVector,
)
    n_walkers = length(z_init)
    n_dims = totalndof(varshape(target))
    minimum_walkers = _ensemble_minimum_walkers(proposal, n_dims)
    n_walkers >= minimum_walkers || throw(ArgumentError(
        "DESnookerMove requires at least max(2 * d, 4) walkers; got $n_walkers walkers for dimension $n_dims (minimum $minimum_walkers)",
    ))
    all(z -> all(isfinite, z), z_init) || throw(ArgumentError(
        "DESnookerMove requires finite transformed coordinates during initialization",
    ))

    centered_z = reduce(hcat, map(z -> z .- first(z_init), z_init))
    rank_rtol = max(size(centered_z)...)*eps(float(real(eltype(centered_z))))
    observed_rank = rank(centered_z; rtol = rank_rtol)
    observed_rank == n_dims || throw(ArgumentError(
        "DESnookerMove initialization has $n_walkers walkers in dimension $n_dims with affine rank $observed_rank; expected affine rank $n_dims",
    ))
    return nothing
end


const _DE_SNOOKER_GROUP_ORDERS = (
    (1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1),
)

function _de_snooker_companion_indices(
    rng::AbstractRNG,
    complement_groups,
)
    length(complement_groups) == 3 || throw(ArgumentError(
        "DESnookerMove requires exactly three frozen complement groups",
    ))
    all(!isempty, complement_groups) || throw(ArgumentError(
        "DESnookerMove requires nonempty frozen complement groups",
    ))
    group_order = rand(rng, _DE_SNOOKER_GROUP_ORDERS)
    return (
        rand(rng, complement_groups[group_order[1]]),
        rand(rng, complement_groups[group_order[2]]),
        rand(rng, complement_groups[group_order[3]]),
    )
end

function _de_snooker_direction_norm(a, b)
    T = float(promote_type(eltype(a), eltype(b)))
    result = zero(T)
    for i in eachindex(a, b)
        result = hypot(result, a[i] - b[i])
    end
    return result
end

function _de_snooker_candidate!!(
    candidate,
    current,
    reference,
    companion_a,
    companion_b,
    scale::Real,
)
    direction_norm = _de_snooker_direction_norm(current, reference)
    if iszero(direction_norm) || !isfinite(direction_norm)
        copyto!(candidate, current)
        return nothing
    end

    @. candidate = (current - reference) / direction_norm
    displacement = scale * (dot(candidate, companion_a) - dot(candidate, companion_b))
    @. candidate = current + candidate * displacement
    proposed_direction_norm = _de_snooker_direction_norm(candidate, reference)
    if iszero(proposed_direction_norm) || !isfinite(proposed_direction_norm)
        copyto!(candidate, current)
        return nothing
    end
    return (; reference, direction_norm, proposed_direction_norm)
end

function _propose_ensemble_candidate!!(
    candidate,
    proposal::DESnookerMoveProposalState,
    current,
    walker_idx::Integer,
    complement_groups,
    rng::AbstractRNG,
    step_rngpart::RNGPartition,
    proposal_idx::Integer,
    walkerid::Integer,
)
    companion_rngpart = _mcmc_walker_rngpart(
        step_rngpart, _MCMC_COMPANION_SELECTION_PURPOSE, proposal_idx,
    )
    set_rng!(rng, companion_rngpart, walkerid)
    reference_idx, companion_a_idx, companion_b_idx =
        _de_snooker_companion_indices(rng, complement_groups)
    return _de_snooker_candidate!!(
        candidate,
        current[walker_idx],
        current[reference_idx],
        current[companion_a_idx],
        current[companion_b_idx],
        proposal.scale,
    )
end

_ensemble_proposal_is_valid(::DESnookerMoveProposalState, proposal_aux) =
    !isnothing(proposal_aux)

function _ensemble_log_hastings(
    ::DESnookerMoveProposalState,
    current,
    proposed,
    proposal_aux,
)
    return (length(current) - 1) * (
        log(proposal_aux.proposed_direction_norm) - log(proposal_aux.direction_norm)
    )
end
