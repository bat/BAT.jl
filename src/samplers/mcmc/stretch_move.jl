# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct StretchMove <: MCMCProposal

Experimental affine-invariant ensemble proposal using the Goodman--Weare
stretch move. `nwalkers` must be set explicitly on `TransformedMCMC`.

Constructors:

* ```$(FUNCTIONNAME)(; scale = 2, executor = BAT.SequentialExec())```
"""
struct StretchMove{S<:Real,E<:BATExecutor} <: MCMCProposal
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

_mcmc_n_rng_purposes(::StretchMoveProposalState) = _MCMC_N_RNG_PURPOSES
_ensemble_group_count(::StretchMoveProposalState) = 2
_ensemble_minimum_walkers(::StretchMoveProposalState, n_dims::Integer) = 2 * n_dims

function _proposal_diagnostics(
    ::StretchMoveProposalState,
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
    proposal::StretchMove,
    weighting::AbstractMCMCWeightingScheme,
    store_burnin::Bool,
    context::BATContext,
)
    _validate_mcmc_weighting_configuration(proposal, weighting)
    store_burnin && return nothing
    return _pooled_ensemble_ess(chain_outputs, merged_output, context)
end


bat_default(::Type{TransformedMCMC}, ::Val{:proposal_tuning}, ::StretchMove) =
    NoMCMCProposalTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:adaptive_transform}, ::StretchMove) =
    NoAdaptiveTransform()

bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, ::StretchMove, ::NoAdaptiveTransform) =
    NoMCMCTransformTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:tempering}, ::StretchMove) =
    NoMCMCTempering()

get_tuning_success(
    ::MCMCChainState,
    ::StretchMoveProposalState,
    ::NoMCMCProposalTunerState,
) = true

function bat_default(
    ::Type{TransformedMCMC},
    ::Val{:nwalkers},
    ::StretchMove,
    ::TransformIntent,
    ::MCMCTransformTuning,
    ::Integer,
)
    throw(ArgumentError("StretchMove requires an explicit nwalkers setting on TransformedMCMC"))
end

bat_default(
    ::Type{TransformedMCMC},
    ::Val{:init},
    ::StretchMove,
    ::TransformIntent,
    ::MCMCTransformTuning,
    ::Integer,
    ::Integer,
    ::Integer,
) = MCMCRetryInit()


_validate_mcmc_proposal_configuration(
    ::StretchMove,
    ::NoMCMCProposalTuning,
) = nothing

function _validate_mcmc_proposal_configuration(
    ::StretchMove,
    tuning::MCMCProposalTuning,
)
    throw(ArgumentError(
        "StretchMove requires NoMCMCProposalTuning, got $(nameof(typeof(tuning)))",
    ))
end

_validate_mcmc_transform_tuning_configuration(
    ::StretchMove,
    ::NoMCMCTransformTuning,
) = nothing

function _validate_mcmc_transform_tuning_configuration(
    ::StretchMove,
    tuning::MCMCTransformTuning,
)
    throw(ArgumentError(
        "StretchMove requires NoMCMCTransformTuning, got $(nameof(typeof(tuning)))",
    ))
end


_validate_mcmc_adaptive_transform_configuration(
    ::StretchMove,
    ::NoAdaptiveTransform,
) = nothing

function _validate_mcmc_adaptive_transform_configuration(
    ::StretchMove,
    adaptive_transform::AbstractAdaptiveTransform,
)
    throw(ArgumentError(
        "StretchMove requires NoAdaptiveTransform, got $(nameof(typeof(adaptive_transform)))",
    ))
end


function _validate_mcmc_weighting_configuration(
    ::StretchMove,
    ::RepetitionWeighting,
)
    return nothing
end

function _validate_mcmc_weighting_configuration(
    ::StretchMove,
    weighting::AbstractMCMCWeightingScheme,
)
    throw(ArgumentError(
        "StretchMove supports RepetitionWeighting only, got $(nameof(typeof(weighting)))",
    ))
end


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


function _create_proposal_state(
    proposal::StretchMove,
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
    proposal::StretchMoveProposalState,
    target::BATMeasure,
    z_init::AbstractVector,
)
    n_walkers = length(z_init)
    n_dims = totalndof(varshape(target))
    n_walkers >= _ensemble_minimum_walkers(proposal, n_dims) || throw(ArgumentError(
        "StretchMove requires at least 2 * d walkers; got $n_walkers walkers for dimension $n_dims",
    ))
    all(z -> all(isfinite, z), z_init) || throw(ArgumentError(
        "StretchMove requires finite transformed coordinates during initialization",
    ))

    centered_z = reduce(hcat, map(z -> z .- first(z_init), z_init))
    # `rank` compares singular values with `rtol * σ₁`; scaling rtol by the
    # matrix dimensions and scalar precision therefore preserves the affine
    # rank decision under a common coordinate rescaling.
    rank_rtol = max(size(centered_z)...)*eps(float(real(eltype(centered_z))))
    observed_rank = rank(centered_z; rtol = rank_rtol)
    observed_rank == n_dims || throw(ArgumentError(
        "StretchMove initialization has $n_walkers walkers in dimension $n_dims with affine rank $observed_rank; expected affine rank $n_dims",
    ))

    return nothing
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


function _propose_ensemble_candidate!!(
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

function _ensemble_log_hastings(
    ::StretchMoveProposalState,
    current,
    proposed,
    stretch::Real,
)
    return _stretch_log_acceptance(length(proposed), stretch, false, false)
end
