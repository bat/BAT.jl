# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct DEMove <: MCMCProposal

Experimental ensemble proposal using differential evolution against the
current complementary ensemble. `nwalkers` must be set explicitly on
`TransformedMCMC`.

Constructors:

* ```$(FUNCTIONNAME)(; gamma0 = nothing, sigma = 1e-5, executor = BAT.SequentialExec())```
"""
struct DEMove{G<:Union{Nothing,Real},S<:Real,E<:BATExecutor} <: MCMCProposal
    gamma0::G
    sigma::S
    executor::E

    function DEMove(
        gamma0::G,
        sigma::S,
        executor::E,
    ) where {G<:Union{Nothing,Real},S<:Real,E<:BATExecutor}
        (isnothing(gamma0) || isfinite(gamma0) && gamma0 > zero(gamma0)) ||
            throw(ArgumentError("DEMove gamma0 must be nothing or finite and positive"))
        isfinite(sigma) && sigma >= zero(sigma) || throw(ArgumentError(
            "DEMove sigma must be finite and nonnegative",
        ))
        _validate_ensemble_executor(executor)
        new{G,S,E}(gamma0, sigma, executor)
    end
end

DEMove(;
    gamma0::Union{Nothing,Real} = nothing,
    sigma::Real = 1e-5,
    executor::BATExecutor = SequentialExec(),
) = DEMove(gamma0, sigma, executor)
export DEMove


const _MCMC_DE_SCALE_PURPOSE = _MCMC_STRETCH_DRAW_PURPOSE


struct DEMoveProposalState{T<:Real,E<:BATExecutor} <: AbstractEnsembleMove
    gamma0::T
    sigma::T
    executor::E
end

_mcmc_n_rng_purposes(::DEMoveProposalState) = _MCMC_N_RNG_PURPOSES
_ensemble_group_count(::DEMoveProposalState) = 2
_ensemble_minimum_walkers(::DEMoveProposalState, n_dims::Integer) = max(2 * n_dims, 4)

function _proposal_diagnostics(
    ::DEMoveProposalState,
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
    proposal::DEMove,
    weighting::AbstractMCMCWeightingScheme,
    store_burnin::Bool,
    context::BATContext,
)
    _validate_mcmc_weighting_configuration(proposal, weighting)
    store_burnin && return nothing
    return _pooled_ensemble_ess(chain_outputs, merged_output, context)
end


bat_default(::Type{TransformedMCMC}, ::Val{:proposal_tuning}, ::DEMove) =
    NoMCMCProposalTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:adaptive_transform}, ::DEMove) =
    NoAdaptiveTransform()

bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, ::DEMove, ::NoAdaptiveTransform) =
    NoMCMCTransformTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:tempering}, ::DEMove) =
    NoMCMCTempering()

get_tuning_success(
    ::MCMCChainState,
    ::DEMoveProposalState,
    ::NoMCMCProposalTunerState,
) = true

function bat_default(
    ::Type{TransformedMCMC},
    ::Val{:nwalkers},
    ::DEMove,
    ::TransformIntent,
    ::MCMCTransformTuning,
    ::Integer,
)
    throw(ArgumentError("DEMove requires an explicit nwalkers setting on TransformedMCMC"))
end

bat_default(
    ::Type{TransformedMCMC},
    ::Val{:init},
    ::DEMove,
    ::TransformIntent,
    ::MCMCTransformTuning,
    ::Integer,
    ::Integer,
    ::Integer,
) = MCMCRetryInit()


_validate_mcmc_proposal_configuration(
    ::DEMove,
    ::NoMCMCProposalTuning,
) = nothing

function _validate_mcmc_proposal_configuration(
    ::DEMove,
    tuning::MCMCProposalTuning,
)
    throw(ArgumentError(
        "DEMove requires NoMCMCProposalTuning, got $(nameof(typeof(tuning)))",
    ))
end

_validate_mcmc_transform_tuning_configuration(
    ::DEMove,
    ::NoMCMCTransformTuning,
) = nothing

function _validate_mcmc_transform_tuning_configuration(
    ::DEMove,
    tuning::MCMCTransformTuning,
)
    throw(ArgumentError(
        "DEMove requires NoMCMCTransformTuning, got $(nameof(typeof(tuning)))",
    ))
end

_validate_mcmc_adaptive_transform_configuration(
    ::DEMove,
    ::NoAdaptiveTransform,
) = nothing

function _validate_mcmc_adaptive_transform_configuration(
    ::DEMove,
    adaptive_transform::AbstractAdaptiveTransform,
)
    throw(ArgumentError(
        "DEMove requires NoAdaptiveTransform, got $(nameof(typeof(adaptive_transform)))",
    ))
end

function _validate_mcmc_weighting_configuration(
    ::DEMove,
    ::RepetitionWeighting,
)
    return nothing
end

function _validate_mcmc_weighting_configuration(
    ::DEMove,
    weighting::AbstractMCMCWeightingScheme,
)
    throw(ArgumentError(
        "DEMove supports RepetitionWeighting only, got $(nameof(typeof(weighting)))",
    ))
end


function _create_proposal_state(
    proposal::DEMove,
    ::BATMeasure,
    ::BATContext,
    ::AbstractVector,
    z_init::AbstractVector{PV},
    ::Function,
    ::AbstractRNG,
) where {P<:Real,PV<:AbstractVector{P}}
    T = float(eltype(first(z_init)))
    n_dims = length(first(z_init))
    gamma0 = if isnothing(proposal.gamma0)
        convert(T, 2.38) / sqrt(convert(T, 2 * n_dims))
    else
        convert(T, proposal.gamma0)
    end
    sigma = convert(T, proposal.sigma)
    isfinite(gamma0) && gamma0 > zero(gamma0) || throw(ArgumentError(
        "DEMove gamma0 must remain finite and positive after conversion to $(T)",
    ))
    isfinite(sigma) && sigma >= zero(sigma) || throw(ArgumentError(
        "DEMove sigma must remain finite and nonnegative after conversion to $(T)",
    ))

    return DEMoveProposalState(gamma0, sigma, proposal.executor)
end

function _create_proposal_state(
    proposal::DEMove,
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
    proposal::DEMoveProposalState,
    target::BATMeasure,
    z_init::AbstractVector,
)
    n_walkers = length(z_init)
    n_dims = totalndof(varshape(target))
    minimum_walkers = _ensemble_minimum_walkers(proposal, n_dims)
    n_walkers >= minimum_walkers || throw(ArgumentError(
        "DEMove requires at least max(2 * d, 4) walkers; got $n_walkers walkers for dimension $n_dims (minimum $minimum_walkers)",
    ))
    all(z -> all(isfinite, z), z_init) || throw(ArgumentError(
        "DEMove requires finite transformed coordinates during initialization",
    ))

    centered_z = reduce(hcat, map(z -> z .- first(z_init), z_init))
    rank_rtol = max(size(centered_z)...)*eps(float(real(eltype(centered_z))))
    observed_rank = rank(centered_z; rtol = rank_rtol)
    observed_rank == n_dims || throw(ArgumentError(
        "DEMove initialization has $n_walkers walkers in dimension $n_dims with affine rank $observed_rank; expected affine rank $n_dims",
    ))

    return nothing
end


function _de_companion_indices(
    rng::AbstractRNG,
    companion_idxs::AbstractVector{<:Integer},
)
    Base.require_one_based_indexing(companion_idxs)
    n_companions = length(companion_idxs)
    n_companions >= 2 || throw(ArgumentError(
        "DEMove requires at least two walkers in each frozen complement",
    ))
    first_pos = rand(rng, 1:n_companions)
    second_pos = rand(rng, 1:(n_companions - 1))
    second_pos >= first_pos && (second_pos += 1)
    return companion_idxs[first_pos], companion_idxs[second_pos]
end

function _de_candidate!!(candidate, current, companion_a, companion_b, gamma::Real)
    @. candidate = current + gamma * (companion_a - companion_b)
    return candidate
end

function _propose_ensemble_candidate!!(
    candidate,
    proposal::DEMoveProposalState,
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
    scale_rngpart = _mcmc_walker_rngpart(
        step_rngpart, _MCMC_DE_SCALE_PURPOSE, proposal_idx,
    )
    set_rng!(rng, companion_rngpart, walkerid)
    companion_a, companion_b = _de_companion_indices(rng, companion_idxs)

    set_rng!(rng, scale_rngpart, walkerid)
    T = typeof(proposal.gamma0)
    gamma = proposal.gamma0 * (one(T) + proposal.sigma * randn(rng, T))
    _de_candidate!!(
        candidate, current[walker_idx], current[companion_a], current[companion_b], gamma,
    )
    return nothing
end

function _ensemble_log_hastings(
    proposal::DEMoveProposalState,
    current,
    proposed,
    ::Nothing,
)
    return zero(proposal.gamma0)
end
