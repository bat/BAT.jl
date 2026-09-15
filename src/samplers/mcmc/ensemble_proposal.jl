# This file is a part of BAT.jl, licensed under the MIT License (MIT).

abstract type AbstractEnsembleProposal <: MCMCProposal end
abstract type AbstractEnsembleMove <: MCMCProposalState end

"""
    EnsembleProposal(move; executor=nothing, batch_logdensity! = nothing,
                     record_transitions=false)

Wrap an EnsembleMCMC.jl move or `MoveMixture` for `TransformedMCMC`.
Load EnsembleMCMC before constructing this experimental proposal. `executor`
accepts its serial or threaded executor and defaults to `SerialExecutor()`.
Each BAT chain is one coupled ensemble. Set `nwalkers` explicitly. The adapter
supports fixed transforms, repetition weights, and fixed BAT proposal mixtures.

Set `batch_logdensity!` to a CPU callback `(values, transformed_target, positions)`.
It must fill one full log density per column of `positions`, equal to
`checked_logdensityof(transformed_target, positions[:, i])`. BAT supplies the
transformed, unshaped target, including its prior and Jacobian terms. The callback
must not mutate positions or retain borrowed arrays and controls its own
parallelism. BAT supplies cached initial densities without calling this callback.

Set `record_transitions=true` to retain every ensemble sweep in proposal
diagnostics under `transitions`, with `walker_ids` identifying the array order.
Each record owns `accepted` and `acceptance_probabilities` arrays and records
the BAT `cycle`, BAT `step`, and inner `move_index`. Records include initialization
and burn-in cycles, even when their samples are discarded. Storage grows with
sweeps times walkers. BAT sample proposal IDs still identify the outer proposal.
BAT owns the RNG addresses in all modes.
"""
struct EnsembleProposal{M,E,B} <: AbstractEnsembleProposal
    move::M
    executor::E
    batch_logdensity!::B
    record_transitions::Bool
end

function EnsembleProposal(move; executor=nothing, batch_logdensity! = nothing, record_transitions::Bool=false)
    pkgext(Val(:EnsembleMCMC))
    return EnsembleProposal(move, executor, batch_logdensity!, record_transitions)
end
EnsembleProposal(move, executor) = EnsembleProposal(move; executor)
export EnsembleProposal

_contains_ensemble_move(::AbstractEnsembleProposal) = true
_contains_ensemble_move(::AbstractEnsembleMove) = true
_mcmc_n_rng_purposes(::AbstractEnsembleMove) = _MCMC_N_RNG_PURPOSES

get_tuning_success(::MCMCChainState, ::AbstractEnsembleMove, ::NoMCMCProposalTunerState) = true

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
