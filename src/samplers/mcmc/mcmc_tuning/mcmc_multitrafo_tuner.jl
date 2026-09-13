# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct MultiTrafoTuning <: MCMCTransformTuning

Tuning algorithm for chains of adaptive transformations (see
[`AdaptiveTransformChain`](@ref)): one transform tuning per chain
component, each tuning its component against the samples in that
component's input/output spaces.

Score-based tunings (like `FisherTransformTuning`) are not supported as
components yet, their score transport would require the chain rule
through the other components.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
struct MultiTrafoTuning{
    TT<:Tuple{Vararg{MCMCTransformTuning}}
} <: MCMCTransformTuning
    trafo_tunings::TT
end

export MultiTrafoTuning


mutable struct MultiTrafoTunerState{
    TTS<:AbstractVector{MCMCTransformTunerState}
} <: MCMCTransformTunerState
    trafo_tuners::TTS
    # Whether the last post-step transform change came from a component
    # whose changes require a proposal step-size restart (queried via
    # transform_change_restarts_stepsize after each post-step):
    last_change_restarts::Bool
end

# Only changes committed by components that request it restart the
# proposal step-size adaptation (e.g. a RAM component changing the
# transform every step must not):
transform_change_restarts_stepsize(multi_tuner_state::MultiTrafoTunerState) =
    multi_tuner_state.last_change_restarts


function create_trafo_tuner_state(
    multi_tuning::MultiTrafoTuning,
    chain_state::MCMCChainState,
    n_steps_hint::Integer,
    adaptive_transform::AdaptiveTransformChain
)
    trafo_tunings = multi_tuning.trafo_tunings
    components = fchainfs(chain_state.f_transform)
    length(components) == length(trafo_tunings) || throw(ArgumentError(
        "MultiTrafoTuning has $(length(trafo_tunings)) component tunings but the transform chain has $(length(components)) components"
    ))
    any(t -> t isa FisherTransformTuning, trafo_tunings) && throw(ArgumentError(
        "FisherTransformTuning is not supported inside transform chains yet"
    ))

    trafo_tuners = map(eachindex(trafo_tunings)) do i
        # Component tuners are created against their own component
        # transform, not the whole chain:
        chain_state_i = @set chain_state.f_transform = components[i]
        create_trafo_tuner_state(
            trafo_tunings[i], chain_state_i, n_steps_hint, adaptive_transform.f[i]
        )
    end

    return MultiTrafoTunerState(trafo_tuners, false)
end

function create_trafo_tuner_state(
    multi_tuning::MultiTrafoTuning,
    chain_state::MCMCChainState,
    n_steps_hint::Integer
)
    throw(ArgumentError("MultiTrafoTuning requires an AdaptiveTransformChain adaptive transform"))
end


function mcmc_trafo_tuning_init!!(
    multi_tuner_state::MultiTrafoTunerState,
    chain_state::MCMCChainState,
    max_nsteps::Integer
)
    for tuner in multi_tuner_state.trafo_tuners
        mcmc_trafo_tuning_init!!(tuner, chain_state, max_nsteps)
    end
end

function mcmc_trafo_tuning_reinit!!(
    multi_tuner_state::MultiTrafoTunerState,
    chain_state::MCMCChainState,
    max_nsteps::Integer
)
    for tuner in multi_tuner_state.trafo_tuners
        mcmc_trafo_tuning_reinit!!(tuner, chain_state, max_nsteps)
    end
end

function mcmc_trafo_tuning_postinit!!(
    multi_tuner_state::MultiTrafoTunerState,
    chain_state::MCMCChainState,
    samples::AbstractVector{<:DensitySampleVector}
)
    inv_intermediate_results = trafo_samples_with_interm_results(inverse(chain_state.f_transform), samples)
    trafo_tuners = multi_tuner_state.trafo_tuners
    n = length(trafo_tuners)
    # Components use distinct intermediate samples.
    for j in eachindex(trafo_tuners)
        samples_j = inv_intermediate_results[n + 1 - j]
        mcmc_trafo_tuning_postinit!!(trafo_tuners[j], chain_state, samples_j)
    end
end


# Component updates are functional: a changed component yields a rebuilt
# chain, so the tuning orchestration detects the change (via !==) and
# resynchronizes the z-positions and the proposal. Mutating the chain's
# component container in place would defeat that detection and leave the
# chain state stale under the new geometry.

function mcmc_tune_trafo_post_cycle!!(
    f_transform::FunctionChain,
    multi_tuner_state::MultiTrafoTunerState,
    chain_state::MCMCChainState,
    proposal::MCMCProposalState,
    samples::AbstractVector{<:DensitySampleVector}
)
    # Entry i of the inverse-chain intermediates holds the samples in the
    # output space of component n + 1 - i:
    inv_intermediate_results = trafo_samples_with_interm_results(inverse(f_transform), samples)
    components_new = collect(Function, fchainfs(f_transform))
    trafo_tuners = multi_tuner_state.trafo_tuners
    n = length(components_new)

    changed = false
    restart = false
    # Each update threads chain state.
    for j in eachindex(components_new)
        samples_j = inv_intermediate_results[n + 1 - j]
        f_j_new, trafo_tuners[j], chain_state = mcmc_tune_trafo_post_cycle!!(
            components_new[j],
            trafo_tuners[j],
            chain_state,
            proposal,
            samples_j
        )
        if f_j_new !== components_new[j]
            components_new[j] = f_j_new
            changed = true
            restart |= transform_change_restarts_stepsize(trafo_tuners[j])
        end
    end
    # The restart policy must reflect this change, not a previous
    # post-step one:
    multi_tuner_state.last_change_restarts = changed && restart

    f_transform_new = changed ? fchain((components_new...,)) : f_transform
    return f_transform_new, multi_tuner_state, chain_state
end

function mcmc_trafo_tuning_finalize!!(
    trafo_chain::FunctionChain,
    multi_tuner_state::MultiTrafoTunerState,
    chain_state::MCMCChainState
)
    components_new = collect(Function, fchainfs(trafo_chain))
    trafo_tuners = multi_tuner_state.trafo_tuners

    changed = false
    # Each update threads chain state.
    for j in eachindex(components_new)
        f_j_new, trafo_tuners[j], chain_state = mcmc_trafo_tuning_finalize!!(
            components_new[j], trafo_tuners[j], chain_state
        )
        if f_j_new !== components_new[j]
            components_new[j] = f_j_new
            changed = true
        end
    end

    trafo_chain_new = changed ? fchain((components_new...,)) : trafo_chain
    return trafo_chain_new, multi_tuner_state, chain_state
end

function mcmc_tune_trafo_post_step!!(
    f_transform::FunctionChain,
    multi_tuner_state::MultiTrafoTunerState,
    chain_state::MCMCChainState,
    proposal::MCMCProposalState,
    current::NamedTuple{<:Any, <:Tuple{Vararg{DensitySampleVector}}},
    proposed::NamedTuple{<:Any, <:Tuple{Vararg{DensitySampleVector}}},
    step_info::MCMCStepInfo
)
    # Entry j holds the (current, proposed) samples at component j's
    # input (z) and output (x) spaces, all computed under the pre-update
    # chain, so every component tuner sees the same step's data:
    intermediate_results = trafo_samples_with_interm_results(f_transform, current, proposed)
    components_new = collect(Function, fchainfs(f_transform))
    trafo_tuners = multi_tuner_state.trafo_tuners

    changed = false
    restart = false
    # Each update threads chain state.
    for j in eachindex(components_new)
        current_j, proposed_j = intermediate_results[j]
        f_j_new, trafo_tuners[j], chain_state = mcmc_tune_trafo_post_step!!(
            components_new[j],
            trafo_tuners[j],
            chain_state,
            proposal,
            current_j,
            proposed_j,
            step_info
        )
        if f_j_new !== components_new[j]
            components_new[j] = f_j_new
            changed = true
            restart |= transform_change_restarts_stepsize(trafo_tuners[j])
        end
    end
    multi_tuner_state.last_change_restarts = changed && restart

    f_transform_new = changed ? fchain((components_new...,)) : f_transform
    return f_transform_new, multi_tuner_state, chain_state
end


# One transform tuning per chain component, by the per-component defaults:
function bat_default(TM::Type{TransformedMCMC}, tt::Val{:transform_tuning}, proposal::MCMCProposal, f_transform::AdaptiveTransformChain)
    tunings = bat_default.(TM, tt, Ref(proposal), f_transform.f)
    # Fail at configuration time, not later during state creation: the
    # per-component defaults for gradient-based proposals are score-based
    # (Fisher), which transform chains don't support yet:
    if any(t -> t isa FisherTransformTuning, tunings)
        throw(ArgumentError(
            "The default transform tuning for $(nameof(typeof(proposal))) components is score-based (FisherTransformTuning), which is not supported inside an AdaptiveTransformChain yet - please specify a supported transform_tuning (e.g. MultiTrafoTuning of RAMTuning components) explicitly"
        ))
    end
    return MultiTrafoTuning(Tuple(tunings))
end
