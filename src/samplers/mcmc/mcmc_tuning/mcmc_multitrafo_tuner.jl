# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct MultiTrafoTuning <: MCMCTransformTuning

Tuning algorithm for chains of adaptive transformations.

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
end

export MultiTrafoTunerState

function create_trafo_tuner_state(
    multi_tuning::MultiTrafoTuning,
    chain_state::MCMCChainState,
    n_steps_hint::Integer
)
    trafo_tuners = Vector{MCMCTransformTunerState}()

    trafo_tunings = multi_tuning.trafo_tunings

    for i in eachindex(multi_tuning.trafo_tunings)
        tuner_tmp = create_trafo_tuner_state(
            trafo_tunings[i],
            chain_state,
            n_steps_hint
        )

        push!(trafo_tuners, tuner_tmp)
    end

    return MultiTrafoTunerState(trafo_tuners)
end

function mcmc_tuning_init!!(
    multi_tuner_state::MultiTrafoTunerState,
    chain_state::MCMCChainState,
    max_nsteps::Integer
)
    for tuner in multi_tuner_state.trafo_tuners
        mcmc_tuning_init!!(tuner, chain_state, max_nsteps)
    end
end

function mcmc_tuning_reinit!!(
    multi_tuner_state::MultiTrafoTunerState, 
    chain_state::MCMCChainState, 
    max_nsteps::Integer
)
    for tuner in multi_tuner_state.trafo_tuners
        mcmc_tuning_reinit!!(tuner, chain_state, max_nsteps)
    end    
end

function mcmc_tuning_postinit!!(
    multi_tuner_state::MultiTrafoTunerState,
    chain_state::MCMCChainState,
    samples::AbstractVector{<:DensitySampleVector}
)
    for tuner in multi_tuner_state.trafo_tuners
        mcmc_tuning_postinit!!(tuner, chain_state, samples)
    end 
end

function mcmc_tune_post_cycle!!(
    f_transform::FunctionChain,
    multi_tuner_state::MultiTrafoTunerState,
    chain_state::MCMCChainState,
    samples::AbstractVector{<:DensitySampleVector}
)
    inv_intermediate_results = trafo_samples_with_interm_results(inverse(f_transform), samples)
    trafo_components = fchainfs(f_transform)
    trafo_tuners = multi_tuner_state.trafo_tuners

    for i in eachindex(trafo_components)
        j = length(trafo_components) + 1 - i

        trafo = trafo_components[j]
        tuner = trafo_tuners[j]
        intermediate_result = inv_intermediate_results[i]

        trafo_components[j], trafo_tuners[j], chain_state = mcmc_tune_post_cycle!!(
            trafo, 
            tuner, 
            chain_state, 
            intermediate_result
        )
    end

    return f_transform, multi_tuner_state, chain_state
end

function mcmc_tuning_finalize!!(
    trafo_chain::Function,
    multi_tuner_state::MultiTrafoTunerState,
    chain_state::MCMCChainState
)
    for i in eachindex(multi_tuner_state.trafo_tuners)
        f_transform = fchainfs(trafo_chain)[i]
        tuner = multi_tuner_state.trafo_tuners[i]
        mcmc_tuning_finalize!!(f_transform, tuner, chain_state)
    end 
    return trafo_chain, multi_tuner_state, chain_state
end

function mcmc_tune_post_step!!(
    f_transform::FunctionChain,
    multi_tuner_state::MultiTrafoTunerState, 
    chain_state::MCMCChainState,
    current::NamedTuple{<:Any, <:Tuple{Vararg{DensitySampleVector}}},
    proposed::NamedTuple{<:Any, <:Tuple{Vararg{DensitySampleVector}}},
    p_accept::AbstractVector{<:Real}
)
    intermediate_results = trafo_samples_with_interm_results(f_transform, curredfsdfat, proposed)
    trafo_components = fchainfs(f_transform)    
    trafo_tuners = multi_tuner_state.trafo_tuners

    for i in eachindex(trafo_components)
        j = length(trafo_components) + 1 - i

        trafo = trafo_components[j]
        tuner = trafo_tuners[j]
        current_interm_res, proposed_interm_res = intermediate_results[j]

        trafo_components[j], trafo_tuners[j], chain_state = mcmc_tune_post_step!!(
            trafo,
            tuner,
            chain_state,
            current_interm_res,
            proposed_interm_res,
            p_accept
        )
    end

    return f_transform, multi_tuner_state, chain_state
end
