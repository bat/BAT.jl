# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct MultiTrafoTuning <: MCMCTransformTuning

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
    TTS<:Tuple{Vararg{MCMCTransformTunerState}}
} <: MCMCTransformTunerState
    trafo_tuners::TTS
end

export MultiTrafoTunerState

function create_trafo_tuner_state(
    multi_tuning::MultiTrafoTuning,
    chain_state::MCMCChainState,
    n_steps_hint::Integer
)
    trafo_tuners_init = Vector{MCMCTransformTunerState}()

    trafo_tunings = multi_tuning.trafo_tunings

    for i in eachindex(multi_tuning.trafo_tunings)
        tuner_tmp = create_trafo_tuner_state(
            trafo_tunings[i],
            chain_state,
            n_steps_hint
        )

        push!(trafo_tuners_init, tuner_tmp)
    end

    trafo_tuners = Tuple(trafo_tuners_init)

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
    trafo_components = fchainfs(f_transform)
    inv_intermediate_results = with_intermediate_results.(inverse(f_transform), 1)

    prepend!(inv_intermediate_results, samples)

    for i in eachindex(trafo_components)
        trafo = trafo_components[i]
        tuner = multi_tuner_state.trafo_tuners[i]
        intermediate_result = inv_intermediate_results[end - i]

        trafo, tuner, chain_state = mcmc_tune_post_cycle!!(trafo, tuner, chain_state, intermediate_result)
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
end

function mcmc_tune_post_step!!(
    f_transform::FunctionChain,
    multi_tuner_state::MultiTrafoTunerState, 
    chain_state::MCMCChainState,
    current::NamedTuple{<:Any, <:Tuple{Vararg{DensitySampleVector}}},
    proposed::NamedTuple{<:Any, <:Tuple{Vararg{DensitySampleVector}}},
    p_accept::AbstractVector{<:Real}
)
    trafo_components = fchainfs(f_transform)    

    inv_current_intermediate_results = with_intermediate_results(inverse(f_transform), current)
    prepend!(inv_current_intermediate_results, current)
    inv_proposed_intermediate_results = with_intermediate_results(inverse(f_transform), proposed)
    prepend!(inv_proposed_intermediate_results, proposed)

    for i in eachindex(trafo_components)
        trafo = trafo_components[i]
        tuner = multi_tuner_state.trafo_tuners[i]
        current_intermediate_result = inv_current_intermediate_results[end - i]
        proposed_intermediate_result = inv_proposed_intermediate_results[end - i]

        trafo, tuner, chain_state = mcmc_tune_post_step!!(
            trafo,
            tuner,
            chain_state,
            current_intermediate_result,
            proposed_intermediate_result,
            p_accept
        )
    end

    return f_transform, multi_tuner_state, chain_state
end
