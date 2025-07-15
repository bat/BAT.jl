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
    # TODO: MD, incorporate intermediate transformations, decide how to handle the passing of trafos.

    trafo_components = fchainfs(f_transform)
    intermediate_results = with_intermediate_results(chain_state.f_transform, samples)

    for i in eachindex(trafo_components)
        trafo = trafo_components[i]
        tuner = multi_tuner_state.trafo_tuners[i]
        intermediate_result = intermediate_results[i]

        trafo, tuner, chain_state = mcmc_tuner_post_cycle!!(trafo, tuner, chain_state, intermediate_result)
    end

    return f_transform, multi_tuner_state, chain_state
end

function mcmc_tuning_finalize!!(
    multi_tuner_state::MultiTrafoTunerState,
    chain_state::MCMCChainState
)
    for tuner in multi_tuner_state.trafo_tuners
        mcmc_tuning_finalize!!(tuner, chain_state)
    end 
end

function mcmc_tune_post_step!!(
    f_transform::FunctionChain,
    multi_tuner_state::MultiTrafoTunerState, 
    chain_state::MCMCChainState,
    p_accept::AbstractVector{<:Real},
)
    trafo_components = fchainfs(f_transform)    

    for i in eachindex(trafo_components)
        trafo = trafo_components[i]
        tuner = multi_tuner_state.trafo_tuners[i]

        trafo, tuner, chain_state = mcmc_tune_post_step!!(trafo, tuner, chain_state, p_accept)
    end

    return f_transform, multi_tuner_state, chain_state
end
