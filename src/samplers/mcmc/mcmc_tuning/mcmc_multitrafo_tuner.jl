# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct MultiTrafoTuning <: MCMCTransformTuning

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
struct MultiTrafoTuning{TT<:MCMCTransformTuning} <: MCMCTransformTuning
    trafo_tunings::Tuple{TT}
end

export MultiTrafoTuning


mutable struct MultiTrafoTunerState{
    TTS<:MCMCTransformTunerState
} <: MCMCTransformTunerState
    trafo_tuner_states::Tuple{TTS}
end


function create_trafo_tuner_state(
    tuning::MultiTrafoTuning,
    chain_state::MCMCChainState,
    n_steps_hint::Integer
)
    trafo_tuner_states = create_trafo_tuner_state.(
        tuning.trafo_tunings,
        chain_state,
        n_steps_hint
    )

    return MultiTrafoTunerState(trafo_tuner_states)
end

function mcmc_tuning_init!!(
    tuner_state::MultiTrafoTunerState,
    chain_state::MCMCChainState,
    max_nsteps::Integer
)
    mcmc_tuning_init!!.(tuner_state.trafo_tuner_states, chain_state, max_nsteps)
end

function mcmc_tuning_reinit!!(
    tuner_state::MultiTrafoTunerState, 
    chain_state::MCMCChainState, 
    max_nsteps::Integer
)
    mcmc_tuning_reinit!!(tuner_state.trafo_tuner_states, chain_state, max_nsteps)
end

function mcmc_tuning_postinit!!(
    tuner::MultiTrafoTunerState,
    chain_state::MCMCChainState,
    samples::AbstractVector{<:DensitySampleVector}
)
    mcmc_tuning_postinit!!(tuner.trafo_tuner_states, chain_state, samples)
end

function mcmc_tune_post_cycle!!(
    multi_tuner::MultiTrafoTunerState,
    chain_state::MCMCChainState,
    samples::AbstractVector{<:DensitySampleVector}
)
    # TODO: MD, incorporate intermediate transformations, decide how to handle the passing of trafos.

    tuner_states = multi_tuner.trafo_tuner_states
    for tuner in tuner_states
        chain_state, tuner = mcmc_tune_post_cycle!!(tuner, chain_state, samples)
    end

    return chain_state, multi_tuner
end

function mcmc_tuning_finalize!!(
    tuner::MultiTrafoTunerState,
    chain::MCMCChainState
)
    return nothing 
end

function mcmc_tune_post_step!!(
    multi_tuner::MultiTrafoTunerState, 
    chain_state::MCMCChainState,
    p_accept::AbstractVector{<:Real},
)
    # TODO: MD; Handle intermediate trafo results, decide how to pass trafo.
    
    for tuner in multi_tuner.trafo_tuner_states
        chain_state, tuner = mcmc_tuner_post_step!!(tuner, chain_state, p_accept)
    end

    return mc_state_new, tuner_state_new
end
