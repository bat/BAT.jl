# This file is a part of BAT.jl, licensed under the MIT License (MIT).

struct MultiProposalTuning{
    PT<:MCMCProposalTuning,
}<:MCMCProposalTuning
    proposal_tunings::Tuple{PT}
end

export MultiProposalTuning

struct MultiProposalTunerState{
    PTS<:MCMCProposalState,
}<:MCMCProposalTunerState
    proposal_tuners::Tuple{PTS}
end

export MultiProposalTunerState



function create_proposal_tuner_state(
    tuning::MultiProposalTuning, 
    chain_state::MCMCChainState, 
    iteration::Integer
)
    proposal_tuners = create_proposal_tuner_state.(
        tuning.proposal_tunings,
        chain_state,
        iteration
    )

    return MultiProposalTunerState(proposal_tuners)
end

function mcmc_tuning_init!!(
    tuner_state::MultiProposalTunerState, 
    chain_state::MCMCChainState, 
    max_nsteps::Integer
)
    mcmc_tuning_init!!.(tuner_state.proposal_tuners, chain_state, max_nsteps)
end

function mcmc_tuning_reinit!!(
    tuner_state::MultiProposalTunerState,
    chain_state::MCMCChainState,
    max_nsteps::Integer
)
    mcmc_tuning_reinit!!.(tuner_state.proposal_tuners, chain_state, max_nsteps)
end


function mcmc_tuning_postinit!!(
    tuner::MultiProposalTunerState, 
    chain_state::MCMCChainState, 
    samples::AbstractVector{<:DensitySampleVector}
)
    mcmc_tuning_postinit!!.(tuner.proposal_tuners, chain_state, samples)
end


function mcmc_tune_post_cycle!!(
    tuner::MultiProposalTunerState,
    chain_state::MCMCChainState,
    samples::AbstractVector{<:DensitySampleVector}
)
    ### TODO: MD: Implement

    return chain_state, tuner
end


mcmc_tuning_finalize!!(tuner::MultiProposalTunerState, chain_state::MCMCChainState) = nothing

function mcmc_tune_post_step!!(
    tuner::MultiProposalTunerState,
    chain_state::MCMCChainState,
    p_accept::AbstractVector{<:Real}
)
    ### TODO: MD: Implement

    return chain_state, tuner
end
