# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct MultiProposalTuning <: MCMCProposalTuning

Tuning algorithm for MCMCMultiProposals.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
struct MultiProposalTuning{
    PT<:Vector{<:MCMCProposalTuning},
}<:MCMCProposalTuning
    proposal_tunings::PT
end

export MultiProposalTuning

struct MultiProposalTunerState{
    PTS<:Vector{<:MCMCProposalTunerState},
}<:MCMCProposalTunerState
    proposal_tuners::PTS
end

export MultiProposalTunerState



function create_proposal_tuner_state(
    multi_tuning::MultiProposalTuning, 
    chain_state::MCMCChainState,
    multi_proposal::MultiProposalState,
    iteration::Integer
)
    proposal_tuners_init = Vector{MCMCProposalTunerState}()

    proposal_tunings = multi_tuning.proposal_tunings
    proposals = multi_proposal.proposal_states

    for i in eachindex(multi_tuning.proposal_tunings)
        tuner_tmp = create_proposal_tuner_state(
            proposal_tunings[i],
            chain_state,
            proposals[i],
            iteration
        )

        push!(proposal_tuners_init, tuner_tmp)
    end

    return MultiProposalTunerState(proposal_tuners_init)
end

function mcmc_proposal_tuning_init!!(
    multi_tuner_state::MultiProposalTunerState, 
    chain_state::MCMCChainState, 
    max_nsteps::Integer
)
    for tuner in multi_tuner_state.proposal_tuners
        mcmc_proposal_tuning_init!!(tuner, chain_state, max_nsteps)
    end
end

function mcmc_proposal_tuning_reinit!!(
    multi_tuner_state::MultiProposalTunerState,
    chain_state::MCMCChainState,
    max_nsteps::Integer
)
    for tuner in multi_tuner_state.proposal_tuners
        mcmc_proposal_tuning_reinit!!(tuner, chain_state, max_nsteps)
    end
end


function mcmc_proposal_tuning_postinit!!(
    multi_tuner::MultiProposalTunerState, 
    chain_state::MCMCChainState, 
    samples::AbstractVector{<:DensitySampleVector}
)
    for tuner in multi_tuner.proposal_tuners
        mcmc_proposal_tuning_postinit!!(tuner, chain_state, samples)
    end
end


# Make properly !!. In the for loop the proposals/tuners are overwritten
function mcmc_tune_proposal_post_cycle!!(
    multi_proposal::MultiProposalState,
    multi_tuner::MultiProposalTunerState,
    chain_state::MCMCChainState,
    samples::AbstractVector{<:DensitySampleVector}
)
    proposals = multi_proposal.proposal_states
    for i in eachindex(proposals)
        proposal = proposals[i]
        tuner = multi_tuner.proposal_tuners[i] 
        
        proposal, tuner, chain_state = mcmc_tune_proposal_post_cycle!!(
            proposal,
            tuner,
            chain_state,
            samples
        )
    end

    return multi_proposal, multi_tuner, chain_state 
end


function mcmc_proposal_tuning_finalize!!(
    multi_proposal::MultiProposalState,
    multi_tuner::MultiProposalTunerState, 
    chain_state::MCMCChainState
)
   proposals = multi_proposal.proposal_states
    for i in eachindex(proposals)
        proposal = proposals[i]
        tuner = multi_tuner.proposal_tuners[i] 

        mcmc_proposal_tuning_finalize!!(proposal, tuner, chain_state) 
    end

    return multi_proposal, multi_tuner, chain_state
end

function mcmc_tune_proposal_post_step!!(
    multi_proposal::MultiProposalState,
    multi_tuner::MultiProposalTunerState,
    chain_state::MCMCChainState,
    p_accept::AbstractVector{<:Real}
)
    idx_current = multi_proposal.current_idx
    
    current_proposal = get_current_proposal(multi_proposal)
    current_tuner = multi_tuner.proposal_tuners[idx_current]

    current_proposal_tuned, current_tuner, chain_state = mcmc_tune_proposal_post_step!!(
        current_proposal, 
        current_tuner, 
        chain_state, 
        p_accept
    )

    multi_proposal = @set multi_proposal.proposal_states[idx_current] = current_proposal_tuned

    return multi_proposal, multi_tuner, chain_state
end
