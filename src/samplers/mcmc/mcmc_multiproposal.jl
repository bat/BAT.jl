# This file is a part of BAT.jl, licensed under the MIT License (MIT).
"""
    struct MCMCMultiProposal<: MCMCProposal

MCMC sampling algorithm that allows for using multiple
different proposal algorithms during sampling.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MCMCMultiProposal{
    P<:Vector{<:MCMCProposal},
    R<:Union{Vector{<:Integer}, Categorical}
}<:MCMCProposal
    # TODO: MD, should we put a default tuple of proposals, if so, what should it be?
    proposals::P = (RandomWalk(), HamiltonianMC())
    picking_rule::R = Categorical(1/length(proposals) .* ones(length(proposals)))
end

export MCMCMultiProposal

struct MultiProposalState{
    PS<:Vector{<:MCMCProposalState},
    R<:Union{Vector{<:Integer}, Categorical},
    I<:Integer
}<:MCMCProposalState
    proposal_states::PS
    picking_rule::R
    current_idx::I
end

export MultiProposalState

function bat_default(
    TM::Type{TransformedMCMC}, 
    pt::Val{:proposal_tuning}, 
    proposal::MCMCMultiProposal
)
    tunings = bat_default.(TM, pt, proposal.proposals)
    return MultiProposalTuning(tunings)
end

bat_default(
    ::Type{TransformedMCMC}, 
    ::Val{:adaptive_transform}, 
    proposal::MCMCMultiProposal
) = TriangularAffineTransform()

bat_default(
    ::Type{TransformedMCMC}, 
    ::Val{:tempering}, 
    proposal::MCMCMultiProposal
) = NoMCMCTempering()


get_current_proposal_idx(proposal_state::MCMCProposalState) = 1

get_current_proposal_idx(proposal_state::MultiProposalState) = proposal_state.current_idx

function set_current_proposal!!(
    proposal_state::MCMCProposalState,
    stepno::Integer,
    rng::AbstractRNG
)
    return proposal_state
end

function get_current_proposal(
    proposal_state::MCMCProposalState,
)
    return proposal_state
end


function set_current_proposal!!(
    proposal_state::MultiProposalState{<:Any, <:Vector{Integer}}, 
    stepno::Integer, 
    rng::AbstractRNG
)
    picking_rule_cum = cumsum(proposal_state.picking_rule)
    m = stepno % last(picking_rule_cum)

    idx = m > 0 ? findfirst(y -> m <= y, picking_rule_cum) : lastindex(picking_rule_cum)
    proposal_state_new = @set proposal_state.current_idx = idx

    return proposal_state_new
end

function set_current_proposal!!(
    proposal_state::MultiProposalState{<:Any, <:Distribution}, 
    stepno::Integer, 
    rng::AbstractRNG
)
    idx = rand(rng, proposal_state.picking_rule)

    proposal_state = @set proposal_state.current_idx = idx
    return proposal_state
end

function get_current_proposal(
    multi_proposal_state::MultiProposalState
)
    current_proposal = multi_proposal_state.proposal_states[multi_proposal_state.current_idx]
    return current_proposal
end

function get_target_acceptance_ratio(proposal::MultiProposalState)
    target_acc_ratios = Tuple(get_target_acceptance_ratio.(proposal.proposal_states))
    picking_rule = proposal.picking_rule

    proposal_probs = if picking_rule isa Distribution
        Tuple(picking_rule.p)
    else
        picking_rule ./ sum(picking_rule)
    end

    return dot(target_acc_ratios, proposal_probs)
end


function get_target_acceptance_int(proposal::MultiProposalState)
    target_acc_ints = Tuple(get_target_acceptance_int.(proposal.proposal_states))
    picking_rule = proposal.picking_rule

    lowers = first.(target_acc_ints)
    uppers = last.(target_acc_ints)

    proposal_probs = if picking_rule isa Distribution
        Tuple(picking_rule.p)
    else
        picking_rule ./ sum(picking_rule)
    end

    mean_target_acc_int = (dot(lowers, proposal_probs), dot(uppers, proposal_probs))
    return mean_target_acc_int
end

function get_tuning_success(
    chain_state::MCMCChainState,
    multi_proposal::MultiProposalState
)
    component_eff_acc= detailed_eff_acceptance_ratio(chain_state)
    component_acc_ints = get_target_acceptance_int.(multi_proposal.proposal_states)

    component_tuning_successes = [int[1] <= component_eff_acc[i] <= int[2] for (i, int) in enumerate(component_acc_ints)]
 
    return any(component_tuning_successes)
end

function _create_proposal_state(
    multi_proposal::MCMCMultiProposal, 
    target::BATMeasure, 
    context::BATContext, 
    v_init::AbstractVector{PV},
    f_transform::Function,
    rng::AbstractRNG
) where {P<:Real, PV<:AbstractVector{P}}

    proposal_states_init = Vector{MCMCProposalState}()

    for proposal in multi_proposal.proposals
        proposal_state_tmp = _create_proposal_state(
            proposal,
            target,
            context,
            v_init,
            f_transform,
            rng
        )
        push!(proposal_states_init, proposal_state_tmp)
    end

    picking_rule = multi_proposal.picking_rule

    idx = picking_rule isa Distribution ? rand(rng, picking_rule) : 1

    return MultiProposalState(proposal_states_init, picking_rule, idx)
end

function set_proposal_transform!!(
    multi_proposal::MultiProposalState,
    chain_state::MCMCChainState 
)

    for i in 1:length(multi_proposal.proposal_states)
	    multi_proposal.proposal_states[i] = set_proposal_transform!!(multi_proposal.proposal_states[i], chain_state)
    end

    return multi_proposal
end
