# This file is a part of BAT.jl, licensed under the MIT License (MIT).

struct MCMCMultiProposal{
    P<:Tuple{Vararg{MCMCProposal}},
    R<:Union{Tuple{Vararg{Integer}}, Categorical}
}<:MCMCProposal
    proposals::P
    picking_rule::R
end

export MCMCMultiProposal

struct MultiProposalState{
    PS<:Tuple{Vararg{MCMCProposalState}},
    R<:Union{Tuple{Vararg{Integer}}, Categorical},
    I<:Integer
}<:MCMCProposalState
    proposal_states::PS
    picking_rule::R
    current_idx::I
end

export MultiProposalState

function bat_default(
    ::Type{TransformedMCMC}, 
    ::Val{:proposal_tuning}, 
    proposal::MCMCMultiProposal
)
    proposal_tunings = Tuple(bat_default.(TransformedMCMC, Val(:proposal_tuning), proposal.proposals))

    return MultiProposalTuning(proposal_tunings)
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
    proposal_state::MultiProposalState{<:Any, <:Tuple{Vararg{Integer}}}, 
    stepno::Integer, 
    rng::AbstractRNG
)
    picking_rule = proposal_state.picking_rule
    m = stepno % last(picking_rule)

    idx = m > 0 ? findfirst(y -> m <= y, picking_rule) : lastindex(picking_rule)
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

    proposal_states = Tuple(proposal_states_init)
    picking_rule = multi_proposal.picking_rule

    # To simplify calculation of current proposal, we store the cumulative sum in case of Integer Tuple picking rule.
    picking_rule = picking_rule isa Distribution ? picking_rule : cumsum(picking_rule)

    idx = picking_rule isa Distribution ? rand(rng, picking_rule) : 1

    return MultiProposalState(proposal_states, picking_rule, idx)
end

function set_proposal_transform!!(
    multi_proposal::MultiProposalState,
    chain_state::MCMCChainState 
)

    for proposal in multi_proposal.proposal_states
	    proposal = set_proposal_transform!!(proposal, chain_state)
    end

    return multi_proposal
end
