# This file is a part of BAT.jl, licensed under the MIT License (MIT).

struct MCMCMultiProposal{
    P<:MCMCProposal,
    S<:Union{Integer, AbstractFloat}
}<:MCMCProposal
    proposals::Tuple{P}
    picking_rule::Tuple{S}
end

export MCMCMultiProposal

struct SequentialMultiProposalState{
    PS<:MCMCProposalState,
    S<:Tuple{Integer},
    I<:Integer
}<:MCMCProposalState
    proposal_states::Tuple{PS}
    schedule::S
    current_idx::I
end

export SequentialMultiProposalState

struct RandomMultiProposalState{
    PS<:MCMCProposalState,
    D::Multinomial,
    I<:Integer
}<:MCMCProposalState
    proposal_states::Tuple{PS}
    picking_distribution::D
    current_idx::I
end

export RandomMultiProposalState

function set_current_proposal!!(
    proposal_state::SequentialMultiProposalState, 
    stepno::Integer, 
    rng::AbstractRNG
)
    m = stepno%last(proposal_state.schedule)
    idx = findfirst(y -> m<y, proposal_state.schedule)

    proposal_state = @set proposal_state.current_idx = idx
    return proposal_state
end

function set_current_proposal!!(
    proposal_state::RandomMultiProposalState, 
    stepno::Integer, 
    rng::AbstractRNG
)
    idx = rand(rng, proposal_state.picking_distribution)
    proposal_state = @set proposal_state.current_idx = idx
    return proposal_state
end

function get_current_proposal!!(
    proposal_state::MCMCMultiProposalState,
    stepno::Integer,
    rng::AbstractRNG
)
    proposal_state = set_current_proposal!!(proposal_state, stepno, rng)
    current_proposal = proposal_state.proposal_states[propsal_state.current_idx]
    return proposal_state, current_proposal
end

function _get_sample_id(
    proposal_state::MCMCMultiProposalState, 
    chainid::Int32, 
    walkerid::Int32, 
    cycle::Int32, 
    stepno::Integer, 
    sample_type::Integer
)
    current_proposal = get_current_proposal(proposal_state)
    _get_sample_id(
        current_proposal,
        chainid,
        walkerid,
        cycle,
        stepno,
        sample_type
    )
end

function _create_proposal_state(
    proposal::MCMCMultiProposal, 
    target::BATMeasure, 
    context::BATContext, 
    v_init::AbstractVector{PV},
    f_transform::Function,
    rng::AbstractRNG
) where {P<:Real, PV<:AbstractVector{P}}

    proposal_states = _create_proposal_state.(
        proposal.proposals,
        target,
        context,
        v_init,
        f_transform,
        rng
    )

    picking_rule = proposal.picking_rule

    if picking_rule isa Tuple{AbstractFloat}
        picking_distribution = Multinomial(1, vec(picking_rule))
        return MCMCMultiProposalState(proposal_states, picking_distribution, 0)
    else
        return MCMCMultiProposalState(proposal_states, cumsum(picking_rule), 0)
    end
end
