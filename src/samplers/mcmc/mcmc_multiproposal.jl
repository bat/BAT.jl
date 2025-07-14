# This file is a part of BAT.jl, licensed under the MIT License (MIT).

struct MCMCMultiProposal{
    P<:Tuple{Vararg{MCMCProposal}},
    S<:Tuple{Vararg{Union{Integer, AbstractFloat}}}
}<:MCMCProposal
    proposals::P
    picking_rule::S
end

export MCMCMultiProposal

struct SequentialMultiProposalState{
    PS<:Tuple{Vararg{MCMCProposalState}},
    S<:Tuple{Vararg{Integer}},
    I<:Integer
}<:MCMCProposalState
    proposal_states::PS
    schedule::S
    current_idx::I
end

export SequentialMultiProposalState

struct RandomMultiProposalState{
    PS<:Tuple{Vararg{MCMCProposalState}},
    D<:Multinomial,
    I<:Integer
}<:MCMCProposalState
    proposal_states::PS
    picking_distribution::D
    current_idx::I
end

export RandomMultiProposalState

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
    idx = findfirst(0 .< rand(rng, proposal_state.picking_distribution))

    proposal_state = @set proposal_state.current_idx = idx
    return proposal_state
end

function get_current_proposal(
    multi_proposal_state::Union{SequentialMultiProposalState, RandomMultiProposalState}
)
    current_proposal = multi_proposal_state.proposal_states[multi_proposal_state.current_idx]
    return current_proposal
end

function _get_sample_id(
    multi_proposal_state::Union{SequentialMultiProposalState, RandomMultiProposalState},
    chainid::Int32, 
    walkerid::Int32, 
    cycle::Int32, 
    stepno::Integer, 
    sample_type::Integer
)
    current_proposal = get_current_proposal(multi_proposal_state)
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

    if picking_rule isa Tuple{Vararg{AbstractFloat}}
        @assert sum(picking_rule) == 1 throw(ArgumentError("Picking probabilities for proposals do not add up to 1."))

        picking_distribution = Multinomial(1, collect(picking_rule))
        current_proposal_idx = findfirst(0 .< rand(rng, picking_distribution))
        return RandomMultiProposalState(proposal_states, picking_distribution, current_proposal_idx)
    else
        current_proposal_idx = 1
        return SequentialMultiProposalState(proposal_states, cumsum(picking_rule), current_proposal_idx)
    end
end
