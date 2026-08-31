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
    proposals::P = MCMCProposal[RandomWalk(), HamiltonianMC()]
    picking_rule::R = Categorical(1/length(proposals) .* ones(length(proposals)))
end

export MCMCMultiProposal

_contains_ensemble_move(proposal::MCMCProposal) =
    proposal isa Union{StretchMove,DEMove,DESnookerMove}

_contains_ensemble_move(::MCMCProposalState) = false

_contains_ensemble_move(proposal::MCMCMultiProposal) =
    any(_contains_ensemble_move, proposal.proposals)

function _validate_mcmc_proposal_configuration(
    multi_proposal::MCMCMultiProposal,
    tuning::MCMCProposalTuning,
)
    proposals = multi_proposal.proposals
    picking_rule = multi_proposal.picking_rule
    n_proposals = length(proposals)

    n_proposals > 0 || throw(ArgumentError(
        "MCMCMultiProposal requires at least one component proposal",
    ))
    any(proposal -> proposal isa MCMCMultiProposal, proposals) && throw(ArgumentError(
        "Nested MCMCMultiProposal components are not supported",
    ))

    n_rule_components = picking_rule isa Categorical ?
        length(picking_rule.p) : length(picking_rule)
    n_rule_components == n_proposals || throw(ArgumentError(
        "MCMCMultiProposal has $n_proposals component proposals but its picking rule has $n_rule_components components",
    ))

    if picking_rule isa AbstractVector
        all(weight -> weight >= 0, picking_rule) || throw(ArgumentError(
            "MCMCMultiProposal integer picking-rule weights must be nonnegative",
        ))
        any(weight -> weight > 0, picking_rule) || throw(ArgumentError(
            "MCMCMultiProposal integer picking-rule weights must have positive mass",
        ))
        issorted(cumsum(picking_rule)) || throw(ArgumentError(
            "MCMCMultiProposal integer picking-rule cumulative mass overflows",
        ))
    end

    _validate_mcmc_proposal_tuning_configuration(multi_proposal, tuning)
    return nothing
end

function _validate_mcmc_weighting_configuration(
    proposal::MCMCMultiProposal,
    weighting::AbstractMCMCWeightingScheme,
)
    _contains_ensemble_move(proposal) || return nothing
    weighting isa RepetitionWeighting || throw(ArgumentError(
        "MCMCMultiProposal with an ensemble move requires RepetitionWeighting, got $(nameof(typeof(weighting)))",
    ))
    return nothing
end

function _validate_mcmc_adaptive_transform_configuration(
    proposal::MCMCMultiProposal,
    adaptive_transform::AbstractAdaptiveTransform,
)
    _contains_ensemble_move(proposal) || return nothing
    adaptive_transform isa NoAdaptiveTransform || throw(ArgumentError(
        "MCMCMultiProposal with an ensemble move requires NoAdaptiveTransform, got $(nameof(typeof(adaptive_transform)))",
    ))
    return nothing
end

function _validate_mcmc_transform_tuning_configuration(
    proposal::MCMCMultiProposal,
    tuning::MCMCTransformTuning,
)
    _contains_ensemble_move(proposal) || return nothing
    tuning isa NoMCMCTransformTuning || throw(ArgumentError(
        "MCMCMultiProposal with an ensemble move requires NoMCMCTransformTuning, got $(nameof(typeof(tuning)))",
    ))
    return nothing
end

struct MultiProposalState{
    PS<:Vector{<:MCMCProposalState},
    R<:Union{Vector{<:Integer}, Categorical},
    I<:Integer
}<:MCMCProposalState
    proposal_states::PS
    picking_rule::R
    active_idx::I
end

_contains_ensemble_move(proposal::MultiProposalState) =
    any(_contains_ensemble_move, proposal.proposal_states)

_mcmc_n_rng_purposes(proposal::MultiProposalState) =
    _contains_ensemble_move(proposal) ? _MCMC_N_RNG_PURPOSES : _MCMC_ACCEPTANCE_PURPOSE


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
) = _contains_ensemble_move(proposal) ? NoAdaptiveTransform() : TriangularAffineTransform()

bat_default(
    ::Type{TransformedMCMC}, 
    ::Val{:tempering}, 
    proposal::MCMCMultiProposal
) = NoMCMCTempering()

function _mcmc_ess(
    chain_outputs::AbstractVector{<:AbstractVector{<:DensitySampleVector}},
    merged_output::DensitySampleVector,
    proposal::MCMCMultiProposal,
    weighting::AbstractMCMCWeightingScheme,
    store_burnin::Bool,
    context::BATContext,
)
    _contains_ensemble_move(proposal) || return _pooled_walker_ess(
        chain_outputs, merged_output, weighting, context,
    )
    _validate_mcmc_weighting_configuration(proposal, weighting)
    store_burnin && return nothing
    return _pooled_ensemble_ess(chain_outputs, merged_output, context)
end

get_active_proposal_idx(proposal_state::MultiProposalState) = proposal_state.active_idx

_invalidate_mala_cache!!(::MCMCProposalState) = nothing

function _invalidate_mala_cache!!(proposal_state::MultiProposalState)
    foreach(_invalidate_mala_cache!!, proposal_state.proposal_states)
    return nothing
end

function _activate_proposal!!(proposal_state::MultiProposalState, idx::Integer)
    # A re-entered MALA may have missed moves by the previously active
    # component; consecutive use retains its promoted gradient.
    if idx != proposal_state.active_idx
        _invalidate_mala_cache!!(proposal_state.proposal_states[idx])
    end
    proposal_state_new = @set proposal_state.active_idx = idx
    return proposal_state_new, get_active_proposal(proposal_state_new)
end

function next_proposal!!(
    rng::AbstractRNG,
    proposal_state::MultiProposalState{<:Any, <:Vector}, 
    stepno::Integer
)
    picking_rule_cum = cumsum(proposal_state.picking_rule)
    m = mod1(stepno, last(picking_rule_cum))
    idx = findfirst(y -> m <= y, picking_rule_cum)
    return _activate_proposal!!(proposal_state, idx)
end

function next_proposal!!(
    rng::AbstractRNG,
    proposal_state::MultiProposalState{<:Any, <:Distribution}, 
    stepno::Integer
)
    idx = rand(rng, proposal_state.picking_rule)
    return _activate_proposal!!(proposal_state, idx)
end

function get_active_proposal(
    multi_proposal_state::MultiProposalState
)
    current_proposal = multi_proposal_state.proposal_states[multi_proposal_state.active_idx]
    return current_proposal
end

function mcmc_mark_warmup_end!(multi_proposal_state::MultiProposalState)
    foreach(mcmc_mark_warmup_end!, multi_proposal_state.proposal_states)
    return nothing
end

function update_active_proposal!!(
    multi_proposal_state::MultiProposalState,
    active_proposal_new::MCMCProposalState
)
    active_idx = get_active_proposal_idx(multi_proposal_state)
    active_proposal = multi_proposal_state.proposal_states[active_idx]

    if active_proposal !== active_proposal_new
        multi_proposal_state.proposal_states[active_idx] = active_proposal_new
    end
    return multi_proposal_state
end

function get_target_acceptance_ratio(proposal::MultiProposalState)
    target_acc_ratios = Tuple(get_target_acceptance_ratio.(proposal.proposal_states))
    picking_rule = proposal.picking_rule
    proposal_probs = _get_proposal_picking_probabilities(picking_rule)
    return dot(target_acc_ratios, proposal_probs)
end


function get_target_acceptance_int(proposal::MultiProposalState)
    target_acc_ints = Tuple(get_target_acceptance_int.(proposal.proposal_states))
    picking_rule = proposal.picking_rule

    lowers = first.(target_acc_ints)
    uppers = last.(target_acc_ints)

    proposal_probs = _get_proposal_picking_probabilities(picking_rule)

    mean_target_acc_int = (dot(lowers, proposal_probs), dot(uppers, proposal_probs))
    return mean_target_acc_int
end

function _get_proposal_picking_probabilities(picking_rule::Distribution)
    return picking_rule.p
end

function _get_proposal_picking_probabilities(picking_rule::Vector)
    return picking_rule ./ sum(picking_rule) 
end

get_tuning_success(
    chain_state::MCMCChainState,
    multi_proposal::MultiProposalState,
) = all(_component_acceptance_successes(chain_state, multi_proposal))

function _component_acceptance_successes(
    chain_state::MCMCChainState,
    multi_proposal::MultiProposalState,
)
    component_acceptance_rates = detailed_eff_acceptance_ratio(chain_state)
    return map(eachindex(component_acceptance_rates)) do i
        chain_state.nattempts[i] > 0 || return false
        return _acceptance_in_target(
            multi_proposal.proposal_states[i], component_acceptance_rates[i],
        )
    end
end

function _create_proposal_state(
    multi_proposal::MCMCMultiProposal, 
    target::BATMeasure, 
    context::BATContext, 
    v_init::AbstractVector{PV},
    f_transform::Function,
    rng::AbstractRNG
) where {P<:Real, PV<:AbstractVector{P}}

    nproposals = length(multi_proposal.proposals)
    nproposals <= _MCMC_PROPOSALS_PER_PURPOSE || throw(ArgumentError(
        "MCMCMultiProposal supports at most $_MCMC_PROPOSALS_PER_PURPOSE proposals, got $nproposals",
    ))

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

    picking_rule = _copy_picking_rule(multi_proposal.picking_rule)

    idx = _init_active_idx(rng, picking_rule)

    return MultiProposalState(proposal_states_init, picking_rule, idx)
end

_copy_picking_rule(picking_rule::AbstractVector) = copy(picking_rule)
_copy_picking_rule(picking_rule::Categorical) = Categorical(copy(picking_rule.p))

function _init_active_idx(rng::AbstractRNG, picking_rule::Distribution)
    return rand(rng, picking_rule)
end

function _init_active_idx(rng::AbstractRNG, picking_rule::Vector)
    return 1
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

function _proposal_diagnostics(
    multi_proposal::MultiProposalState,
    chain_state::MCMCChainState,
)
    # Chain counters reset at each cycle; nested proposal diagnostics may span
    # the whole run, so keep the counter scope explicit in the field names.
    component_rates = detailed_eff_acceptance_ratio(chain_state)
    components = map(eachindex(multi_proposal.proposal_states)) do i
        proposal = multi_proposal.proposal_states[i]
        return (
            index = i,
            proposal_type = nameof(typeof(proposal)),
            cycle_n_attempts = chain_state.nattempts[i],
            cycle_n_accepted = chain_state.nsamples[i],
            cycle_acceptance_rate = component_rates[i],
            diagnostics = _proposal_diagnostics(proposal),
        )
    end
    return (; components)
end
