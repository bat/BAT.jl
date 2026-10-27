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
struct MCMCMultiProposal{
    P<:Tuple{Vararg{MCMCProposal}},
    R<:Union{Vector{<:Integer}, Categorical}
}<:MCMCProposal
    proposals::P
    picking_rule::R
end

function MCMCMultiProposal(
    proposals::Vector{<:MCMCProposal},
    picking_rule::Union{Vector{<:Integer}, Categorical}
)
    return MCMCMultiProposal(Tuple(proposals), picking_rule)
end

function MCMCMultiProposal(
    ; proposals::Union{Tuple{Vararg{<:MCMCProposal}}, Vector{<:MCMCProposal}} = (RandomWalk(),),
    picking_rule::Union{Nothing, Vector{<:Integer}, Categorical} = nothing
)
    picking_rule === nothing && (picking_rule = Categorical(fill(inv(length(proposals)), length(proposals))))
    return MCMCMultiProposal(proposals, picking_rule)
end

export MCMCMultiProposal

_contains_ensemble_move(::MCMCProposal) = false

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
    foreach(p -> _validate_mcmc_weighting_configuration(p, weighting), proposal.proposals)
    return nothing
end

function _validate_mcmc_adaptive_transform_configuration(
    proposal::MCMCMultiProposal,
    adaptive_transform::AbstractAdaptiveTransform,
)
    foreach(p -> _validate_mcmc_adaptive_transform_configuration(p, adaptive_transform), proposal.proposals)
    return nothing
end

function _validate_mcmc_transform_tuning_configuration(
    proposal::MCMCMultiProposal,
    tuning::MCMCTransformTuning,
)
    foreach(p -> _validate_mcmc_transform_tuning_configuration(p, tuning), proposal.proposals)
    return nothing
end

struct MultiProposalState{
    PS<:Tuple{Vararg{MCMCProposalState}},
    R<:Union{Vector{<:Integer}, Categorical},
    I<:Integer
}<:MCMCProposalState
    proposal_states::PS
    picking_rule::R
    active_idx::I
end

_contains_ensemble_move(proposal::MultiProposalState) =
    any(_contains_ensemble_move, proposal.proposal_states)

function _validate_mcmc_ensemble_invariants(
    proposal::MultiProposalState,
    target::BATMeasure,
    z_init::AbstractVector,
)
    foreach(proposal.proposal_states) do proposal_state
        _validate_mcmc_ensemble_invariants(proposal_state, target, z_init)
    end
    return nothing
end

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

bat_default(
    ::Type{TransformedMCMC},
    ::Val{:init},
    proposal::MCMCMultiProposal,
    ::TransformIntent,
    ::MCMCTransformTuning,
    ::Integer,
    ::Integer,
    nsteps::Integer,
) = _contains_ensemble_move(proposal) ?
    MCMCRetryInit() : MCMCChainPoolInit(nsteps_init = max(div(nsteps, 100), 250))

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
    store_burnin && return nothing
    return _pooled_ensemble_ess(chain_outputs, merged_output, context)
end

get_active_proposal_idx(proposal_state::MultiProposalState) = proposal_state.active_idx

_invalidate_position_cache!!(::MCMCProposalState) = nothing

function _invalidate_position_cache!!(proposal_state::MultiProposalState)
    foreach(_invalidate_position_cache!!, proposal_state.proposal_states)
    return nothing
end

function _activate_proposal!!(proposal_state::MultiProposalState, idx::Integer)
    # A re-entered proposal may have missed moves by the previously active
    # component; consecutive use retains its position-dependent cache.
    if idx != proposal_state.active_idx
        _invalidate_position_cache!!(proposal_state.proposal_states[idx])
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

get_active_proposal(proposal::MCMCProposalState, ::Val) = proposal
get_active_proposal(proposal::MultiProposalState, ::Val{I}) where {I} = proposal.proposal_states[I]

update_active_proposal!!(proposal::MCMCProposalState, active::MCMCProposalState, ::Val) = active
update_active_proposal!!(proposal::MultiProposalState, active::MCMCProposalState, ::Val{I}) where {I} =
    @set proposal.proposal_states[I] = active

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
        multi_proposal_state = @set multi_proposal_state.proposal_states[active_idx] = active_proposal_new
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
        proposal = multi_proposal.proposal_states[i]
        _contains_ensemble_move(proposal) && return true
        chain_state.nattempts[i] > 0 || return false
        return _acceptance_in_target(
            proposal, component_acceptance_rates[i],
        )
    end
end

function _create_proposal_state(
    multi_proposal::MCMCMultiProposal,
    target::BATMeasure,
    context::BATContext,
    v_init::AbstractVector{PV},
    z_init::AbstractVector,
    logd_z_init::Union{Nothing,AbstractVector},
    f_transform::Function,
    rng::AbstractRNG,
) where {P<:Real, PV<:AbstractVector{P}}

    nproposals = length(multi_proposal.proposals)
    nproposals <= _MCMC_PROPOSALS_PER_PURPOSE || throw(ArgumentError(
        "MCMCMultiProposal supports at most $_MCMC_PROPOSALS_PER_PURPOSE proposals, got $nproposals",
    ))

    proposal_states_init = map(multi_proposal.proposals) do proposal
        _create_proposal_state(
            proposal,
            target,
            context,
            v_init,
            z_init,
            logd_z_init,
            f_transform,
            rng,
        )
    end

    picking_rule = _copy_picking_rule(multi_proposal.picking_rule)

    idx = _init_active_idx(rng, picking_rule)

    return MultiProposalState(proposal_states_init, picking_rule, idx)
end

function _create_proposal_state(
    multi_proposal::MCMCMultiProposal, target::BATMeasure, context::BATContext,
    v_init::AbstractVector, z_init::AbstractVector, f_transform::Function, rng::AbstractRNG,
)
    return _create_proposal_state(
        multi_proposal, target, context, v_init, z_init, nothing, f_transform, rng,
    )
end

function _create_proposal_state(
    multi_proposal::MCMCMultiProposal,
    target::BATMeasure,
    context::BATContext,
    v_init::AbstractVector,
    f_transform::Function,
    rng::AbstractRNG,
)
    z_init = inverse(f_transform).(v_init)
    proposal_state = _create_proposal_state(
        multi_proposal, target, context, v_init, z_init, f_transform, rng,
    )
    _validate_mcmc_ensemble_invariants(proposal_state, target, z_init)
    return proposal_state
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

    proposals = set_proposal_transform!!.(multi_proposal.proposal_states, Ref(chain_state))
    return @set multi_proposal.proposal_states = proposals
end

# Keep the selected tuple slot static without specializing on the random index.
_with_proposal_index(f, proposals::Tuple, idx::Integer) =
    _with_proposal_index(f, ntuple(Val, Val(length(proposals))), idx)
@inline function _with_proposal_index(f::F, indices::Tuple{Vararg{Val}}, idx::Integer) where {F}
    idx == 1 && return f(first(indices))
    _with_proposal_index(f, Base.tail(indices), idx - 1)
end
_with_proposal_index(f, ::Tuple{}, idx::Integer) = throw(BoundsError())

function _mcmc_propose_and_tune!!(state, proposal::MultiProposalState, rngpart)
    _with_proposal_index(proposal.proposal_states, proposal.active_idx) do i
        _mcmc_propose_and_tune!!(state, proposal, rngpart, i)
    end
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
