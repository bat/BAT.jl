# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    abstract type MHProposalDistTuning

Abstract type for Metropolis-Hastings tuning strategies for
proposal distributions.
"""
abstract type MHProposalDistTuning <: MCMCTuning end
export MHProposalDistTuning


"""
    struct MetropolisHastings <: MCMCAlgorithm

Metropolis-Hastings MCMC sampling algorithm.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MetropolisHastings{
    Q<:ContinuousDistribution,
    WS<:AbstractMCMCWeightingScheme,
} <: MCMCProposal
    proposaldist::Q = TDist(1.0)
    weighting::WS = RepetitionWeighting()
end

export MetropolisHastings

mutable struct MHProposalState{
    Q<:ContinuousDistribution,
    WS<:AbstractMCMCWeightingScheme,
} <: MCMCProposalState
    proposaldist::Q
    weighting::WS
end
export MHProposalState

function _create_proposal_state(
    proposal::MetropolisHastings, 
    target::BATMeasure, 
    context::BATContext, 
    v_init::AbstractVector{<:Real}, 
    rng::AbstractRNG
)
    return MHProposalState(proposal.proposaldist, proposal.weighting)
end


function _get_sample_id(proposal::MHProposalState, id::Int32, cycle::Int32, stepno::Int64, sample_type::Integer)
    return MCMCSampleID(id, cycle, stepno, sample_type), MCMCSampleID
end


const MHState = MCMCState{<:BATMeasure,
                          <:RNGPartition,
                          <:Function,
                          <:MHProposalState,
                          <:DensitySampleVector,
                          <:DensitySampleVector,
                          <:BATContext
} 

# TODO: MD, should this be a !! function?  
function mcmc_propose!!(mc_state::MHState)
    @unpack target, proposal, f_transform, context = mc_state
    rng = get_rng(context)

    proposed_x_idx = _proposed_sample_idx(mc_state)

    sample_z_current = current_sample_z(mc_state)

    z_current, logd_z_current = sample_z_current.v, sample_z_current.logd

    n_dims = size(z_current, 1)
    z_proposed = z_current + rand(rng, proposal.proposaldist, n_dims) #TODO: check if proposal is symmetric? otherwise need additional factor?
    x_proposed, ladj = with_logabsdet_jacobian(f_transform, z_proposed)
    logd_x_proposed = BAT.checked_logdensityof(target, x_proposed)
    logd_z_proposed = logd_x_proposed + ladj

    @assert logd_z_proposed ≈ logdensityof(MeasureBase.pullback(f_transform, target), z_proposed) #TODO: MD, Remove, only for debugging

    mc_state.samples[proposed_x_idx] = DensitySample(x_proposed, logd_x_proposed, 0, _get_sample_id(proposal, mc_state.info.id, mc_state.info.cycle, mc_state.stepno, PROPOSED_SAMPLE)[1], nothing)
    mc_state.sample_z[2] = DensitySample(z_proposed, logd_z_proposed, 0, _get_sample_id(proposal, mc_state.info.id, mc_state.info.cycle, mc_state.stepno, PROPOSED_SAMPLE)[1], nothing)

    # TODO: MD, should we check for symmetriy of proposal distribution?
    p_accept = clamp(exp(logd_z_proposed - logd_z_current), 0, 1)
    @assert p_accept >= 0

    accepted = rand(rng) <= p_accept

    return mc_state, accepted, p_accept
end

function _accept_reject!(mc_state::MHState, accepted::Bool, p_accept::Float64, current::Integer, proposed::Integer)
    @unpack samples, proposal = mc_state

    if accepted
        samples.info.sampletype[current] = ACCEPTED_SAMPLE
        samples.info.sampletype[proposed] = CURRENT_SAMPLE
        
        mc_state.nsamples += 1

        mc_state.sample_z[1] = deepcopy(proposed_sample_z(mc_state))
    else
        samples.info.sampletype[proposed] = REJECTED_SAMPLE
    end

    delta_w_current, w_proposed = _weights(proposal, p_accept, accepted)
    samples.weight[current] += delta_w_current
    samples.weight[proposed] = w_proposed
end

function _weights(
    proposal::MHProposalState{Q,<:RepetitionWeighting},
    p_accept::Real,
    accepted::Bool
) where Q
    if accepted
        (0, 1)
    else
        (1, 0)
    end
end

function _weights(
    proposal::MHProposalState{Q,<:ARPWeighting},
    p_accept::Real,
    accepted::Bool
) where Q
    T = typeof(p_accept)
    if p_accept ≈ 1
        (zero(T), one(T))
    elseif p_accept ≈ 0
        (one(T), zero(T))
    else
        (T(1 - p_accept), p_accept)
    end
end

eff_acceptance_ratio(mc_state::MHState) = nsamples(mc_state) / nsteps(mc_state)
