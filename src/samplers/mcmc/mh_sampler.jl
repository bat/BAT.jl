# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct RandomWalk <: MCMCAlgorithm

Metropolis-Hastings MCMC sampling algorithm.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct RandomWalk{Q<:ContinuousUnivariateDistribution} <: MCMCProposal
    proposaldist::Q = TDist(1.0)
end

export RandomWalk

struct MHProposalState{Q<:ContinuousUnivariateDistribution} <: MCMCProposalState
    proposaldist::Q
end
export MHProposalState

bat_default(::Type{MCMCSampling}, ::Val{:pre_transform}, proposal::RandomWalk) = PriorToGaussian()

bat_default(::Type{MCMCSampling}, ::Val{:proposal_tuning}, proposal::RandomWalk) = NoMCMCProposalTuning()

bat_default(::Type{MCMCSampling}, ::Val{:transform_tuning}, proposal::RandomWalk) = RAMTuning()

bat_default(::Type{MCMCSampling}, ::Val{:adaptive_transform}, proposal::RandomWalk) = TriangularAffineTransform()

bat_default(::Type{MCMCSampling}, ::Val{:tempering}, proposal::RandomWalk) = NoMCMCTempering()

bat_default(::Type{MCMCSampling}, ::Val{:nsteps}, proposal::RandomWalk, pre_transform::AbstractTransformTarget, nchains::Integer) = 10^5

bat_default(::Type{MCMCSampling}, ::Val{:init}, proposal::RandomWalk, pre_transform::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCChainPoolInit(nsteps_init = max(div(nsteps, 100), 250))

bat_default(::Type{MCMCSampling}, ::Val{:burnin}, proposal::RandomWalk, pre_transform::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCMultiCycleBurnin(nsteps_per_cycle = max(div(nsteps, 10), 2500))


function _create_proposal_state(
    proposal::RandomWalk, 
    target::BATMeasure, 
    context::BATContext, 
    v_init::AbstractVector{<:Real}, 
    rng::AbstractRNG
)
    return MHProposalState(proposal.proposaldist)
end


function _get_sample_id(proposal::MHProposalState, id::Int32, cycle::Int32, stepno::Integer, sample_type::Integer)
    return MCMCSampleID(id, cycle, stepno, sample_type), MCMCSampleID
end


const MHChainState = MCMCChainState{<:BATMeasure, <:RNGPartition, <:Function, <:MHProposalState} 

function mcmc_propose!!(mc_state::MHChainState)
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


function _accept_reject!(mc_state::MHChainState, accepted::Bool, p_accept::Float64, current::Integer, proposed::Integer)
    @unpack samples, proposal = mc_state

    if accepted
        samples.info.sampletype[current] = ACCEPTED_SAMPLE
        samples.info.sampletype[proposed] = CURRENT_SAMPLE
        
        mc_state.nsamples += 1

        mc_state.sample_z[1] = deepcopy(proposed_sample_z(mc_state))
    else
        samples.info.sampletype[proposed] = REJECTED_SAMPLE
    end

    delta_w_current, w_proposed = mcmc_weight_values(mc_state.weighting, p_accept, accepted)
    samples.weight[current] += delta_w_current
    samples.weight[proposed] = w_proposed
end


eff_acceptance_ratio(mc_state::MHChainState) = nsamples(mc_state) / nsteps(mc_state)
