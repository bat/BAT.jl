# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct RandomWalk <: MCMCAlgorithm

Metropolis-Hastings MCMC sampling algorithm.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct RandomWalk{Q<:Union{AbstractMeasure,Distribution{<:Union{Univariate,Multivariate},Continuous}}} <: MCMCProposal
    proposaldist::Q = TDist(1.0)
end

export RandomWalk

struct MHProposalState{Q<:BATMeasure} <: MCMCProposalState
    proposaldist::Q
end
export MHProposalState

bat_default(::Type{TransformedMCMC}, ::Val{:pretransform}, proposal::RandomWalk) = PriorToNormal()

bat_default(::Type{TransformedMCMC}, ::Val{:proposal_tuning}, proposal::RandomWalk) = NoMCMCProposalTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, proposal::RandomWalk) = RAMTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:adaptive_transform}, proposal::RandomWalk) = TriangularAffineTransform()

bat_default(::Type{TransformedMCMC}, ::Val{:tempering}, proposal::RandomWalk) = NoMCMCTempering()

bat_default(::Type{TransformedMCMC}, ::Val{:nsteps}, proposal::RandomWalk, pretransform::AbstractTransformTarget, nchains::Integer) = 10^5

bat_default(::Type{TransformedMCMC}, ::Val{:init}, proposal::RandomWalk, pretransform::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCChainPoolInit(nsteps_init = max(div(nsteps, 100), 250))

bat_default(::Type{TransformedMCMC}, ::Val{:burnin}, proposal::RandomWalk, pretransform::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCMultiCycleBurnin(nsteps_per_cycle = max(div(nsteps, 10), 2500))


function _get_sample_id(proposal::MHProposalState, id::Int32, cycle::Int32, stepno::Integer, sample_type::Integer)
    return MCMCSampleID(id, cycle, stepno, sample_type), MCMCSampleID
end


function _create_proposal_state(
    proposal::RandomWalk, 
    target::BATMeasure, 
    context::BATContext, 
    v_init::AbstractVector{<:Real},
    f_transform::Function,
    rng::AbstractRNG
)
    n_dims = length(v_init)
    mv_pdist = batmeasure(_full_random_walk_proposal(proposal.proposaldist, n_dims))
    return MHProposalState(mv_pdist)
end


function _full_random_walk_proposal(m::AbstractMeasure, n_dims::Integer)
    x = testvalue(m)
    @argcheck x isa AbstractVector{<:Real} && length(x) == n_dims
    return m
end

function _full_random_walk_proposal(m::BATDistMeasure, n_dims::Integer)
    d = convert(Distribution, m)
    return batmeasure(_full_random_walk_proposal(d, n_dims))
end

function _full_random_walk_proposal(d::Distribution{Multivariate,Continuous}, n_dims::Integer)
    @assert false
    @argcheck length(d) == n_dims
    return d
end

function _full_random_walk_proposal(d::Normal, n_dims::Integer)
    # Theoretical optimally proposal scale for random walk with gaussian proposal, according to
    # [Gelman et al., Ann. Appl. Probab. 7 (1) 110 - 120, 1997](https://doi.org/10.1214/aoap/1034625254):
    proposal_scale = 2.38 / sqrt(n_dims)

    @argcheck mean(d) ≈ 0 
    σ² = var(d)
    Σ = ScalMat(n_dims, proposal_scale^2 * σ²)
    return MvNormal(Σ)
end

function _full_random_walk_proposal(d::TDist, n_dims::Integer)
    # Theoretically optimal proposal scale for gaussian seems to work quite well for
    # t-distribution proposals with any degrees of freedom as well:
    proposal_scale = 2.38 / sqrt(n_dims)

    ν = dof(d)
    Σ = ScalMat(n_dims, proposal_scale^2)
    return Distributions.IsoTDist(ν, Σ)
end


const MHChainState = MCMCChainState{<:BATMeasure, <:RNGPartition, <:Function, <:MHProposalState} 

function mcmc_propose!!(mc_state::MHChainState)
    @unpack target, proposal, f_transform, context = mc_state
    rng = get_rng(context)
    pdist = proposal.proposaldist

    proposed_x_idx = _proposed_sample_idx(mc_state)

    sample_z_current = current_sample_z(mc_state)

    z_current, logd_z_current = sample_z_current.v, sample_z_current.logd
    T = eltype(z_current)

    # ToDo: Use gen-context:
    z_proposed = z_current + T.(rand(rng, pdist))
    x_proposed, ladj = with_logabsdet_jacobian(f_transform, z_proposed)
    logd_x_proposed = BAT.checked_logdensityof(target, x_proposed)
    logd_z_proposed = logd_x_proposed + ladj

    @assert logd_z_proposed ≈ logdensityof(MeasureBase.pullback(f_transform, target), z_proposed) #TODO: MD, Remove, only for debugging

    mc_state.samples[proposed_x_idx] = DensitySample(x_proposed, logd_x_proposed, 0, _get_sample_id(proposal, mc_state.info.id, mc_state.info.cycle, mc_state.stepno, PROPOSED_SAMPLE)[1], nothing)
    mc_state.sample_z[2] = DensitySample(z_proposed, logd_z_proposed, 0, _get_sample_id(proposal, mc_state.info.id, mc_state.info.cycle, mc_state.stepno, PROPOSED_SAMPLE)[1], nothing)

    # TODO: check if proposal is symmetric - otherwise need Hastings correction:
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
    else
        samples.info.sampletype[proposed] = REJECTED_SAMPLE
    end

    delta_w_current, w_proposed = mcmc_weight_values(mc_state.weighting, p_accept, accepted)
    samples.weight[current] += delta_w_current
    samples.weight[proposed] = w_proposed
end


eff_acceptance_ratio(mc_state::MHChainState) = nsamples(mc_state) / nsteps(mc_state)

function set_mc_state_transform!!(mc_state::MHChainState, f_transform_new::Function) 
    mc_state_new = @set mc_state.f_transform = f_transform_new
    return mc_state_new
end
