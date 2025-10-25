# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct RandomWalk <: MCMCProposal

Metropolis-Hastings MCMC sampling algorithm.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct RandomWalk{
    TA<:Real,
    TAI<:Tuple{Vararg{<:Real}},
    Q<:Union{
        AbstractMeasure,
        Distribution{<:Union{Univariate,Multivariate},Continuous}
    }
} <: MCMCProposal
    # TODO: MD, is this correct?
    target_acceptance::TA = 0.234
    target_acceptance_int::TAI = (0.15, 0.35)
    proposaldist::Q = TDist(1.0)
end

export RandomWalk

struct MHProposalState{
    TA<:Real,
    TAI<:Tuple{Vararg{<:Real}},
    Q<:BATMeasure
} <: MCMCProposalState
    target_acceptance::TA
    target_acceptance_int::TAI
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


function _create_proposal_state(
    proposal::RandomWalk, 
    target::BATMeasure, 
    context::BATContext, 
    v_init::AbstractVector{PV},
    f_transform::Function,
    rng::AbstractRNG
) where {P<:Real, PV<:AbstractVector{P}}
    n_dims = totalndof(varshape(target))
    mv_pdist = batmeasure(_full_random_walk_proposal(proposal.proposaldist, n_dims))
    return MHProposalState(
        proposal.target_acceptance,
        proposal.target_acceptance_int,
        mv_pdist
    )
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


function mcmc_propose!!(chain_state::MCMCChainState, proposal::MHProposalState)
    @unpack target, f_transform, context = chain_state
    genctx = get_gencontext(context)
    rng = get_rng(genctx)
    proposal_measure = batmeasure(proposal.proposaldist)
    n_walkers = nwalkers(chain_state)

    current_z = chain_state.current.z.v
    logd_z_current = chain_state.current.z.logd

    z_proposed = current_z .+ rand(genctx, proposal_measure^n_walkers)

    x_ladj_proposed = with_logabsdet_jacobian.(f_transform, z_proposed)
    x_proposed = first.(x_ladj_proposed)
    ladj = getsecond.(x_ladj_proposed)

    logd_x_proposed = BAT.checked_logdensityof.(target, x_proposed)
    logd_z_proposed::typeof(logd_x_proposed) = logd_x_proposed .+ ladj

    chain_state.proposed.x.v .= x_proposed
    chain_state.proposed.z.v .= z_proposed

    chain_state.proposed.x.logd .= logd_x_proposed
    chain_state.proposed.z.logd .= logd_z_proposed

    # TODO: check if proposal is symmetric - otherwise need Hastings correction:
    p_accept = clamp.(exp.(logd_z_proposed - logd_z_current), 0, 1)
    @assert all(p_accept .>= 0)
    accepted = rand(rng, length(p_accept)) .<= p_accept

    chain_state.accepted .= accepted

    return chain_state, p_accept
end

set_proposal_transform!!(proposal::MHProposalState, ::MCMCChainState) = proposal

