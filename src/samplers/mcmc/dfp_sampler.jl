# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct DrawFromPrior <: MCMCProposal

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct DrawFromPrior{
    TA<:Real,
    TAI<:Tuple{Vararg{<:Real}},
    Q<:Union{
        AbstractMeasure,
        Distribution{<:Union{Univariate,Multivariate},Continuous}
    },
} <: MCMCProposal
    # TODO: MD, is this correct?
    target_acceptance::TA = 0.23
    target_acceptance_int::TAI = (0.15, 0.35)
    prior::Q = TDist(1.0)
end

export DrawFromPrior

struct DFPProposalState{
    TA<:Real,
    TAI<:Tuple{Vararg{<:Real}},
    Q<:BATMeasure,
} <: MCMCProposalState
    target_acceptance::TA
    target_acceptance_int::TAI
    prior::Q
end

export DFPProposalState

bat_default(::Type{TransformedMCMC}, ::Val{:pretransform}, proposal::DrawFromPrior) = PriorToNormal()

bat_default(::Type{TransformedMCMC}, ::Val{:proposal_tuning}, proposal::DrawFromPrior) = NoMCMCProposalTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, proposal::DrawFromPrior) = RAMTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:adaptive_transform}, proposal::DrawFromPrior) = TriangularAffineTransform()

bat_default(::Type{TransformedMCMC}, ::Val{:tempering}, proposal::DrawFromPrior) = NoMCMCTempering()

bat_default(::Type{TransformedMCMC}, ::Val{:nsteps}, proposal::DrawFromPrior, pretransform::AbstractTransformTarget, nchains::Integer) = 10^5

bat_default(::Type{TransformedMCMC}, ::Val{:init}, proposal::DrawFromPrior, pretransform::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCChainPoolInit(nsteps_init = max(div(nsteps, 100), 250))

bat_default(::Type{TransformedMCMC}, ::Val{:burnin}, proposal::DrawFromPrior, pretransform::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCMultiCycleBurnin(nsteps_per_cycle = max(div(nsteps, 10), 2500))


function _create_proposal_state(
    proposal::DrawFromPrior,
    target::BATMeasure,
    context::BATContext,
    v_init::AbstractVector{PV},
    f_transform::Function,
    rng::AbstractRNG
) where {P<:Real, PV<:AbstractVector{P}}
    if target isa BAT.PosteriorMeasure
        prior = target.prior
    else
        n_dims = totalndof(varshape(target))
        prior = batmeasure(_full_random_walk_proposal(proposal.prior, n_dims))
    end

    return DFPProposalState(
        proposal.target_acceptance,
        proposal.target_acceptance_int,
        prior
    )
end

function mcmc_propose!!(chain_state::MCMCChainState, proposal::DFPProposalState)
    @unpack target, f_transform, context = chain_state
    genctx = get_gencontext(context)
    rng = get_rng(genctx)
    prior_measure = batmeasure(proposal.prior)

    n_walkers = nwalkers(chain_state)

    current_z = chain_state.current.z.v
    logd_z_current = chain_state.current.z.logd

    z_proposed = rand(genctx, prior_measure^n_walkers)

    x_ladj_proposed = with_logabsdet_jacobian.(f_transform, z_proposed)
    x_proposed = first.(x_ladj_proposed)
    ladj = getsecond.(x_ladj_proposed)

    logd_x_proposed = BAT.checked_logdensityof.(target, x_proposed)
    logd_z_proposed::typeof(logd_x_proposed) = logd_x_proposed .+ ladj

    chain_state.proposed.x.v .= x_proposed
    chain_state.proposed.z.v .= z_proposed

    chain_state.proposed.x.logd .= logd_x_proposed
    chain_state.proposed.z.logd .= logd_z_proposed

    p_accept = clamp.(exp.(logd_z_proposed - logd_z_current), 0, 1)
    @assert all(p_accept .>= 0)
    accepted = rand(rng, length(p_accept)) .<= p_accept

    chain_state.accepted .= accepted

    return chain_state, p_accept
end

set_proposal_transform!!(proposal::DFPProposalState, ::MCMCChainState) = proposal
