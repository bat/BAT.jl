# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct MALA <: MCMCProposal

Metropolis adjusted Langevin sampling algorithm.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MALA{
    TA<:Real,
    TAI<:Tuple{Vararg{<:Real}},
    Q<:Union{
        AbstractMeasure,
        Distribution{<:Union{Univariate,Multivariate},Continuous}
    },
    R<:Real
} <: MCMCProposal
    # TODO: MD, is this correct?
    target_acceptance::TA = 0.23
    target_acceptance_int::TAI = (0.15, 0.35)
    proposaldist::Q = TDist(1.0)
    τ::R = 0.001
end

export MALA

struct MALAProposalState{
    TA<:Real,
    TAI<:Tuple{Vararg{<:Real}},
    Q<:BATMeasure,
    G<:Function,
    R<:Real
} <: MCMCProposalState
    target_acceptance::TA
    target_acceptance_int::TAI
    proposaldist::Q
    target_gradient::G
    τ::R
end

export MALAProposalState

bat_default(::Type{TransformedMCMC}, ::Val{:pretransform}, proposal::MALA) = PriorToNormal()

bat_default(::Type{TransformedMCMC}, ::Val{:proposal_tuning}, proposal::MALA) = NoMCMCProposalTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, proposal::MALA) = RAMTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:adaptive_transform}, proposal::MALA) = TriangularAffineTransform()

bat_default(::Type{TransformedMCMC}, ::Val{:tempering}, proposal::MALA) = NoMCMCTempering()

bat_default(::Type{TransformedMCMC}, ::Val{:nsteps}, proposal::MALA, pretransform::AbstractTransformTarget, nchains::Integer) = 10^5

bat_default(::Type{TransformedMCMC}, ::Val{:init}, proposal::MALA, pretransform::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCChainPoolInit(nsteps_init = max(div(nsteps, 100), 250))

bat_default(::Type{TransformedMCMC}, ::Val{:burnin}, proposal::MALA, pretransform::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCMultiCycleBurnin(nsteps_per_cycle = max(div(nsteps, 10), 2500))


function _create_proposal_state(
    proposal::MALA,
    target::BATMeasure,
    context::BATContext,
    v_init::AbstractVector{PV},
    f_transform::Function,
    rng::AbstractRNG
) where {P<:Real, PV<:AbstractVector{P}}
    n_dims = totalndof(varshape(target))
    mv_pdist = batmeasure(_full_random_walk_proposal(proposal.proposaldist, n_dims))

    adsel = get_adselector(context)
    target_checked = checked_logdensityof(MeasureBase.pullback(f_transform, target))
    target_gradient = valgrad_func(target_checked, adsel)

    return MALAProposalState(
        proposal.target_acceptance,
        proposal.target_acceptance_int,
        mv_pdist,
        target_gradient,
        proposal.τ
    )
end

function mcmc_propose!!(chain_state::MCMCChainState, proposal::MALAProposalState)
    @unpack target, f_transform, context = chain_state
    genctx = get_gencontext(context)
    rng = get_rng(genctx)
    proposal_measure = batmeasure(proposal.proposaldist)
    (; target_gradient, τ) = proposal

    n_walkers = nwalkers(chain_state)

    current_z = chain_state.current.z.v
    logd_z_current = chain_state.current.z.logd

    gradient_res = target_gradient.(current_z)
    grads = last.(gradient_res)

    z_proposed = current_z .+ τ .* grads .+ sqrt(2τ) .* rand(genctx, proposal_measure^n_walkers)

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

function set_proposal_transform!!(proposal::MALAProposalState, chain_state::MCMCChainState)
    f_transform_new = chain_state.f_transform
    adsel = get_adselector(chain_state.context)
    f = checked_logdensityof(MeasureBase.pullback(f_transform_new, chain_state.target))
    fg = valgrad_func(f, adsel)

    proposal_new = @set proposal.target_gradient = fg

    return proposal_new
end
