# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct IndependentMH <: MCMCProposal

MCMC proposal algorithm for drawing samples from a global proposal
distribution - independent from the current position of the MCMC walker.

If no distribution is passed by the user, the target is checked for the 
best known approximation for the posterior, e.g. the prior.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct IndependentMH{
    TA<:Real,
    TAI<:Tuple{Vararg{<:Real}},
    Q<:Union{
        AbstractMeasure,
        Distribution{<:Union{Univariate,Multivariate},Continuous},
        Nothing
    },
} <: MCMCProposal
    # TODO: MD, review default values. The theoretical target accpeptance rate
    # seems to depend on the ratio of the proposal and target measure.
    target_acceptance::TA = 0.5
    target_acceptance_int::TAI = (0., 1.) # We don't want to punish low acceptance ratios
    global_proposal::Q = nothing
end

export IndependentMH

struct IndependentMHProposalState{
    TA<:Real,
    TAI<:Tuple{Vararg{<:Real}},
    Q<:BATMeasure,
} <: MCMCProposalState
    target_acceptance::TA
    target_acceptance_int::TAI
    global_proposal::Q
end

export IndependentMHProposalState

bat_default(::Type{TransformedMCMC}, ::Val{:pretransform}, proposal::IndependentMH) = PriorToNormal()

bat_default(::Type{TransformedMCMC}, ::Val{:proposal_tuning}, proposal::IndependentMH) = NoMCMCProposalTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, proposal::IndependentMH) = RAMTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:adaptive_transform}, proposal::IndependentMH) = TriangularAffineTransform()

bat_default(::Type{TransformedMCMC}, ::Val{:tempering}, proposal::IndependentMH) = NoMCMCTempering()

bat_default(::Type{TransformedMCMC}, ::Val{:nsteps}, proposal::IndependentMH, pretransform::AbstractTransformTarget, nchains::Integer) = 10^5

bat_default(::Type{TransformedMCMC}, ::Val{:init}, proposal::IndependentMH, pretransform::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCChainPoolInit(nsteps_init = max(div(nsteps, 100), 250))

bat_default(::Type{TransformedMCMC}, ::Val{:burnin}, proposal::IndependentMH, pretransform::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCMultiCycleBurnin(nsteps_per_cycle = max(div(nsteps, 10), 2500))


function _create_proposal_state(
    proposal::IndependentMH,
    target::BATMeasure,
    context::BATContext,
    v_init::AbstractVector{PV},
    f_transform::Function,
    rng::AbstractRNG
) where {P<:Real, PV<:AbstractVector{P}}
    if !isnothing(proposal.global_proposal)
        global_prop = batmeasure(proposal.global_proposal)
    elseif target isa BAT.PosteriorMeasure
        global_prop = target.prior
    else
        throw(ArgumentError("No adequate global proposal measure detected. Please supply one to IndependentMH() or make sure the sampling target supplies a suitable measure."))
    end

    return IndependentMHProposalState(
        proposal.target_acceptance,
        proposal.target_acceptance_int,
        global_prop
    )
end

function mcmc_propose!!(chain_state::MCMCChainState, proposal::IndependentMHProposalState)
    @unpack target, f_transform, context = chain_state
    genctx = get_gencontext(context)
    rng = get_rng(genctx)
    proposal_measure = batmeasure(proposal.global_proposal)

    n_walkers = nwalkers(chain_state)

    current_z = chain_state.current.z.v
    logd_z_current = chain_state.current.z.logd

    z_proposed = rand(genctx, proposal_measure^n_walkers)

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

set_proposal_transform!!(proposal::IndependentMHProposalState, ::MCMCChainState) = proposal
