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
    # TODO: MD, review these values
    target_acceptance::TA = 0.574
    target_acceptance_int::TAI = (0.5, 0.65)
    proposaldist::Q = TDist(1.0)
    τ_base::R = 1.65^2
end

export MALA

struct MALAProposalState{
    TA<:Real,
    TAI<:Tuple{Vararg{<:Real}},
    Q<:BATMeasure,
    G<:Function,
    R<:Real
} <: SimpleMCMCProposalState
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
        n_dims^(-1/3) * proposal.τ_base
    )
end

function mcmc_propose_transition(
    current_z::ArrayOfSimilarArrays,
    proposal::MALAProposalState,
    n_walkers::Integer,
    genctx
)
    proposal_measure = batmeasure(proposal.proposaldist)
    (; target_gradient, τ) = proposal

    gradient_res = target_gradient.(current_z)
    grads = last.(gradient_res)

    proposed_z = current_z .+ τ/2 .* grads .+ sqrt(τ) .* rand(genctx, proposal_measure^n_walkers)

    hastings_correction = checked_logdensityof.(proposal_measure, current_z) .- checked_logdensityof.(proposal_measure, proposed_z)
    return proposed_z, hastings_correction
end

function set_proposal_transform!!(proposal::MALAProposalState, chain_state::MCMCChainState)
    f_transform_new = chain_state.f_transform
    adsel = get_adselector(chain_state.context)
    f = checked_logdensityof(MeasureBase.pullback(f_transform_new, chain_state.target))
    fg = valgrad_func(f, adsel)

    proposal_new = @set proposal.target_gradient = fg

    return proposal_new
end
