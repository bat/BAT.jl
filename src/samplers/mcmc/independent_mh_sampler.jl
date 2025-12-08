# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct MCMCGlobalProposal <: MCMCProposal

MCMC proposal algorithm for drawing samples from a global proposal
distribution - independent from the current position of the MCMC walker.

If no distribution is passed by the user, the target is checked for the 
best known approximation for the posterior, e.g. the prior.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MCMCGlobalProposal{
    TA<:Real,
    TAI<:Tuple{Vararg{<:Real}},
    Q<:Union{
        AbstractMeasure,
        Distribution{<:Union{Univariate,Multivariate},Continuous},
        Nothing
    },
} <: MCMCProposal
    target_acceptance::TA = 1.0
    target_acceptance_int::TAI = (0.01, 1.) # We don't want to punish low acceptance ratios, but kick out if it doesnt perform at all.
    global_proposal::Q = nothing
end

export MCMCGlobalProposal

struct MCMCGlobalProposalProposalState{
    TA<:Real,
    TAI<:Tuple{Vararg{<:Real}},
    Q<:BATMeasure,
} <: SimpleMCMCProposalState
    target_acceptance::TA
    target_acceptance_int::TAI
    global_proposal::Q
end

export MCMCGlobalProposalProposalState

bat_default(::Type{TransformedMCMC}, ::Val{:pretransform}, proposal::MCMCGlobalProposal) = PriorToNormal()

bat_default(::Type{TransformedMCMC}, ::Val{:proposal_tuning}, proposal::MCMCGlobalProposal) = NoMCMCProposalTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, proposal::MCMCGlobalProposal) = RAMTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:adaptive_transform}, proposal::MCMCGlobalProposal) = TriangularAffineTransform()

bat_default(::Type{TransformedMCMC}, ::Val{:tempering}, proposal::MCMCGlobalProposal) = NoMCMCTempering()

bat_default(::Type{TransformedMCMC}, ::Val{:nsteps}, proposal::MCMCGlobalProposal, pretransform::AbstractTransformTarget, nchains::Integer) = 10^5

bat_default(::Type{TransformedMCMC}, ::Val{:init}, proposal::MCMCGlobalProposal, pretransform::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCChainPoolInit(nsteps_init = max(div(nsteps, 100), 250))

bat_default(::Type{TransformedMCMC}, ::Val{:burnin}, proposal::MCMCGlobalProposal, pretransform::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCMultiCycleBurnin(nsteps_per_cycle = max(div(nsteps, 10), 2500))


function _create_proposal_state(
    proposal::MCMCGlobalProposal,
    target::BATMeasure,
    context::BATContext,
    v_init::AbstractVector{PV},
    f_transform::Function,
    rng::AbstractRNG
) where {P<:Real, PV<:AbstractVector{P}}

    if isnothing(proposal.global_proposal)
        global_prop = get_iid_sampleable_approx(target)
        #throw(ArgumentError("No adequate global proposal measure detected. Please supply one to MCMCGlobalProposal() or make sure the sampling target supplies a suitable measure."))
    else
        global_prop = batmeasure(proposal.global_proposal)
    end

    return MCMCGlobalProposalProposalState(
        proposal.target_acceptance,
        proposal.target_acceptance_int,
        global_prop
    )
end

function mcmc_propose_transition(
    current_z::ArrayOfSimilarArrays,
    proposal::MCMCGlobalProposalProposalState,
    n_walkers::Integer,
    genctx
)
    proposal_measure = batmeasure(proposal.global_proposal)
    proposed_z = rand(genctx, proposal_measure^n_walkers)

    hastings_correction = checked_logdensityof.(proposal_measure, current_z) .- checked_logdensityof.(proposal_measure, proposed_z)
    return proposed_z, hastings_correction
end

get_proposal_tuning_quality(proposal::MCMCGlobalProposalProposalState, ::MCMCChainState, ::Float64) = 1.0

set_proposal_transform!!(proposal::MCMCGlobalProposalProposalState, ::MCMCChainState) = proposal
