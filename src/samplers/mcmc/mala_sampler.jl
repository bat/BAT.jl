# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct MALAProposal <: MCMCProposal

Metropolis adjusted Langevin sampling algorithm.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MALAProposal{
    TA<:Real,
    TAI<:Tuple{Vararg{Real}},
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

export MALAProposal

# Per-walker z-space gradients of the last transition, at the current and
# the proposed states, mutated in place (all functional proposal-state
# copies share this object by reference). They provide the selected-state
# gradients for MCMCStepInfo, at no extra gradient evaluations:
mutable struct _MALAGradCache
    grads_curr::Vector{Vector{Float64}}
    grads_prop::Vector{Vector{Float64}}
end

_MALAGradCache() = _MALAGradCache(Vector{Vector{Float64}}(), Vector{Vector{Float64}}())

struct MALAProposalState{
    TA<:Real,
    TAI<:Tuple{Vararg{Real}},
    Q<:BATMeasure,
    G<:Function,
    R<:Real
} <: SimpleMCMCProposalState
    target_acceptance::TA
    target_acceptance_int::TAI
    proposaldist::Q
    target_gradient::G
    τ::R
    grad_cache::_MALAGradCache
end

mcmc_step_provides_grads(::MALAProposalState) = true

function _selected_z_grads(proposal::MALAProposalState, accepted::AbstractVector{Bool})
    c = proposal.grad_cache
    length(c.grads_curr) == length(accepted) || return nothing
    return [accepted[i] ? c.grads_prop[i] : c.grads_curr[i] for i in eachindex(accepted)]
end


bat_default(::Type{TransformedMCMC}, ::Val{:pretransform}, proposal::MALAProposal) = NormalBased()

bat_default(::Type{TransformedMCMC}, ::Val{:proposal_tuning}, proposal::MALAProposal) = NoMCMCProposalTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:adaptive_transform}, proposal::MALAProposal) = TriangularAffineTransform()

bat_default(::Type{TransformedMCMC}, ::Val{:tempering}, proposal::MALAProposal) = NoMCMCTempering()


function _create_proposal_state(
    proposal::MALAProposal,
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
        n_dims^(-1/3) * proposal.τ_base,
        _MALAGradCache()
    )
end

function mcmc_propose_transition(
    current_z::ArrayOfSimilarArrays,
    proposal::MALAProposalState,
    n_walkers::Integer,
    genctx
)
    # https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm

    proposal_measure = batmeasure(proposal.proposaldist)
    (; target_gradient, τ) = proposal

    gradient_res_curr = target_gradient.(current_z)
    grads_curr = last.(gradient_res_curr)

    transition = τ/2 .* grads_curr .+ sqrt(τ) .* rand(genctx, proposal_measure^n_walkers)

    proposed_z = current_z .+ transition

    gradient_res_prop = target_gradient.(proposed_z)
    grads_prop = last.(gradient_res_prop)

    proposal.grad_cache.grads_curr = convert(Vector{Vector{Float64}}, grads_curr)
    proposal.grad_cache.grads_prop = convert(Vector{Vector{Float64}}, grads_prop)

    p_prop_to_curr = norm.(-transition .- τ .* grads_prop).^2
    p_curr_to_prop = norm.(transition .- τ .* grads_curr).^2

    hastings_correction = (p_curr_to_prop - p_prop_to_curr) ./ (4τ)

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
