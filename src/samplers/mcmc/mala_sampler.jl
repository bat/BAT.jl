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
    proposaldist::Q = Normal()
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
    mv_pdist = batmeasure(_mala_innovation_dist(proposal.proposaldist, n_dims))

    adsel = get_valid_adselector(context, proposal)
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

# The Langevin innovation acts at unit scale per dimension, the step
# scale lives in τ (unlike random-walk proposals, whose distribution
# carries the dimension-dependent optimal scale itself):
function _mala_innovation_dist(d::UnivariateDistribution, n_dims::Integer)
    return product_distribution(fill(d, n_dims))
end

function _mala_innovation_dist(m, n_dims::Integer)
    x = testvalue(batmeasure(m))
    @argcheck x isa AbstractVector{<:Real} && length(x) == n_dims
    return m
end

# Exact log proposal ratio log q(x|y) - log q(y|x) of the Langevin
# proposal y = x + (τ/2) ∇log π(x) + √τ ξ, for any innovation
# distribution: the forward and reverse innovations are recovered from
# the transition and the respective drifts, and the common √τ scale
# Jacobians cancel:
function _mala_log_proposal_ratio(proposal_measure::BATMeasure, τ::Real, transition, grads_curr, grads_prop)
    logd = logdensityof(proposal_measure)
    ξ_fwd = (transition .- τ/2 .* grads_curr) ./ sqrt(τ)
    ξ_rev = (.-transition .- τ/2 .* grads_prop) ./ sqrt(τ)
    return logd.(ξ_rev) .- logd.(ξ_fwd)
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

    hastings_correction = _mala_log_proposal_ratio(proposal_measure, τ, transition, grads_curr, grads_prop)

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
