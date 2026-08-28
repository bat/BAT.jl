# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct MALAProposal <: MCMCProposal

Metropolis adjusted Langevin sampling algorithm.

See [G. O. Roberts and R. L. Tweedie, "Exponential convergence of
Langevin distributions and their discrete approximations"
(1996)](https://doi.org/10.2307/3318418). The default target
acceptance rate and the dimension-dependent step scaling follow
[G. O. Roberts and J. S. Rosenthal, "Optimal scaling of discrete
approximations to Langevin diffusions"
(1998)](https://doi.org/10.1111/1467-9868.00123); that optimality
theory assumes Gaussian innovations, so with a non-Gaussian
`proposaldist` consider setting `target_acceptance` explicitly.

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
    # 0.574 and the n^(-1/3) step scaling are the asymptotically optimal
    # values of Roberts & Rosenthal (1998), see the docstring above:
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
mutable struct _MALAGradCache{T<:AbstractFloat}
    grads_curr::Vector{Vector{T}}
    grads_prop::Vector{Vector{T}}
end

_MALAGradCache() = _MALAGradCache(Vector{Float64}[], Vector{Float64}[])
_MALAGradCache(::AbstractVector{<:AbstractVector{P}}) where {P<:Real} =
    _MALAGradCache(Vector{float(P)}[], Vector{float(P)}[])

struct MALAProposalState{
    TA<:Real,
    TAI<:Tuple{Vararg{Real}},
    Q<:BATMeasure,
    G<:Function,
    R<:Real,
    C<:_MALAGradCache,
} <: SimpleMCMCProposalState
    target_acceptance::TA
    target_acceptance_int::TAI
    proposaldist::Q
    target_gradient::G
    τ::R
    grad_cache::C
end

mcmc_step_provides_grads(::MALAProposalState) = true

function _selected_z_grads(proposal::MALAProposalState, accepted::AbstractVector{Bool})
    c = proposal.grad_cache
    length(c.grads_curr) == length(accepted) || return nothing
    return [accepted[i] ? c.grads_prop[i] : c.grads_curr[i] for i in eachindex(accepted)]
end


bat_default(::Type{TransformedMCMC}, ::Val{:pretransform}, proposal::MALAProposal) = NormalBased()

bat_default(::Type{TransformedMCMC}, ::Val{:proposal_tuning}, proposal::MALAProposal) = StepSizeAdaptor()

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

    # The analytic affine pullback keeps AD in the fixed x-space, so the
    # AD preparation survives geometry changes and operator-valued affine
    # transforms never enter the AD graph:
    target_gradient = _target_logdgrad_func(target, f_transform, context, proposal, convert(Vector{float(P)}, first(v_init)))

    return MALAProposalState(
        proposal.target_acceptance,
        proposal.target_acceptance_int,
        mv_pdist,
        target_gradient,
        n_dims^(-1/3) * proposal.τ_base,
        _MALAGradCache(v_init),
    )
end

# The Langevin innovation acts at unit scale per dimension, the step
# scale lives in τ (unlike random-walk proposals, whose distribution
# carries the dimension-dependent optimal scale itself). The
# n_dims^(-1/3) scaling of τ is the Roberts & Rosenthal (1998)
# optimal-scaling rate for Langevin diffusion approximations:
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
    # MALA proposal (Roberts & Tweedie 1996), see the MALAProposal docstring.

    proposal_measure = batmeasure(proposal.proposaldist)
    (; target_gradient, τ) = proposal

    gradient_res_curr = target_gradient.(current_z)
    grads_curr = last.(gradient_res_curr)

    transition = τ/2 .* grads_curr .+ sqrt(τ) .* rand(genctx, proposal_measure^n_walkers)

    proposed_z = current_z .+ transition

    gradient_res_prop = target_gradient.(proposed_z)
    grads_prop = last.(gradient_res_prop)

    proposal.grad_cache.grads_curr = grads_curr
    proposal.grad_cache.grads_prop = grads_prop

    hastings_correction = _mala_log_proposal_ratio(proposal_measure, τ, transition, grads_curr, grads_prop)

    return proposed_z, hastings_correction
end

function set_proposal_transform!!(proposal::MALAProposalState, chain_state::MCMCChainState)
    fg_new = _updated_logdgrad_func(
        proposal.target_gradient, chain_state.target, chain_state.f_transform,
        chain_state.context, proposal, Vector(first(chain_state.current.x.v))
    )
    return @set proposal.target_gradient = fg_new
end
