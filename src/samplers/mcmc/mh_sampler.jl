# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct RandomWalk <: MCMCProposal

Metropolis-Hastings MCMC sampling algorithm.

For Gaussian random-walk Metropolis in the high-dimensional symmetric
product-target regime, the asymptotic target acceptance rate and proposal
scaling are
[G. O. Roberts, A. Gelman and W. R. Gilks, "Weak convergence and
optimal scaling of random walk Metropolis algorithms"
(1997)](https://doi.org/10.1214/aoap/1034625254).

BAT applies the same scale heuristically to its default Cauchy (`TDist(1)`)
innovation.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct RandomWalk{
    TA<:Real,
    TAI<:Tuple{Vararg{Real}},
    Q<:Union{
        AbstractMeasure,
        Distribution{<:Union{Univariate,Multivariate},Continuous}
    }
} <: MCMCProposal
    # 0.234 is the asymptotic Gaussian-RWM acceptance rate in the regime of
    # Roberts, Gelman & Gilks (1997), see the docstring above:
    target_acceptance::TA = 0.234
    target_acceptance_int::TAI = (0.15, 0.35)
    proposaldist::Q = TDist(1.0)
end

export RandomWalk

# Isotropic normal innovations of a random walk with the given scale in all
# dimensions:
struct _NormalInnovations{T<:Real}
    scale::T
end

# Innovations of a random walk that follow an isotropic multivariate
# t-distribution with `ν` degrees of freedom:
struct _TInnovations{T<:Real}
    ν::T
    scale::T
end

# These innovations are drawn as arrays on the compute unit of the chain.
# They are symmetric, so a random walk with them has no Hastings terms:
const _SymmetricInnovations = Union{_NormalInnovations,_TInnovations}

struct RandomWalkProposalState{
    TA<:Real,
    TAI<:Tuple{Vararg{Real}},
    Q<:Union{_SymmetricInnovations,AbstractMeasure}
} <: SimpleMCMCProposalState
    target_acceptance::TA
    target_acceptance_int::TAI
    proposaldist::Q
end

bat_default(::Type{TransformedMCMC}, ::Val{:pretransform}, proposal::RandomWalk) = NormalBased()

bat_default(::Type{TransformedMCMC}, ::Val{:proposal_tuning}, proposal::RandomWalk) = NoMCMCProposalTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:adaptive_transform}, proposal::RandomWalk) = TriangularAffineTransform()

bat_default(::Type{TransformedMCMC}, ::Val{:tempering}, proposal::RandomWalk) = NoMCMCTempering()


function _create_proposal_state(
    proposal::RandomWalk,
    target::AbstractMeasure,
    context::BATContext,
    v_init::AbstractVector{PV},
    f_transform::Function,
    rng::AbstractRNG
) where {P<:Real, PV<:AbstractVector{P}}
    n_dims = totalndof(varshape(target))
    return RandomWalkProposalState(
        proposal.target_acceptance,
        proposal.target_acceptance_int,
        _full_random_walk_proposal(proposal.proposaldist, n_dims)
    )
end


# The innovations of a random walk in all dimensions. A measure over the
# innovation vectors is used as it is, it may be asymmetric:
function _full_random_walk_proposal(m::AbstractMeasure, n_dims::Integer)
    x = testvalue(m)
    @argcheck x isa AbstractVector{<:Real} && length(x) == n_dims
    return m
end

function _full_random_walk_proposal(m::AsMeasure{<:Distribution}, n_dims::Integer)
    return _full_random_walk_proposal(m.obj, n_dims)
end

# A user-supplied full multivariate innovation distribution would need a
# symmetry guarantee (the random-walk acceptance assumes a symmetric
# proposal), which can't be checked generically:
function _full_random_walk_proposal(d::Distribution{Multivariate,Continuous}, n_dims::Integer)
    throw(ArgumentError(
        "RandomWalk doesn't support full multivariate proposal distributions yet, use a univariate distribution (like Normal or TDist) that sets the shape of the per-dimension innovations"
    ))
end

function _full_random_walk_proposal(d::Normal, n_dims::Integer)
    # Asymptotic Gaussian-RWM proposal scale in the regime of
    # [Roberts, Gelman & Gilks, Ann. Appl. Probab. 7 (1) 110 - 120, 1997](https://doi.org/10.1214/aoap/1034625254):
    proposal_scale = 2.38 / sqrt(n_dims)

    @argcheck mean(d) ≈ 0 
    return _NormalInnovations(proposal_scale * std(d))
end

function _full_random_walk_proposal(d::TDist, n_dims::Integer)
    # BAT applies the Gaussian-RWM scale heuristically to t innovations:
    proposal_scale = 2.38 / sqrt(n_dims)

    return _TInnovations(promote(float(dof(d)), proposal_scale)...)
end


# The innovations of all walkers, as the columns of a matrix:
function _draw_innovations(walker_draws, innovations::_NormalInnovations, n_dims::Integer)
    Z = _walker_randn(walker_draws, n_dims)
    return convert(eltype(Z), innovations.scale) .* Z
end

# A vector with a multivariate t-distribution is a normal vector, divided by
# the root of a chi-square variate over its degrees of freedom:
function _draw_innovations(walker_draws, innovations::_TInnovations, n_dims::Integer)
    Z = _walker_randn(walker_draws, n_dims)
    T = eltype(Z)
    ν = innovations.ν
    χ² = if isinteger(ν)
        sum(abs2, _walker_randn(walker_draws, Int(ν)); dims = 1)
    else
        # Needs gamma variates, which have no sampler on devices:
        reshape(_walker_variates(walker_draws, batmeasure(Chisq(ν))), 1, :)
    end
    return Z .* (convert(T, innovations.scale) ./ sqrt.(χ² ./ convert(T, ν)))
end


_draws_batched(::RandomWalkProposalState) = true

function mcmc_propose_transition(
    current_z::ArrayOfSimilarArrays,
    proposal::RandomWalkProposalState,
    walker_draws,
)
    Z_current = flatview(current_z)
    transition, hastings_correction = _random_walk_transition(walker_draws, proposal.proposaldist, size(Z_current, 1))
    return nestedview(Z_current .+ transition), hastings_correction
end

# The transitions of all walkers, as the columns of a matrix, and the
# Hastings terms of a random walk with them:
function _random_walk_transition(walker_draws, innovations::_SymmetricInnovations, n_dims::Integer)
    transition = _draw_innovations(walker_draws, innovations, n_dims)
    return transition, Fill(false, size(transition, 2))
end

function _random_walk_transition(walker_draws, innovations::AbstractMeasure, n_dims::Integer)
    # Flat batches, so that the innovation measure can evaluate the Hastings
    # terms for all walkers at once:
    transition = _walker_variates(walker_draws, innovations)
    p_prop_to_curr = checked_logdensities(innovations, nestedview(-flatview(transition)))
    p_curr_to_prop = checked_logdensities(innovations, transition)
    return flatview(transition), p_prop_to_curr .- p_curr_to_prop
end

set_proposal_transform!!(proposal::RandomWalkProposalState, ::MCMCChainState) = proposal
