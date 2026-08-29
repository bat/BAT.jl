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

struct RandomWalkProposalState{
    TA<:Real,
    TAI<:Tuple{Vararg{Real}},
    Q<:BATMeasure
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
    target::BATMeasure,
    context::BATContext,
    v_init::AbstractVector{PV},
    f_transform::Function,
    rng::AbstractRNG
) where {P<:Real, PV<:AbstractVector{P}}
    n_dims = totalndof(varshape(target))
    mv_pdist = batmeasure(_full_random_walk_proposal(proposal.proposaldist, n_dims))
    return RandomWalkProposalState(
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
    σ² = var(d)
    Σ = ScalMat(n_dims, proposal_scale^2 * σ²)
    return MvNormal(Σ)
end

function _full_random_walk_proposal(d::TDist, n_dims::Integer)
    # BAT applies the Gaussian-RWM scale heuristically to t innovations:
    proposal_scale = 2.38 / sqrt(n_dims)

    ν = dof(d)
    Σ = ScalMat(n_dims, proposal_scale^2)
    return Distributions.IsoTDist(ν, Σ)
end


function mcmc_propose_transition(
    current_z::ArrayOfSimilarArrays,
    proposal::MCMCProposalState,
    genctxs::AbstractVector,
)
    proposal_measure = batmeasure(proposal.proposaldist)

    transition = map(genctx -> rand(genctx, proposal_measure), genctxs)
    proposed_z = current_z .+ transition

    p_prop_to_curr = checked_logdensityof.(proposal_measure, -transition)
    p_curr_to_prop = checked_logdensityof.(proposal_measure, transition) 

    hastings_correction = p_prop_to_curr .- p_curr_to_prop

    return proposed_z, hastings_correction
end

set_proposal_transform!!(proposal::RandomWalkProposalState, ::MCMCChainState) = proposal
