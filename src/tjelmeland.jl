# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# similar to mh_sampler.jl

struct MultipleMetropolisHastings{
    Q<:ProposalDistSpec,
    W<:Real,
    WS<:MHWeightingScheme{W},
    IT<:Integer
} <: MCMCAlgorithm{AcceptRejectState}
    q::Q
    weighting_scheme::WS # TODO do we need a separate hierarchy? Probably yes.
    m::IT # m=1 is the ordinary mh_sampler
end

struct MultipleChainState{}
    IT<:Integer
}
    κ::IT,
    current::Vector, # TODO not typesafe
    proposed::Matrix, # TODO not typesafe
end

@doc"""
    transition_matrix2(targetvalues::Vector, κ::Int)

Compute the transition matrix (transition alternative 2 in Tjelmeland (2005))
and return the `κ`-th row. `targetvalues` has the `m+1` values of the target
density at the current and the `m` proposed points.
"""
function transition_matrix2(targetvalues::Vector, κ::Int)

end

# generate multiple proposals, store in `proposed_params`
# proposal_rand!(rng, pdist, proposed_params, current_params)

# follow the example of
# function mcmc_propose_accept_reject!(
