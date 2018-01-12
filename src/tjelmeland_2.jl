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

struct MultipleChainState{
    IT<:Integer
}
    κ::IT
    current::Vector # TODO not typesafe
    proposed::Matrix # TODO not typesafe
end

doc"""
    transition_matrix2(targetvalues::Vector, κ::Int)

Compute the transition matrix (transition alternative 2 in Tjelmeland (2005))
and return the `κ`-th row. `proposedvalues` has the `m+1` proposed points, the
current point is the κ-th, and 'pl' computes the p_l values needed to contruct
the transition function T1.
"""
function transition_matrix2(prob_proposed::Vector, κ::Int)
    row_init = prob_proposed ./ sum(prob_proposed)
    A = row_init
    for i = 1:length(row_init)
         A = vcat(A, row_init)
    end
    a = Int32[]
    for i = 1:length(row_init)
        if A[i,i] != 0
            push!(a,i)
        end
    end
    if len(a) <= 1
        return A[κ, :]
    else
        while len(a) > 1
            b = Float[]
            for i = 1:len(a)
                u = 0.0
                for j = 1:len(a)
                    u += A[i,j]
                end
                push!(b,u)
            end
            u = minimum(b)
            ind = indmin(b)
            for i = 1:len(a)
                for j = 1:len(a)
                    if a[i] != a[j]
                        A[a[i],a[j]] *= u
                    end
                end
                A[a[i],a[i]] = 1 - sum(A[a[i],:]) + A[a[i],a[i]]
                splice!(a,ind)
            end
        end
        return A[κ, :]
    end
end

function pl(proposedvalues::Vector)

end
# prob_proposed = pl(proposedvalues)



# generate multiple proposals, store in `proposed_params`
# proposal_rand!(rng, pdist, proposed_params, current_params)

# follow the example of
# function mcmc_propose_accept_reject!(
