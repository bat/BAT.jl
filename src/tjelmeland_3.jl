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
function diag_inspection(row::Vector)
    row = row ./ sum(row)
    diag = Float64[]
    index = Int32[]
    for i = 1:length(row)
        if row[i] != 0
            push!(index, i)
            push!(diag, row[i])
        end
    end
    return index, diag
end

function outside_submatrix(row::Vector, index::Vector)
    outside = Float64[]
    for i in index
        out::Float64 = 0.0
        for j in index
            out += row[j]
        end
        out = 1 - out
        push!(outside, out)
    end
    return outside
end

function update!(row::Vector, diag::Vector, outside::Vector, index::Vector, u::Float64, κ::Int)
    diag = 1 - outside - u * (1 - outside - diag)
    for i in index
        if i == κ
            row[i] = diag[i]
        else
            row[i] *= u
        end
    end
    for i = 1:length(diag)
        if diag[i] == 0
            splice!(diag,i)
            splice!(outside,i)
            splice!(index,i)
        end
    end
end

function mini_u(index::Vector, diag::Vector, outside::Vector)
    u = typemax(Float64)
    for i = 1:length(diag)
        u_test::Float64 = (1 - outside[i]) / (1 - outside[i] - diag[i])
        if u_test < u
            u = u_test
        end
    end
    return u
end

function T2(row::Vector, κ::Int)

    if row[κ] == 0
        return row
    end

    index, diag = diag_inspection(row)
    outside_submatrix(row, index)

    while in(κ, index) == true
        u = mini_u(index, diag, outside)
        update!(row, diag, outside, index, u, κ)
    end
    return row
end
