# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# Note: Matrices stored with swapped indices to Tjelmeland's paper,
# due to Julia's array memory layout.


function _tjl_multipropT1!(P_T1::AbstractVector{<:AbstractFloat}, pdist::GenericProposalDist, params::AbstractMatrix{<:Real}, logdensity::Vector{<:Real}) # TODO include checks for input, optimize and write test
    indices(params, 2) != indices(logdensity, 1) && throw(ArgumentError("Number of parameter sets doesn't match number of log(density) values"))
    indices(params, 2) != indices(P_T1, 1) && throw(ArgumentError("Number of parameter sets doesn't match size of P_T1"))

    # ToDo: Optimize for symmetric proposals?
    p_d = similar(logdensity, size(params, 2), size(params, 2))
    @inbounds for j in indices(params, 2)
        distribution_logpdf!(view(p_d, :, j), pdist, params, view(params, :, j))  # Memory allocation due to view
    end

    @inbounds for j in eachindex(P_T1)
        P_T1[j] = sum_first_dim(p_d, j) - p_d[j, j] + logdensity[j]
    end
    P_T1 .-= maximum(P_T1)
    P_T1 .= exp.(P_T1)
    # @assert !any(isnan, P_T1)

    # @assert !all(iszero, P_T1)
    normalize!(P_T1, 1)
    # @assert !any(isnan, P_T1)
    @assert sum(P_T1) ≈ 1
    #println("T1 transitions are : $P_T1")
    P_T1
end


doc"""
    multipropT2(P_T2::AbstractVector, P_T1::AbstractVector)

Compute the transition probabilities T2 from Tjelmeland (2002).

`P_transition` contains the T1 transition probabilities to the proposed
states, the first entry refers to the current state ( probability to stay in
place).
"""
function _tjl_multipropT2!(P_T2::AbstractVector{<:AbstractFloat}, P_T1::AbstractVector{<:AbstractFloat})
    idxs = eachindex(P_T2)
    idxs != eachindex(P_T1) && throw(ArgumentError("P_T2 and P_T1 must have the same indices"))
    any(x -> x < 0, P_T1) && throw(ArgumentError("All values in P_T1 must be positive"))
    !(sum(P_T1) ≈ 1) && throw(ArgumentError("Sum of P_T1 must be one"))

    # Construct initial P matrix
    P = repeat(P_T1, outer=(1, length(idxs)))  # Memory allocation!

    selidxs = collect(idxs)  # Memory allocation!

    done = false
    while !done
        _tjl_update_selidxs!(selidxs, P)
        P_sub = view(P, selidxs, selidxs)  # Memory allocation!
        if (length(selidxs) < 2) || P[1, 1] ≈ 0
            @inbounds for l in idxs
                P_T2[l] = P[l, 1]
            end
            done = true
        else
            u = _tjl_min_u(P_sub)
            #println("The u_min is : $u")
            _tjl_update_sub_P!(P_sub, u)
        end
    end

    assert(sum(P_T2) ≈ 1)
    normalize!(P_T2, 1)
    #@log_msg 1 "The weights are : $P_T2"
    #println("T2 transitions are : $P_T2")
end


# Calculate min(u) for Tjelmeland T2
function _tjl_min_u(P::AbstractMatrix{<:AbstractFloat})
    idxs = indices(P, 1)
    @assert idxs == indices(P, 2)
    length(idxs) < 2 && throw(ArgumentError("Size of matrix P must be at least 2x2"))
    u = typemax(eltype(P))
    @inbounds for k in idxs
        s = sum_first_dim(P, k)
        u = min(u, s / (s - P[k, k]))
    end
    u
end


function _tjl_update_sub_P!(P::AbstractMatrix{<:AbstractFloat}, u::AbstractFloat)
    idxs = indices(P, 1)
    @assert idxs == indices(P, 2)
    @inbounds for k in idxs
        s = sum_first_dim(P, k)
        P[k, k] = 1 - u * (s - P[k, k]) - (1 - s)
    end
    P .*= u
    inv_u = inv(u)
    for k in idxs
        P[k, k] *= inv_u
    end
    P
end


function _tjl_update_row!(row::Vector, submat_row::Vector, index::Vector, indexnew::Vector)
    for (i,a) in zip(index, indexnew)
        row[i] = submat_row[a]
    end
end


function _tjl_update_selidxs!(selidxs::AbstractVector{<:Integer}, P::AbstractMatrix{<:AbstractFloat})
    j = first(eachindex(selidxs))
    while j in eachindex(selidxs)
        if P[j, j] ≈ 0
            splice!(selidxs, j)
        else
            j += 1
        end
    end
end
