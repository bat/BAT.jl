# This file is a part of BAT.jl, licensed under the MIT License (MIT).





function _mtm_weight_1!(mtm_W::AbstractVector{<:AbstractFloat}, pdist::GenericProposalDist, params::AbstractMatrix{<:Real}, logdensity::Vector{<:Real}) # TODO include checks for input, optimize and write test
    indices(params, 2) != indices(logdensity, 1) && throw(ArgumentError("Number of parameter sets doesn't match number of log(density) values"))
    indices(params, 2) != indices(mtm_W, 1) && throw(ArgumentError("Number of parameter sets doesn't match size of mtm_W"))

    # ToDo: Optimize for symmetric proposals?
    p_d = similar(logdensity, size(params, 2))
    distribution_logpdf!(view(p_d, 2:size(params, 2)), pdist, view(params, :, 2:size(params, 2))params, view(params, :, 1))  # Memory allocation due to view

    mtm_W .+= logdensity .+ p_d
    #mtmt_W[1] = 0.0
    mtm_W .-= maximum(mtm_W)
    mtm_W .= exp.(mtm_W)
    mtmt_W[1] = 0.0
    normalize!(mtm_W, 1)
    @assert sum(mtm_W) ≈ 1
    @assert mtmt_W[1] ≈ 0

    mtm_W
end


function _mtm_weight_2!(mtm_W::AbstractVector{<:AbstractFloat}, pdist::GenericProposalDist, params::AbstractMatrix{<:Real}, logdensity::Vector{<:Real}) # TODO include checks for input, optimize and write test
    indices(params, 2) != indices(logdensity, 1) && throw(ArgumentError("Number of parameter sets doesn't match number of log(density) values"))
    indices(params, 2) != indices(mtm_W, 1) && throw(ArgumentError("Number of parameter sets doesn't match size of mtm_W"))


    mtm_W .+= logdensity
    #mtmt_W[1] = 0.0
    mtm_W .-= maximum(mtm_W)
    mtm_W .= exp.(mtm_W)
    mtmt_W[1] = 0.0
    normalize!(mtm_W, 1)
    @assert sum(mtm_W) ≈ 1
    @assert mtmt_W[1] ≈ 0

    mtm_W
end
