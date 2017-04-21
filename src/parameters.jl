# This file is a part of BAT.jl, licensed under the MIT License (MIT).


using Compat


typealias ParamValues{T} StridedVector{T}


@compat abstract type AbstractParamBounds end



immutable UnboundedParams <: AbstractParamBounds
    ndims::Int
end

in(params::AbstractVector, bounds::UnboundedParams) = true


immutable HyperCubeBounds{T} <: AbstractParamBounds
    from::Vector{T}
    to::Vector{T}
end

in(params::AbstractVector, bounds::HyperCubeBounds) =
    _multi_array_le(bounds.from, params, bounds.to)



_param_bounds(bounds::AbstractParamBounds, log_f) = (bounds, log_f)

function _param_bounds{T}(bounds::Vector{NTuple{2,T}}, log_f)
    # TODO: promote to float
    from = map(x -> x[1], bounds)
    to = map(x -> x[2], bounds)
    (HyperCubeBounds(from, to), log_f)
end
