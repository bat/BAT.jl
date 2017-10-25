# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct DensityProduct{
    N,
    D<:NTuple{N,AbstractDensityFunction},
    B<:AbstractParamBounds
} <: AbstractDensityFunction
    densities::D
    bounds::B
end

Base.parent(density::DensityProduct) = density.densities

param_bounds(density::DensityProduct) = density.bounds

nparams(density::DensityProduct) = nparams(density.bounds)


import Base.*
function Base.*(a::AbstractDensityFunction, b::AbstractDensityFunction)
    nparams(a) != nparams(b) && throw(ArgumentError("Can't multiply densities with different number of arguments"))
    new_bounds = param_bounds(a) ∩ param_bounds(b)
    _unsafe_prod(a, b, new_bounds)
end


_unsafe_prod(a::AbstractDensityFunction, b::AbstractDensityFunction, new_bounds::AbstractParamBounds) =
    DensityProduct((a,b), new_bounds)

_unsafe_prod(a::AbstractDensityFunction, b::DensityProduct, new_bounds::AbstractParamBounds) =
    DensityProduct((a, b.densities...), new_bounds)

_unsafe_prod(a::DensityProduct, b::DensityProduct, new_bounds::AbstractParamBounds) =
    DensityProduct((a.densities..., b.densities...), new_bounds)

_unsafe_prod(a::DensityProduct, b::AbstractDensityFunction, new_bounds::AbstractParamBounds) =
    DensityProduct((a.densities...,b), new_bounds)


function density_logval(density::DensityProduct, args...)
    ds = densities.densities
    isempty(ds) && throw(ArgumentError("Can't evaluate density_logval on empty DensityProduct"))
    sum(map(d -> density_logval(d, args...), ds))
end

exec_capabilities(::typeof(density_logval), density::DensityProduct, args...)
    sum(map(d -> exec_capabilities(density_logval, d, args...), density.densities))


function @inline density_logval!(r::AbstractArray{<:Real}, density::DensityProduct, args...)
    ds = densities.densities
    isempty(ds) && throw(ArgumentError("Can't evaluate density_logval! on empty DensityProduct"))
    fill!(r, 0)
    tmp = similar(d)  # ToDo: Avoid memory allocation
    for d in ds
        density_logval!(tmp, d, args...)
        r .+= tmp
    end
    r
end

exec_capabilities(::typeof(density_logval!), r::AbstractArray{<:Real}, density::DensityProduct, args...)
    sum(map(d -> exec_capabilities(density_logval!, r, d, args...), density.densities))
