# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct DensityProduct{
    N,
    D<:NTuple{N,AbstractDensity},
    B<:AbstractParamBounds
} <: AbstractDensity
    densities::D
    bounds::B
end

# ToDo: Check for equal number of parameters in DensityProduct ctor.

DensityProduct(densities::D, bounds::B) where {N,D<:NTuple{N,AbstractDensity},B<:AbstractParamBounds} =
    BAT.DensityProduct{N,D,B}(densities, bounds)

Base.parent(density::DensityProduct) = density.densities

param_bounds(density::DensityProduct) = density.bounds

nparams(density::DensityProduct) = nparams(density.bounds)


import Base.*
function *(a::AbstractDensity, b::AbstractDensity)
    nparams(a) != nparams(b) && throw(ArgumentError("Can't multiply densities with different number of arguments"))
    new_bounds = param_bounds(a) ∩ param_bounds(b)
    _unsafe_prod(a, b, new_bounds)
end


_unsafe_prod(a::AbstractDensity, b::AbstractDensity, new_bounds::AbstractParamBounds) =
    DensityProduct((a,b), new_bounds)

_unsafe_prod(a::AbstractDensity, b::DensityProduct, new_bounds::AbstractParamBounds) =
    DensityProduct((a, b.densities...), new_bounds)

_unsafe_prod(a::DensityProduct, b::DensityProduct, new_bounds::AbstractParamBounds) =
    DensityProduct((a.densities..., b.densities...), new_bounds)

_unsafe_prod(a::DensityProduct, b::AbstractDensity, new_bounds::AbstractParamBounds) =
    DensityProduct((a.densities...,b), new_bounds)


function unsafe_density_logval(density::DensityProduct, args...)
    ds = density.densities
    isempty(ds) && throw(ArgumentError("Can't evaluate density_logval on empty DensityProduct"))
    sum(map(d -> unsafe_density_logval(d, args...), ds))
end

exec_capabilities(::typeof(unsafe_density_logval), density::DensityProduct, args...) =
    ∩(map(d -> exec_capabilities(density_logval, d, args...), density.densities)...)

function unsafe_density_logval!(r::AbstractArray{<:Real}, density::DensityProduct, args...)
    ds = density.densities
    isempty(ds) && throw(ArgumentError("Can't evaluate density_logval! on empty DensityProduct"))
    fill!(r, 0)
    tmp = similar(r)  # ToDo: Avoid memory allocation
    for d in ds
        unsafe_density_logval!(tmp, d, args...)
        r .+= tmp
    end
    r
end

exec_capabilities(::typeof(unsafe_density_logval!), r::AbstractArray{<:Real}, density::DensityProduct, args...) =
    ∩(map(d -> exec_capabilities(unsafe_density_logval!, r, d, args...), density.densities)...)
