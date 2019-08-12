# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct DensityProduct{
    N,
    D<:NTuple{N,AbstractDensity},
    B<:AbstractParamBounds
} <: AbstractDensity
    densities::D
    bounds::B

    # ToDo: Check for equal number of parameters in DensityProduct ctor.

    DensityProduct(densities::D, bounds::B) where {N,D<:NTuple{N,AbstractDensity},B<:AbstractParamBounds} =
        new{N,D,B}(densities, bounds)

end

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


function density_logval(density::DensityProduct, params::AbstractVector{<:Real})
    ds = density.densities
    isempty(ds) && throw(ArgumentError("Can't evaluate density_logval on empty DensityProduct"))
    sum(map(d -> density_logval(d, params), ds))
end
