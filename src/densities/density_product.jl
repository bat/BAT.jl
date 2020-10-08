# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct DensityProduct{
    N,
    D<:NTuple{N,AbstractDensity},
    B<:AbstractVarBounds
} <: AbstractDensity
    densities::D
    bounds::B

    # ToDo: Check for equal number of parameters in DensityProduct ctor.

    DensityProduct(densities::D, bounds::B) where {N,D<:NTuple{N,AbstractDensity},B<:AbstractVarBounds} =
        new{N,D,B}(densities, bounds)

end

Base.parent(density::DensityProduct) = density.densities

var_bounds(density::DensityProduct) = density.bounds

ValueShapes.varshape(density::DensityProduct) = ArrayShape{Real}(totalndof(density.bounds))


import Base.*
function *(a::AbstractDensity, b::AbstractDensity)
    totalndof(a) != totalndof(b) && throw(ArgumentError("Can't multiply densities with different number of arguments"))
    new_bounds = var_bounds(a) ∩ var_bounds(b)
    _unsafe_prod(a, b, new_bounds)
end


_unsafe_prod(a::AbstractDensity, b::AbstractDensity, new_bounds::AbstractVarBounds) =
    DensityProduct((a,b), new_bounds)

_unsafe_prod(a::AbstractDensity, b::DensityProduct, new_bounds::AbstractVarBounds) =
    DensityProduct((a, b.densities...), new_bounds)

_unsafe_prod(a::DensityProduct, b::DensityProduct, new_bounds::AbstractVarBounds) =
    DensityProduct((a.densities..., b.densities...), new_bounds)

_unsafe_prod(a::DensityProduct, b::AbstractDensity, new_bounds::AbstractVarBounds) =
    DensityProduct((a.densities...,b), new_bounds)


function eval_logval_unchecked(density::DensityProduct, v::AbstractVector{<:Real})
    ds = density.densities
    isempty(ds) && throw(ArgumentError("Can't evaluate empty DensityProduct"))
    sum(map(d -> eval_logval_unchecked(d, v), ds))
end
