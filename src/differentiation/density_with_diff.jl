# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct DensityWithDiff{D<:AbstractDensity,VJP<:DifferentiationAlgorithm} <: AbstractDensity
    vjpalg::VJP
    density::D
end 

@inline Base.parent(density::DensityWithDiff) = density.density

vjp_algorithm(density::DensityWithDiff) = density.vjpalg

Base.:(|)(vjpalg::DifferentiationAlgorithm, density::AbstractDensity) = DensityWithDiff(vjpalg, density)
Base.:(|)(vjpalg::DifferentiationAlgorithm, density::DensityWithDiff) = DensityWithDiff(vjpalg, parent(density))
Base.:(|)(vjpalg::DifferentiationAlgorithm, density::PosteriorDensity) = PosteriorDensity(vjpalg | density.likelihood, vjpalg | density.prior)

function Base.show(io::IO, density::DensityWithDiff)
    vjpalg = vjp_algorithm(density)
    Base.show(io, vjpalg)
    print(io, "|")
    Base.show(io, parent(density))
end

var_bounds(density::DensityWithDiff) = var_bounds(density.density)

ValueShapes.varshape(density::DensityWithDiff) = varshape(density.density)

ValueShapes.unshaped(density::DensityWithDiff) = DensityWithDiff(density.vjpalg, unshaped(density.density))


@inline DensityInterface.logdensityof(density::DensityWithDiff, v::Any) = logdensityof(density.density, v)

@inline checked_logdensityof(density::DensityWithDiff, v::Any) = checked_logdensityof(density.density, v)


@inline get_initsrc_from_target(target::DensityWithDiff) = get_initsrc_from_target(target.density)


function bat_transform_impl(target::NoDensityTransform, density::DensityWithDiff, algorithm::DensityIdentityTransform)
    (result = density, trafo = identity)
end

function bat_transform_impl(target::AbstractDensityTransformTarget, density::DensityWithDiff, algorithm::TransformAlgorithm)
    transformed_density, trafo = bat_transform_impl(target, parent(density), algorithm)
    (result = DensityWithDiff(vjp_algorithm(density), transformed_density), trafo = trafo)
end
