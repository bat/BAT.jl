# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct WithDiff{D<:AbstractMeasureOrDensity,VJP<:DifferentiationAlgorithm} <: AbstractMeasureOrDensity
    vjpalg::VJP
    density::D
end

@inline DensityInterface.DensityKind(x::WithDiff) = DensityKind(x.density)

function Base.rand(rng::Random.AbstractRNG, m::WithDiff)
    @argcheck DensityKind(m) isa HasDensity
    rand(rng, m.density)
end


@inline Base.parent(density::WithDiff) = density.density

vjp_algorithm(density::WithDiff) = density.vjpalg

Base.:(|)(vjpalg::DifferentiationAlgorithm, density::AbstractMeasureOrDensity) = WithDiff(vjpalg, density)
Base.:(|)(vjpalg::DifferentiationAlgorithm, density::WithDiff) = WithDiff(vjpalg, parent(density))
Base.:(|)(vjpalg::DifferentiationAlgorithm, density::PosteriorMeasure) = PosteriorMeasure(vjpalg | density.likelihood, vjpalg | density.prior)

function Base.show(io::IO, density::WithDiff)
    vjpalg = vjp_algorithm(density)
    Base.show(io, vjpalg)
    print(io, "|")
    Base.show(io, parent(density))
end

var_bounds(density::WithDiff) = var_bounds(density.density)

ValueShapes.varshape(density::WithDiff) = varshape(density.density)

ValueShapes.unshaped(density::WithDiff) = WithDiff(density.vjpalg, unshaped(density.density))


@inline DensityInterface.logdensityof(density::WithDiff, v::Any) = logdensityof(density.density, v)

@inline checked_logdensityof(density::WithDiff, v::Any) = checked_logdensityof(density.density, v)


@inline get_initsrc_from_target(target::WithDiff) = get_initsrc_from_target(target.density)


function bat_transform_impl(target::DoNotTransform, density::WithDiff, algorithm::IdentityTransformAlgorithm)
    (result = density, trafo = identity)
end

function bat_transform_impl(target::AbstractTransformTarget, density::WithDiff, algorithm::TransformAlgorithm)
    transformed_density, trafo = bat_transform_impl(target, parent(density), algorithm)
    (result = WithDiff(vjp_algorithm(density), transformed_density), trafo = trafo)
end
