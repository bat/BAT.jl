# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function vjp_algorithm end
vjp_algorithm(f::Function) = ZygoteAD()
vjp_algorithm(d::AbstractDensity) = ForwardDiffAD()
vjp_algorithm(d::AbstractPosteriorDensity) = vjp_algorithm(getlikelihood(d))
vjp_algorithm(d::DensityWithShape) = vjp_algorithm(parent(d))
vjp_algorithm(d::TransformedDensity) = vjp_algorithm(parent(d))
vjp_algorithm(d::TruncatedDensity) = vjp_algorithm(parent(d))

vjp_algorithm(f::Union{LogDensityOf,NegLogDensityOf}) = vjp_algorithm(f.density)


function jvp_algorithm end
jvp_algorithm(f::Function) = ForwardDiffAD()
jvp_algorithm(d::AbstractDensity) = ForwardDiffAD()
jvp_algorithm(d::AbstractPosteriorDensity) = jvp_algorithm(getprior(d))
jvp_algorithm(d::DensityWithShape) = jvp_algorithm(parent(d))
jvp_algorithm(d::TransformedDensity) = jvp_algorithm(parent(d))
jvp_algorithm(d::TruncatedDensity) = jvp_algorithm(parent(d))

jvp_algorithm(f::Union{LogDensityOf,NegLogDensityOf}) = jvp_algorithm(f.density)


struct DensityWithDiff{JVP<:DifferentiationAlgorithm,D<:AbstractDensity,VJP<:DifferentiationAlgorithm} <: AbstractDensity
    vjpalg::VJP
    density::D
    jvpalg::JVP
end 

@inline Base.parent(density::DensityWithDiff) = density.density

vjp_algorithm(density::DensityWithDiff) = density.vjpalg
jvp_algorithm(density::DensityWithDiff) = density.jvpalg

Base.:(|)(vjpalg::DifferentiationAlgorithm, density::AbstractDensity) = DensityWithDiff(vjpalg, density, jvp_algorithm(density))
Base.:(|)(vjpalg::DifferentiationAlgorithm, density::DensityWithDiff) = DensityWithDiff(vjpalg, parent(density), jvp_algorithm(density))

Base.:(|)(density::AbstractDensity, jvpalg::DifferentiationAlgorithm) = DensityWithDiff(vjp_algorithm(density), density, jvpalg)
Base.:(|)(density::DensityWithDiff, jvpalg::DifferentiationAlgorithm) = DensityWithDiff(vjp_algorithm(density), parent(density), jvpalg)

function Base.show(io::IO, density::DensityWithDiff)
    vjpalg = vjp_algorithm(density)
    jvpalg = jvp_algorithm(density)
    parent_vjpalg = vjp_algorithm(parent(density))
    parent_jvpalg = jvp_algorithm(parent(density))
    if vjpalg != parent_vjpalg || jvpalg == parent_jvpalg
        Base.show(io, vjpalg)
        print(io, "|")
    end
    Base.show(io, parent(density))
    if jvpalg != parent_jvpalg
        print(io, "|")
        Base.show(io, jvpalg)
    end
end

var_bounds(density::DensityWithDiff) = var_bounds(density.density)

ValueShapes.varshape(density::DensityWithDiff) = varshape(density.density)

ValueShapes.unshaped(density::DensityWithDiff) = DensityWithDiff(density.vjpalg, unshaped(density.density), density.jvpalg)

@inline function eval_logval_unchecked(density::DensityWithDiff, v::Any)
    eval_logval_unchecked(density.density, v)
end

@inline function eval_logval(density::DensityWithDiff, v::Any, T::Type{<:Real})
    eval_logval(density.density, v, T)
end

@inline get_initsrc_from_target(target::DensityWithDiff) = get_initsrc_from_target(target.density)


function bat_transform_impl(target::NoDensityTransform, density::DensityWithDiff, algorithm::DensityIdentityTransform)
    (result = density, trafo = IdentityVT(varshape(density)))
end

function bat_transform_impl(target::AbstractDensityTransformTarget, density::DensityWithDiff, algorithm::TransformAlgorithm)
    transformed_density, trafo = bat_transform_impl(target, parent(density), algorithm)
    (result = DensityWithDiff(vjp_algorithm(density), transformed_density, jvp_algorithm(density)), trafo = trafo)
end
