# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, DensityInterface, ValueShapes
using InverseFunctions: inverse
using BAT: transform_function, batmeasure

@testset "transform_function" begin
    context = BATContext()
    prior = distprod(a = Normal(2.0, 1.0), b = Exponential(0.7))
    posterior = PosteriorMeasure(logfuncdensity(v -> 0.0), prior)
    x = (a = 1.2, b = 0.4)

    for intent in (NormalBased(), UniformBased())
        f = transform_function(intent, posterior)
        # Same intent and object always yield the same transformation:
        f2 = transform_function(intent, posterior)
        @test f2(x) == f(x)
        # The intent descends to the innermost prior:
        @test f(x) == transform_function(intent, batmeasure(prior))(x)
        nested = PosteriorMeasure(logfuncdensity(v -> 0.0), posterior)
        @test transform_function(intent, nested)(x) == f(x)
        # Invertibility:
        @test all(isapprox.(values(inverse(f)(f(x))), values(x)))
        # bat_transform's intent path uses the implied function:
        @test bat_transform(intent, posterior, context).f_transform(x) == f(x)
    end

    @test transform_function(DoNotTransform(), posterior) === identity

    f_flat = transform_function(ToRealVector(), batmeasure(prior))
    @test f_flat(x) == unshaped(x, varshape(batmeasure(prior)))

    # EvaluatedMeasure targets imply the transformation of their measure:
    em = EvaluatedMeasure(posterior)
    @test transform_function(NormalBased(), em)(x) == transform_function(NormalBased(), posterior)(x)
end
