# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, ValueShapes

@testset "density_functions" begin
    @test @inferred(distprod(
        a = Normal(2, 1),
        b = Exponential(1.3),
        c = [3, 6, 8],
        d = MvNormal([1.6 0.4; 0.4 2.1])
    )) isa ValueShapes.NamedTupleDist

    @test @inferred(distprod((
        a = Normal(2, 1),
        b = Exponential(1.3),
        c = [3, 6, 8],
        d = MvNormal([1.6 0.4; 0.4 2.1])
    ))) isa ValueShapes.NamedTupleDist

    @inferred(distprod(Weibull.([3, 5, 2], [1.3, 1.0, 0.7]))) isa Distributions.Product{<:Continuous,<:Weibull}

    @test lbqintegral(logfuncdensity(x -> -x.*x), Normal()) isa PosteriorMeasure

    @test rand(distbind(
        distprod(
            a = Normal(2, 1),
            b = Exponential(1.3),
        ), merge
    ) do v
        distprod(
            c = Normal(v.a, v.b),
            d = Weibull()
        )
    end) isa NamedTuple
end
