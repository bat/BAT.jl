# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using DensityInterface, Distributions, MeasureBase, ValueShapes

@testset "measure_functions" begin
    @testset "distprod" begin
        marginals = (
            a = Normal(2, 1),
            b = Exponential(1.3),
            c = [3, 6, 8],
            d = MvNormal([1.6 0.4; 0.4 2.1])
        )
        m = @inferred(distprod(; marginals...))
        @test m isa MeasureBase.ProductMeasure
        @test @inferred(distprod(marginals)) == m

        # Constant marginals contribute no degrees of freedom:
        @test varshape(m) == NamedTupleShape(
            a = ScalarShape{Real}(), b = ScalarShape{Real}(),
            c = ConstValueShape([3, 6, 8]), d = ArrayShape{Real}(2)
        )
        @test getdof(m) == 4

        x = rand(m)
        @test x.c == [3, 6, 8]
        @test logdensityof(m, x) ≈ logpdf(marginals.a, x.a) + logpdf(marginals.b, x.b) + logpdf(marginals.d, x.d)

        m_arr = distprod(Weibull.([3, 5, 2], [1.3, 1.0, 0.7]))
        @test m_arr isa MeasureBase.AbstractProductMeasure
        @test varshape(m_arr) == ArrayShape{Real}(3)
    end

    @testset "deprecated wrappers" begin
        @test (@test_deprecated lbqintegral(logfuncdensity(x -> -x .* x), Normal())) isa PosteriorMeasure

        hd = @test_deprecated distbind(
            NamedTupleDist(a = Normal(2, 1), b = Exponential(1.3)), merge
        ) do v
            NamedTupleDist(c = Normal(v.a, v.b), d = Weibull())
        end
        x = rand(hd)
        @test x isa NamedTuple{(:a, :b, :c, :d)}
        @test logdensityof(hd, x) ≈
            logpdf(Normal(2, 1), x.a) + logpdf(Exponential(1.3), x.b) +
            logpdf(Normal(x.a, x.b), x.c) + logpdf(Weibull(), x.d)
    end
end
