# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using ValueShapes, Distributions, ForwardDiff

@testset "test_generic_transforms" begin
    @testset "uniform_cdf" begin
        a = -1.0
        b = 3.0
        x = 2
        u = cdf(Uniform(a, b), x)

        @test @inferred(BAT.uniform_cdf(x, a, b)) ≈ u
        @test @inferred(BAT.uniform_invcdf(u, a, b)) ≈ x
        @test BAT.uniform_invcdf_ladj(u, a, b) ≈ log(abs(ForwardDiff.derivative(u -> BAT.uniform_invcdf(u, a, b), u)))

        @test @inferred(BAT.UniformCDFTrafo(a, b)) isa BAT.VariateTransform{Univariate,BAT.UnitSpace,BAT.MixedSpace}
        trafo = BAT.UniformCDFTrafo(a, b)
        @test @inferred(varshape(trafo)) == ScalarShape{Real}()
        @test BAT.target_space(trafo) == BAT.UnitSpace()
        @test BAT.source_space(trafo) == BAT.MixedSpace()

        @test @inferred(BAT.apply_vartrafo(trafo, x, 0)).v ≈ u
        @test BAT.apply_vartrafo(trafo, x, 0).ladj ≈ log(abs(ForwardDiff.derivative(x -> cdf(Uniform(a, b), x), x)))
        @test isnan(BAT.apply_vartrafo(trafo, x, NaN).ladj)
        @test @inferred(BAT.apply_vartrafo(inv(trafo), u, 0)).v ≈ x
        @test BAT.apply_vartrafo(inv(trafo), u, 0).ladj ≈ log(abs(ForwardDiff.derivative(u -> quantile(Uniform(a, b), u), u)))
        @test isnan(BAT.apply_vartrafo(inv(trafo), u, NaN).ladj)


        @test @inferred(inv(trafo)) isa BAT.VariateTransform{Univariate,BAT.MixedSpace,BAT.UnitSpace}
        @test @inferred(BAT.apply_vartrafo(inv(trafo) ∘ trafo, x, 0)).v ≈ x
        @test BAT.apply_vartrafo(inv(trafo) ∘ trafo, x, 0.79).ladj ≈ 0.79

        @test @inferred(BAT.apply_vartrafo(inv(trafo ∘ inv(trafo)), u, 0)).v ≈ u
        @test BAT.apply_vartrafo(inv(trafo ∘ inv(trafo)), u, 0.79).ladj ≈ 0.79
    end

    @testset "logistic_cdf" begin
        mu = 2.0
        theta = 3.0
        x = 4
        u = cdf(Logistic(mu, theta), x)

        @test @inferred(BAT.logistic_cdf(x, mu, theta)) ≈ u
        @test @inferred(BAT.logistic_invcdf(u, mu, theta)) ≈ x
        @test BAT.logistic_invcdf_ladj(u, mu, theta) ≈ log(abs(ForwardDiff.derivative(u -> BAT.logistic_invcdf(u, mu, theta), u)))

        @test @inferred(BAT.LogisticCDFTrafo(mu, theta)) isa BAT.VariateTransform{Univariate,BAT.UnitSpace,BAT.InfiniteSpace}
        trafo = BAT.LogisticCDFTrafo(mu, theta)
        @test @inferred(varshape(trafo)) == ScalarShape{Real}()
        @test BAT.target_space(trafo) == BAT.UnitSpace()
        @test BAT.source_space(trafo) == BAT.InfiniteSpace()

        @test @inferred(BAT.apply_vartrafo(trafo, x, 0)).v ≈ u
        @test BAT.apply_vartrafo(trafo, x, 0).ladj ≈ log(abs(ForwardDiff.derivative(x -> cdf(Logistic(mu, theta), x), x)))
        @test isnan(BAT.apply_vartrafo(trafo, x, NaN).ladj)
        @test @inferred(BAT.apply_vartrafo(inv(trafo), u, 0)).v ≈ x
        @test BAT.apply_vartrafo(inv(trafo), u, 0).ladj ≈ log(abs(ForwardDiff.derivative(u -> quantile(Logistic(mu, theta), u), u)))
        @test isnan(BAT.apply_vartrafo(inv(trafo), u, NaN).ladj)


        @test @inferred(inv(trafo)) isa BAT.VariateTransform{Univariate,BAT.InfiniteSpace,BAT.UnitSpace}
        @test @inferred(BAT.apply_vartrafo(inv(trafo) ∘ trafo, x, 0.79)).v ≈ x
        @test BAT.apply_vartrafo(inv(trafo) ∘ trafo, x, 0.79).ladj ≈ 0.79

        @test @inferred(BAT.apply_vartrafo(inv(trafo ∘ inv(trafo)), u, 0.79)).v ≈ u
        @test BAT.apply_vartrafo(inv(trafo ∘ inv(trafo)), u, 0.79).ladj ≈ 0.79
    end


    @testset "exponential_cdf" begin
        theta = 3.0
        x = 5
        u = cdf(Exponential(theta), x)

        @test @inferred(BAT.exponential_cdf(x, theta)) ≈ u
        @test @inferred(BAT.exponential_invcdf(u, theta)) ≈ x
        @test BAT.exponential_cdf_ladj(x, theta) ≈ log(abs(ForwardDiff.derivative(x -> BAT.exponential_cdf(x, theta), x)))

        @test @inferred(BAT.ExponentialCDFTrafo(theta)) isa BAT.VariateTransform{Univariate,BAT.UnitSpace,BAT.MixedSpace}
        trafo = BAT.ExponentialCDFTrafo(theta)
        @test @inferred(varshape(trafo)) == ScalarShape{Real}()
        @test BAT.target_space(trafo) == BAT.UnitSpace()
        @test BAT.source_space(trafo) == BAT.MixedSpace()

        @test @inferred(BAT.apply_vartrafo(trafo, x, 0)).v ≈ u
        @test BAT.apply_vartrafo(trafo, x, 0).ladj ≈ log(abs(ForwardDiff.derivative(x -> cdf(Exponential(theta), x), x)))
        @test isnan(BAT.apply_vartrafo(trafo, x, NaN).ladj)
        @test @inferred(BAT.apply_vartrafo(inv(trafo), u, 0)).v ≈ x
        @test BAT.apply_vartrafo(inv(trafo), u, 0).ladj ≈ log(abs(ForwardDiff.derivative(u -> quantile(Exponential(theta), u), u)))
        @test isnan(BAT.apply_vartrafo(inv(trafo), u, NaN).ladj)


        @test @inferred(inv(trafo)) isa BAT.VariateTransform{Univariate,BAT.MixedSpace,BAT.UnitSpace}
        @test @inferred(BAT.apply_vartrafo(inv(trafo) ∘ trafo, x, 0.79)).v ≈ x
        @test BAT.apply_vartrafo(inv(trafo) ∘ trafo, x, 0.79).ladj ≈ 0.79

        @test @inferred(BAT.apply_vartrafo(inv(trafo ∘ inv(trafo)), u, 0.79)).v ≈ u
        @test BAT.apply_vartrafo(inv(trafo ∘ inv(trafo)), u, 0.79).ladj ≈ 0.79
    end

    @testset "scaled_log" begin
        theta = 3.0
        x = 4
        y = log(x / theta)
        
        @test @inferred(BAT.ScaledLogTrafo(theta)) isa BAT.VariateTransform{Univariate,BAT.InfiniteSpace,BAT.MixedSpace}
        trafo = BAT.ScaledLogTrafo(theta)
        @test @inferred(varshape(trafo)) == ScalarShape{Real}()
        @test BAT.target_space(trafo) == BAT.InfiniteSpace()
        @test BAT.source_space(trafo) == BAT.MixedSpace()
        
        @test @inferred(BAT.apply_vartrafo(trafo, x, 0)).v ≈ y
        @test BAT.apply_vartrafo(trafo, x, 0).ladj ≈ log(abs(ForwardDiff.derivative(trafo, x)))
        @test isnan(BAT.apply_vartrafo(trafo, x, NaN).ladj)
        @test @inferred(BAT.apply_vartrafo(inv(trafo), y, 0)).v ≈ x
        @test BAT.apply_vartrafo(inv(trafo), y, 0).ladj ≈ log(abs(ForwardDiff.derivative(inv(trafo), y)))
        @test isnan(BAT.apply_vartrafo(inv(trafo), y, NaN).ladj)
    end
end
