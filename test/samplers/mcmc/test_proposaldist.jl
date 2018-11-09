# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using LinearAlgebra
using Distributions
using Test
using PDMats

struct test_mvdist <: Distribution{Multivariate, Continuous}
    d::Distribution{Multivariate, Continuous}
end

test_sampler(tmv::test_mvdist) = tmv.d

@testset "proposaldist" begin
    @testset "GenericProposalDist" begin
        gpd = GenericProposalDist{test_mvdist, typeof(test_sampler)}(
             test_mvdist(MvNormal([0.0, 1.0], [1. 0.5; 0.5 2])),
             test_sampler)

        @test typeof(gpd.d) <: Distribution{Multivariate, Continuous}
        @test typeof(gpd.s) <: MvNormal
        @test gpd.s.μ ≈ [0.0, 1.0]
        @test gpd.sampler_f == test_sampler

        gpd2 = GenericProposalDist(
             test_mvdist(MvNormal([0.0, 1.0], [1. 0.5; 0.5 2])),
             test_sampler)

        @test typeof(gpd.d) <: Distribution{Multivariate, Continuous}
        @test typeof(gpd.s) <: MvNormal
        @test gpd.s.μ ≈ [0.0, 1.0]
        @test gpd.sampler_f == test_sampler

        mvN_gpd = @inferred GenericProposalDist(MvNormal([0.0, 1.0], [1. 0.5; 0.5 2]))
        @test mvN_gpd.sampler_f == bat_sampler
        @test typeof(mvN_gpd.d) <: MvNormal

        mv_gpd = @inferred GenericProposalDist(MvTDist, 2, 5)
        @test typeof(mv_gpd.d) <: MvTDist
        @test length(mv_gpd.d) == 2
        @test mv_gpd.d.df == 5

        gpd3 = similar(gpd2, test_mvdist(MvNormal([0.0, 1.0], [1. 0.5; 0.5 2])))
        @test gpd3.sampler_f == test_sampler
        @test typeof(gpd3.d) <: test_mvdist

        gpd3 = @inferred similar(mv_gpd, MvNormal([0.0, 1.0], [1. 0.5; 0.5 2]))
        @test gpd3.sampler_f == bat_sampler
        @test typeof(gpd3.d) <: MvNormal

        apd = @inferred convert(AbstractProposalDist, gpd3, Float64, 2)
        @test typeof(apd) <: AbstractProposalDist
        @test_throws ArgumentError convert(AbstractProposalDist,
            gpd3, Float64, 3)

        d = MvTDist(1.5, zeros(2), PDMat([1. 0.5; 0.5 2]))
        gpd = @inferred GenericProposalDist(d)
        @test Matrix(BAT.get_cov(gpd)) ≈ [1. 0.5; 0.5 2]
        gpd = @inferred BAT.set_cov(gpd, Matrix(ScalMat(2, 1)))

        @test Matrix(BAT.get_cov(gpd)) ≈ Matrix{Float64}(I, 2, 2)

        @test nparams(gpd) == 2
        @test issymmetric(gpd)

        d = MvTDist(1.5, ones(2), PDMat([1. 0.5; 0.5 2]))
        gpd = @inferred GenericProposalDist(d)
        @test !issymmetric(gpd)

        gpd = @inferred GenericProposalDist(MvTDist, Float64, 3, 2.0)
        @test nparams(gpd) == 3
        @test typeof(gpd) <: GenericProposalDist{MvTDist, typeof(bat_sampler)}
        @test gpd.d.df ≈ 2.0

    end

    @testset "distribution_logpdf" begin
        d = MvTDist(1.5, zeros(2), PDMat([1. 0.5; 0.5 2]))
        gpd = @inferred GenericProposalDist(d)

        p = zeros(3)

        distribution_logpdf!(p,
            gpd, VectorOfSimilarVectors([0.0 -0.5 1.5;0.0 0.5 0.0]), [0.1,-0.1])
        @test p ≈ [-2.1441505, -2.8830174, -3.6814116]

        p = zeros(1)
        distribution_logpdf!(p,
            gpd, [0.0,0.0], [0.1,-0.1])
        @test p ≈ [-2.1441505]

        @test distribution_logpdf(
            gpd, [0.0,0.0], [0.1,-0.1]) ≈ -2.1441505
    end

    @testset "proposal_rand" begin
        d = MvTDist(1.5, zeros(2), PDMat([1. 0.5; 0.5 2]))
        gpd = @inferred GenericProposalDist(d)

        proposal_rand!(MersenneTwister(7002), gpd, zeros(2), [2.5, -1.0]) ≈
            [1.942373680136, 3.8454891831]
    end

    @testset "ProposalDistSpec" begin
        mvTps = @inferred MvTDistProposalSpec(4.0)
        @test mvTps.df ≈ 4.0
        @test MvTDistProposalSpec().df ≈ 1.0

        gpd = @inferred mvTps(Float64, 3)
        @test nparams(gpd) == 3
        @test typeof(gpd) <: GenericProposalDist{MvTDist, typeof(bat_sampler)}
        @test gpd.d.df ≈ 4.0
    end
end
