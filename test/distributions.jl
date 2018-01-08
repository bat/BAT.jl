# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Compat.Test

using Distributions, PDMats, StatsBase

struct test_dist <: Distribution{Univariate, Continuous} end
Distributions.sampler(d::test_dist) = Distributions.sampler(Distributions.Normal(0,1))

struct test_batsampler{T} <: BATSampler{T, Continuous} end

function Base.rand!(rng::AbstractRNG, s::test_batsampler, x::Integer)
    return 0.5
end
Base.length(s::test_batsampler{Multivariate}) = 2
function Base.rand!(rng::AbstractRNG, s::test_batsampler, x::AbstractArray{T, 1} where T)
    for i in indices(x)[1]
        x[i] = 1
    end
    return x
 end


@testset "random number generation" begin
    @testset "bat_sampler" begin
        @test bat_sampler(@inferred test_dist()) == Distributions.Normal(0,1)
    end

    @testset "_check_rand_compat" begin
        @test BAT._check_rand_compat(MvNormal(ones(2)), ones(2,10)) == nothing
        @test_throws DimensionMismatch BAT._check_rand_compat(MvNormal(ones(3)), ones(2,10))
    end

    @testset "rand" begin
        bsguv = BATGammaMTSampler(Gamma(4f0, 2f0))
        @test size(rand(bsguv, 5)) == (5,)
        @test typeof(rand(bsguv, 5)) == Vector{Float32}
        @test typeof(rand(bsguv)) == Float32

        res = rand!(bsguv, zeros(3))
        @test size(res) == (3,)
        @test typeof(res) == Array{Float64, 1}
        res = rand!(bsguv, zeros(3,4))
        @test size(res) == (3,4,)
        @test typeof(res) == Array{Float64, 2}

        @test rand!(test_batsampler{Univariate}(), 1) == 0.5
        x = zeros(2,3)
        rand!(test_batsampler{Multivariate}(), x)
        @test x == ones(2,3)

        res = rand(Base.GLOBAL_RNG, bsguv, Dims((2,3)))
        @test typeof(res) == Array{Float32,2}
        @test size(res) == (2,3)

        bstmv = BATMvTDistSampler(MvTDist(1.5, PDMat(diagm(ones(3)))))
        res = rand(bstmv)
        @test typeof(res) == Array{Float64, 1}
        @test size(res) == (3,)
        res = rand(bstmv, Int32(2))
        @test typeof(res) == Array{Float64, 2}
        @test size(res) == (3, 2)
    end


    @testset "issymmetric_around_origin" begin
        @test issymmetric_around_origin(Normal(0.0, 3.2)) == true
        @test issymmetric_around_origin(Normal(1.7, 3.2)) == false
        @test issymmetric_around_origin(Gamma(4.2, 2.2)) == false
        @test issymmetric_around_origin(Chisq(20.3)) == false
        @test issymmetric_around_origin(MvNormal(zeros(2), ones(2))) == true
        @test issymmetric_around_origin(MvNormal(ones(2), ones(2))) == false
        @test issymmetric_around_origin(MvTDist(1.5, zeros(2), PDMat(diagm(ones(2)))))
        @test issymmetric_around_origin(MvTDist(1.5, ones(2), PDMat(diagm(ones(2))))) == false
    end


    @testset "BATGammaMTSampler" begin
        @test typeof(@inferred bat_sampler(Gamma(0.5, 20.3))) <: BATGammaMTSampler

        @test rand(MersenneTwister(7002), BATGammaMTSampler(Gamma(0.5, 20.3))) ≈ 0.872195443488103
        @test rand(MersenneTwister(8456), BATGammaMTSampler(Gamma(0.5, 20.3))) ≈ 0.2723580272080763
        @test rand(MersenneTwister(3296), BATGammaMTSampler(Gamma(0.5, 20.3))) ≈ 6.984079300633383
        @test rand(MersenneTwister(5065), BATGammaMTSampler(Gamma(0.5, 20.3))) ≈ 9.387736826614171
        @test rand(MersenneTwister(7002), BATGammaMTSampler(Gamma(0.25, 2))) ≈ 0.0031794482969992074
        @test rand(MersenneTwister(7002), BATGammaMTSampler(Gamma(10, 2))) ≈ 4.923284832865044

        @test typeof(rand(BATGammaMTSampler(Gamma(4.2f0, 2.2f0)))) == Float32
        @test eltype(BATGammaMTSampler(Gamma(4.2f0, 2.2f0))) == Float32
        @test eltype(BATGammaMTSampler(Gamma(4.2, 2.2))) == Float64
        @test typeof(rand(BATGammaMTSampler(Gamma(4.2, 2.2)))) == Float64
    end


    @testset "BATChisqSampler" begin
        @test typeof(@inferred bat_sampler(Chisq(0.5))) <: BATChisqSampler

        @test rand(MersenneTwister(7002), BATChisqSampler(Chisq(0.5))) ≈ 0.0031794482969992074
        @test rand(MersenneTwister(7002), BATChisqSampler(Chisq(20))) ≈ 4.923284832865044

        @test eltype(BATChisqSampler(Chisq(20f0))) == Float32
        @test typeof(rand(BATChisqSampler(Chisq(20f0)))) == Float32
    end


    @testset "BATMvTDistSampler" begin
        d = MvTDist(1.5, PDMat([2.0 1.0; 1.0 3.0]))

        @test typeof(@inferred bat_sampler(d)) <: BATMvTDistSampler
        @test size(@inferred rand(bat_sampler(d), 5)) == (2, 5)

        @test size(rand!(bat_sampler(d), ones(2))) == (2, )
        @test size(rand!(bat_sampler(d), ones(2, 3))) == (2, 3)

        cmat = [3.76748 0.446731 0.625418; 0.446731 3.9317 0.237361; 0.625418 0.237361 3.43867]
        tmean = [1., 2, 3]
        tmv = MvTDist(3, tmean, PDMat(eye(3)))

        tmv = BAT.set_cov!(tmv, cmat)
        @test full(BAT.get_cov(tmv)) ≈ cmat

        bstmv = BATMvTDistSampler(tmv)

        n = 1000
        res = rand(MersenneTwister(7002), bstmv, n)

        @test isapprox(mean(res,2), tmean; atol = 0.5)
        @test isapprox(cov(res, 2)/3, cmat; atol = 1.5)
    end
end
