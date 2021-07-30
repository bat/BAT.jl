# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test
using Statistics, StatsBase, Distributions
using HypothesisTests

@testset "Multimodal Tests" begin
    cm = MixtureModel([Distributions.TDist(1), Distributions.TDist(1)])
    prod = Distributions.Product([Distributions.TDist(1), Distributions.TDist(1)])

    #Instantiates multimodal Cauchy with Mixture Model and Product
    @test @inferred(BAT.MultimodalStudentT(cm, 0.1, 2, prod)) isa BAT.MultimodalStudentT
    #Instantiates multimodal Cauchy with μ, σ and n
    @test @inferred(BAT.MultimodalStudentT(μ=60, σ=1., d = 3, n = 2)) isa BAT.MultimodalStudentT
    #Error if n = 1
    @test_throws ArgumentError  BAT.MultimodalStudentT(μ=60, σ=1., d = 3, n = 1)

    mt = BAT.MultimodalStudentT(μ = 1., σ = 0.1, d = 1, n=4)

    @test @inferred(Base.size(mt)) == (4,)
    @test @inferred(Base.length(mt)) == 4
    @test @inferred(Base.eltype(mt)) == Float64

    params = @inferred(StatsBase.params(mt))
    
    #Parameters
    @test params[1] == [-1., 1., 0., 0.]
    @test params[2] == [0.1, 0.1, 0.1, 0.1]
    @test params[3] == 4

    #logpdf
    @test @inferred(Distributions._logpdf(mt, [-1., 0., 1., 2.])) == -11.283438151658

    #If μ = 0 the Distribution should be Cauchy-like in every dimension
    mmc = BAT.MultimodalStudentT(μ = 0., σ = 0.1, d = 1, n=2)
    ks_test = HypothesisTests.ExactOneSampleKSTest(rand(mmc, 10^6)[1,:], Cauchy(0., 0.1))#First dimension
    @test pvalue(ks_test) > 0.05
    ks_test = HypothesisTests.ExactOneSampleKSTest(rand(mmc, 10^6)[2,:], Cauchy(0., 0.1))#Second dimension
    @test pvalue(ks_test) > 0.05
end