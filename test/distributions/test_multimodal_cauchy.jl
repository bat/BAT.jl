# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test
using Statistics, StatsBase, Distributions

@testset "Multimodal Tests" begin
    cm = MixtureModel([Distributions.Cauchy(), Distributions.Cauchy()])
    prod = Distributions.Product([Distributions.Cauchy(), Distributions.Cauchy()])

    #Instantiates multimodal Cauchy with Mixture Model and Product
    @test @inferred(BAT.MultimodalCauchy(cm, 0.1, 2, prod)) isa BAT.MultimodalCauchy
    #Instantiates multimodal Cauchy with μ, σ and n
    @test @inferred(BAT.MultimodalCauchy(μ = 1., σ = 0.1, n=4)) isa BAT.MultimodalCauchy
    #Error if n = 1
    @test_throws ArgumentError  BAT.MultimodalCauchy(μ = 1., σ = 0.1, n=1)

    mmc = BAT.MultimodalCauchy(μ = 1., σ = 0.1, n=4)

    @test @inferred(Base.size(mmc)) == (4,)
    @test @inferred(Base.length(mmc)) == 4
    @test @inferred(Base.eltype(mmc)) == Float64

    params = @inferred(StatsBase.params(mmc))
    
    #Parameters
    @test params[1] == [-1., 1., 0., 0.]
    @test params[2] == [0.1, 0.1, 0.1, 0.1]
    @test params[3] == 4

    #logpdf
    @test @inferred(Distributions._logpdf(mmc, [-1., 0., 1., 2.])) == -11.283438151658

    #If μ = 0 the Distribution should be Cauchy-like in every dimension
    mmc = BAT.MultimodalCauchy(μ = 0., σ = 0.1, n=2)
    ks_test = HypothesisTests.ExactOneSampleKSTest(rand(mmc, 10^6)[1,:], Cauchy(0., 0.1))#First dimension
    @test pvalue(ks_test) > 0.05
    ks_test = HypothesisTests.ExactOneSampleKSTest(rand(mmc, 10^6)[2,:], Cauchy(0., 0.1))#Second dimension
    @test pvalue(ks_test) > 0.05
end