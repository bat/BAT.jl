# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random, Statistics, LinearAlgebra
using Distributions
using QuadGK, ForwardDiff
using StableRNGs
using StatsBase, FillArrays 


@testset "standard_normal" begin
    stblrng() = StableRNG(789990641)

    @test @inferred(BAT.LogUniform(0.1f0, 100)) isa BAT.LogUniform{Float32}
    @test_throws ArgumentError BAT.LogUniform(0, 100)
    @test_throws ArgumentError BAT.LogUniform(100, 10)

    d = BAT.LogUniform(0.1, 100)
    X = @inferred(rand(stblrng(), d, 10^5))
    
    @test BAT.LogUniform() == BAT.LogUniform{Float64}(1, 2)  

    @test params(d) == (0.1, 100)
    @test @inferred(minimum(d)) == 0.1
    @test @inferred(maximum(d)) == 100

    @test Distributions.location(d) == d.a 
    @test Distributions.scale(d) == d.b - d.a 

    @test isapprox(@inferred(quantile(d, 0.3)), quantile(X, 0.3), rtol = 0.05)
    @test quadgk(x -> pdf(d, x), minimum(d), maximum(d))[1] ≈ 1
    @test cdf(d, minimum(d)) ≈ 0
    @test cdf(d, maximum(d)) ≈ 1
    
    @test isapprox(@inferred(mean(d)), mean(X), rtol = 0.05)
    @test isapprox(@inferred(median(d)), median(X), rtol = 0.05)
    @test @inferred(mode(d)) == minimum(d)
    @test isapprox(@inferred(var(d)), var(X), rtol = 0.05)
    @test isapprox(@inferred(std(d)), std(X), rtol = 0.05)

    @test StatsBase.modes(d) == Fill(mode(d),0) 
    @test Distributions.logccdf(d, 90) == log(ccdf(d, 90)) 
    @test Distributions.ccdf(d, 90) == one(cdf(d, 90)) - cdf(d, 90) 

    @test Distributions.cquantile(d, 0.5) == one(quantile(d, 0.5)) - quantile(d, 0.5) 

    @test Distributions.truncated(d, 10 + 1 / 3, 90 + 1 / 3) == BAT.LogUniform(promote(max(10 + 1 / 3,d.a), min(90 + 1 /3, d.b))...) 
    
    for x in (minimum(d), maximum(d), (minimum(d) + maximum(d)) / 2.9, (minimum(d) + maximum(d)) / 5.2)
        @test @inferred(quantile(d, @inferred(cdf(d, x)))) ≈ x
        @test ForwardDiff.derivative(x -> cdf(d, x), x) ≈ @inferred(pdf(d, x))
        @test log(pdf(d, x)) ≈ @inferred(logpdf(d, x))
        @test log(cdf(d, x)) ≈ @inferred(logcdf(d, x))
    end
end
