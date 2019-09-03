# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random
using Distributions, StatsBase


@testset "HistogramAsMvDistribution" begin
    Random.seed!(123)
    μ = [1.23, -0.67]
    Σ = [0.45 0.32; 0.32 0.76]' * [0.45 0.32; 0.32 0.76]
    true_dist = MvNormal(μ, Σ)
    h = Histogram((μ[1]-10Σ[1]:Σ[1]/10:μ[1]+10Σ[1], μ[2]-10Σ[4]:Σ[4]/10:μ[2]+10Σ[4]))
    n = 10^6
    r = rand(true_dist, n)
    append!(h, (r[1, :], r[2, :])) 
    d = BAT.HistogramAsMvDistribution(h)
    @test all(isapprox.(μ, d.μ,   atol = 0.01))       
    @test all(isapprox.(Σ, d.cov, atol = 0.01))       
    rand!(d, r)
    fit_dist = fit(MvNormal, r)
    @test all(isapprox.(μ, fit_dist.μ,     atol = 0.01))       
    @test all(isapprox.(Σ, fit_dist.Σ.mat, atol = 0.01))       
end  
