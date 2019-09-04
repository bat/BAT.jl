# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random
using Distributions, StatsBase


@testset "HistogramAsUvDistribution" begin
    Random.seed!(123)
    μ, σ = 1.23, 0.74
    true_dist = Normal(μ, σ)
    h = Histogram(μ-10σ:σ/10:μ+10σ)
    append!(h, rand(true_dist, 10^7))
    d = BAT.HistogramAsUvDistribution(h)
    @test isapprox(μ, d.μ, atol = 0.01)
    @test isapprox(σ, d.σ, atol = 0.01)
    fit_dist = fit(Normal, rand(d, 10^7))
    @test isapprox(μ, fit_dist.μ, atol = 0.01)
    @test isapprox(σ, fit_dist.σ, atol = 0.01)
end  
