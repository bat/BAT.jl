# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random
using Distributions, StatsBase


@testset "histograms_as_distributions" begin
    @testset "HistogramAsUvDistribution" begin
        μ, σ = 1.23, 0.74
        true_dist = Normal(μ, σ)
        h = Histogram(μ-10σ:σ/10:μ+10σ)
        Random.seed!(123)
        append!(h, rand(true_dist, 10^7))
        d = BAT.HistogramAsUvDistribution(h)
        @test isapprox(μ, d.μ, rtol = 0.01)
        @test isapprox(σ, d.σ, rtol = 0.01)
        fit_dist = fit(Normal, rand(d, 10^7))
        @test isapprox(μ, fit_dist.μ, rtol = 0.01)
        @test isapprox(σ, fit_dist.σ, rtol = 0.01)
    end  
end
