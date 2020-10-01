# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "distributions" begin
    include("test_distribution_functions.jl")

    Test.@testset "multimodal cauchy" begin
        @test_throws ArgumentError BAT.MultimodalCauchy(n=1)
        for i in 2:6
            μ = randn()
            σ = rand(0.01:0.1:2)
            mmc = BAT.MultimodalCauchy(μ=μ,σ=σ, n=i)
            @test isequal(mean(mmc), fill(NaN, i))
            @test isequal(var(mmc), fill(NaN, i))
            @test isequal(diag(cov(mmc)), fill(NaN, i))
            ps = StatsBase.params(mmc)
            @test sort(ps[1]) == sort(vcat([μ, -μ], zeros(i-2)))
            @test ps[2] == [σ for j in 1:i]
            @test ps[3] == i
        end
    end

    Test.@testset "funnel distribution" begin
        for i in 2:6
            a = randn()
            b = randn()
            funnel = BAT.FunnelDistribution(a=a, b=b, n=i)
            @test mean(funnel) == zeros(i)
            @test StatsBase.params(funnel) == (a, b, i)
        end
    end
end
