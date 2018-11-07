# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random
using Distributions, PDMats, StatsBase


struct test_dist <: Distribution{Univariate, Continuous} end
Distributions.sampler(d::test_dist) = Distributions.sampler(Distributions.Normal(0,1))


struct test_batsampler{T} <: BATSampler{T, Continuous} end

function Random.rand!(rng::AbstractRNG, s::test_batsampler, x::Integer)
    return 0.5
end

Base.length(s::test_batsampler{Multivariate}) = 2
function Random.rand!(rng::AbstractRNG, s::test_batsampler, x::AbstractArray{T, 1} where T)
    for i in axes(x)[1]
        x[i] = 1
    end
    return x
end


@testset "bat_sampler" begin
    @test bat_sampler(@inferred test_dist()) == Distributions.Normal(0,1)

    @test BAT.rand!(test_batsampler{Univariate}(), 1) == 0.5
    x = zeros(2,3)
    BAT.rand!(test_batsampler{Multivariate}(), x)
    @test x == ones(2,3)
end
