# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Base.Test
using StatsBase

@testset "onlineuvstats" begin
    n = 10
    data  = [(-1)^x*exp(x-n/2) for x in 1:n]
    w = 
    tdata = tuple(data...)
    
    @testset "BAT.OnlineUvMean" begin
        @test typeof(@inferred BAT.OnlineUvMean()) <: AbstractVector{Float64}
        @test typeof(@inferred BAT.OnlineUvMean{Float32}()) <: AbstractVector{Float32}
        ouvm = OnlineUvMean()
        w = [1, 2]
        res = cat(ouvm, data, w[1])
        @test Float64(res.sum_v) ≈ sum(data)
        @test isnan(ouvm[])

        res = cat(res, 0.1*data, w[2])
        @test res[] ≈ mean(vcat(data, 0.1*data), Weights(vcat([x*ones(n) for x in w]...)))
    end
end
