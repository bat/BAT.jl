# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Base.Test

@testset "onlinestats" begin
    @testset "BAT.kbn_add" begin
        bg = +1E5
        sm = +1e-4
        res = bg + sm

        bgT = (bg, 0.0)
        smT = (sm, 0.0)

        @test collect(BAT.kbn_add((0.0,0.0),0.0)) ≈ [0.0,0.0]
        @test sum(BAT.kbn_add((0.0,0.0),eps())) ≈ eps()
        @test sum(BAT.kbn_add((bg,0.0),sm)) ≈ res
        @test sum(BAT.kbn_add((sm,0.0),bg)) ≈ res

        @test sum(BAT.kbn_add(bgT,smT)) ≈ res

        @test typeof(@inferred BAT.kbn_add(bgT,sm)) == NTuple{2, Float64}
        @test typeof(@inferred BAT.kbn_add(bgT,smT)) == NTuple{2, Float64}
        @test typeof(@inferred BAT.kbn_add((0.0f0, 0.0f0), 0.0f0)) == NTuple{2, Float32}
        @test typeof(@inferred BAT.kbn_add((0.0f0, 0.0f0), (0.0f0, 0.0f0))) == NTuple{2, Float32}
    end
    @testset "BAT.OnlineMvMean" begin
        n = 3
        @test typeof(@inferred BAT.OnlineMvMean(n)) <: AbstractVector{Float64}
        @test typeof(@inferred BAT.OnlineMvMean{Float32}(n)) <: AbstractVector{Float32}

        mvmean1 = BAT.OnlineMvMean(n)
        mvmean2 = BAT.OnlineMvMean(n)

        @test size(mvmean1)[1] == n
    end
end
