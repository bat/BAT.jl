# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Base.Test
using StatsBase

@testset "onlinestats" begin
    n = 6
    data1  = [(-1)^x*exp(x-n/2) for x in 1:n]
    data2 = flipdim(data1, 1) 

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
        @test typeof(@inferred BAT.OnlineMvMean(n)) <: AbstractVector{Float64}
        @test typeof(@inferred BAT.OnlineMvMean{Float32}(n)) <: AbstractVector{Float32}

        mvmean1 = BAT.OnlineMvMean(n)
        mvmean2 = BAT.OnlineMvMean(n)

        @test size(mvmean1)[1] == n
        
        w1 = 0.5
        w2 = 1.2

        push!(mvmean1, data1, w1)
        @test mvmean1 ≈ data1
        push!(mvmean1, data2, w2)
        @test mvmean1 ≈ (w1*data1 + w2*data2)/(w1 + w2)

        push!(mvmean2, data2, w2)
        merge!(mvmean1, mvmean2)
        @test mvmean1 ≈ (w1*data1 + w2*data2 + w2*data2)/(w1 + w2 + w2)
    end
    @testset "OnlineMvCov" begin
        @test typeof(@inferred BAT.OnlineMvCov(n)) <: AbstractMatrix{Float64}
        @test typeof(@inferred BAT.OnlineMvCov{Float32, ProbabilityWeights}(n)) <: AbstractMatrix{Float32}
        data = hcat(1.:3,1.:2:2*3)
        mvcov = BAT.OnlineMvCov{Float64, ProbabilityWeights}(2)
        for i in indices(data, 1)
            push!(mvcov, data[i,:]);
        end

        @test mvcov ≈ cov(data, ProbabilityWeights(ones(3)), 1; corrected = true)
    end
end
