# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Base.Test
using StatsBase

@testset "onlinestats" begin
    n = 10
    data1  = [(-1)^x*exp(x-n/2) for x in 1:n]
    data2 = flipdim(data1, 1)

    w = [exp(x-n/2) for x in 1:n]

    data = hcat(data1, data2)
    m = size(data)[2]

    
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
        @test mvmean1 ≈ mean(hcat(data1, data2), Weights([w1, w2]), 2)

        push!(mvmean2, data2, w2)

        merge!(mvmean1, mvmean2)
        @test mvmean1 ≈ mean(hcat(data1, data2, data2), Weights([w1, w2, w2]), 2)
    end

    @testset "OnlineMvCov" begin
        @test typeof(@inferred BAT.OnlineMvCov(n)) <: AbstractMatrix{Float64}
        @test typeof(@inferred BAT.OnlineMvCov{Float32, ProbabilityWeights}(n)) <: AbstractMatrix{Float32}

        for wKind in [ProbabilityWeights, FrequencyWeights, AnalyticWeights, Weights]
            mvcov = BAT.OnlineMvCov{Float64, wKind}(m)
            for i in indices(data, 1)
                push!(mvcov, data[i,:], w[i]);
            end

            @test mvcov ≈ cov(data, wKind(w), 1; corrected = (wKind != Weights))
        end

        countMvCovs = 3
        mvcovs = Array{BAT.OnlineMvCov{Float64, ProbabilityWeights}}(countMvCovs)
        for i in indices(mvcovs,1)
            mvcovs[i] = BAT.OnlineMvCov(m)
        end
        mvcovc = BAT.OnlineMvCov(m)
        
        for i in indices(data, 1)
            push!(mvcovs[(i % countMvCovs) + 1], data[i, :], w[i]);
            push!(mvcovc, data[i, :], w[i]);
        end

        res = merge(mvcovs[1], mvcovs[2], mvcovs[3])
        @test res ≈ cov(data, ProbabilityWeights(w), 1; corrected = true)
        @test res ≈ mvcovc
        @test res.sum_w ≈ mvcovc.sum_w
        @test res.sum_w2 ≈ mvcovc.sum_w2
        @test res.n ≈ mvcovc.n
        @test res.Mean_X ≈ mvcovc.Mean_X
        @test res.New_Mean_X ≈ mvcovc.New_Mean_X
    end
    @testset "BasicMvStatistic" begin        
        bmvstats = BasicMvStatistics{Float64, ProbabilityWeights}(
            OnlineMvMean{Float64}(m), OnlineMvCov{Float64, ProbabilityWeights}(m),
            -Inf*ones(Float64, m), Inf*ones(Float64, m)
        )

        countBMS = 3
        bmvs = Array{BAT.BasicMvStatistics{Float64, ProbabilityWeights}}(countBMS)
        for i in indices(bmvs,1)
            bmvs[i] = BasicMvStatistics{Float64, ProbabilityWeights}(
            OnlineMvMean{Float64}(m), OnlineMvCov{Float64, ProbabilityWeights}(m),
            -Inf*ones(Float64, m), Inf*ones(Float64, m)
            )
        end
        
        for i in indices(data, 1)
            BAT.push_contiguous!(bmvstats, data[i,:], 1, w[i]);
            BAT.push_contiguous!(bmvs[(i % countBMS) + 1], data[i, :], 1, w[i]);
        end

        merbmvstats = merge!(bmvs[1], bmvs[2], bmvs[3])

        for bs in [bmvstats, merbmvstats]
            @test bs.mean ≈  mean(data, Weights(w), 1)'
            @test bs.cov ≈ cov(data, ProbabilityWeights(w), 1; corrected = true)
            @test bs.maximum ≈ [maximum(data[:,i]) for i in indices(data, 2)]
            @test bs.minimum ≈ [minimum(data[:,i]) for i in indices(data, 2)]
        end
        
    end
end
