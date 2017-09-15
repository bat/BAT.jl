# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Base.Test
using StatsBase

@testset "onlinestats" begin
    n = 10
    data1  = [(-1)^x*exp(x-n/2) for x in 1:n]
    data2 = flipdim(data1, 1)

    w = [exp(x-n/2) for x in 1:n]

    data = vcat(data1', data2')
    m = size(data, 1)

    
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
        
        mvmean = BAT.OnlineMvMean(m)
        @test size(mvmean, 1) == m

        countM = 3
        mvmeans = Array{BAT.OnlineMvMean{Float64}}(countM)

        for i in indices(mvmeans, 1)
            mvmeans[i] = BAT.OnlineMvMean(m)
        end

        for i in indices(data, 2)
            push!(mvmean, data[:, i], w[i]);
            push!(mvmeans[(i % countM) + 1], data[:, i], w[i]);
        end

        @test mvmean ≈ mean(data, Weights(w), 2)

        res = merge(mvmeans...)
        @test res ≈ mean(data, Weights(w), 2)

        mvmean = BAT.OnlineMvMean(m)
        res = append!(deepcopy(mvmean), data, 2)
        @test res ≈ mean(data, 2)
        res = append!(deepcopy(mvmean), data, w, 2)
        @test res ≈ mean(data, Weights(w), 2)
    end

    @testset "OnlineMvCov" begin
        @test typeof(@inferred BAT.OnlineMvCov(n)) <: AbstractMatrix{Float64}
        @test typeof(@inferred BAT.OnlineMvCov{Float32, ProbabilityWeights}(n)) <: AbstractMatrix{Float32}

        for wKind in [ProbabilityWeights, FrequencyWeights, AnalyticWeights, Weights]
            mvcov = BAT.OnlineMvCov{Float64, wKind}(m)
            for i in indices(data, 2)
                push!(mvcov, data[:, i], w[i]);
            end

            @test mvcov ≈ cov(data, wKind(w), 2; corrected = (wKind != Weights))
        end

        countMvCovs = 3
        mvcovs = Array{BAT.OnlineMvCov{Float64, ProbabilityWeights}}(countMvCovs)
        for i in indices(mvcovs,1)
            mvcovs[i] = BAT.OnlineMvCov(m)
        end
        mvcovc = BAT.OnlineMvCov(m)
        
        for i in indices(data, 2)
            push!(mvcovs[(i % countMvCovs) + 1], data[:, i], w[i]);
            push!(mvcovc, data[:, i], w[i]);
        end

        res = merge(mvcovs...)
        @test res ≈ cov(data, ProbabilityWeights(w), 2; corrected = true)
        @test res ≈ mvcovc
        @test res.sum_w ≈ mvcovc.sum_w
        @test res.sum_w2 ≈ mvcovc.sum_w2
        @test res.n ≈ mvcovc.n
        @test res.Mean_X ≈ mvcovc.Mean_X
        @test res.New_Mean_X ≈ mvcovc.New_Mean_X

        mvcov = BAT.OnlineMvCov(m)
        res = append!(deepcopy(mvcov), data, 2)
        @test res ≈ cov(data, 2)
        res = append!(deepcopy(mvcov), data, w, 2)
        @test res ≈ cov(data, ProbabilityWeights(w), 2; corrected = true)
    end
    @testset "BasicMvStatistic" begin        
        bmvstats = BasicMvStatistics{Float64, ProbabilityWeights}(m)

        countBMS = 3
        bmvs = Array{BAT.BasicMvStatistics{Float64, ProbabilityWeights}}(countBMS)
        for i in indices(bmvs,1)
            bmvs[i] = BasicMvStatistics{Float64, ProbabilityWeights}(m)
        end
        
        for i in indices(data, 2)
            BAT.push!(bmvstats, data[:, i], w[i]);
            BAT.push!(bmvs[(i % countBMS) + 1], data[:, i], w[i]);
        end

        merbmvstats = merge(bmvs...)

        maxData =  [maximum(data[i, :]) for i in indices(data, 1)]
        minData =  [minimum(data[i, :]) for i in indices(data, 1)]
        
        for bs in [bmvstats, merbmvstats]
            @test bs.mean ≈  mean(data, Weights(w), 2)
            @test bs.cov ≈ cov(data, ProbabilityWeights(w), 2; corrected = true)
            @test bs.maximum ≈ maxData
            @test bs.minimum ≈ minData
        end

        mvstat = BasicMvStatistics{Float64, ProbabilityWeights}(m)
        res = append!(deepcopy(mvstat), data, 2)
        @test res.mean ≈ mean(data, 2)
        @test res.cov ≈ cov(data, 2)
        @test res.maximum ≈ maxData
        @test res.minimum ≈ minData
        
        res = append!(deepcopy(mvstat), data, w, 2)
        @test res.mean ≈ mean(data, weights(w), 2)        
        @test res.cov ≈ cov(data, ProbabilityWeights(w), 2; corrected = true)
        @test res.maximum ≈ maxData
        @test res.minimum ≈ minData        
    end
        
end
