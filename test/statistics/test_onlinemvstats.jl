# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Statistics
using ArraysOfArrays, StatsBase

@testset "onlinestats" begin
    n = 10
    data1  = [(-1)^x*exp(x-n/2) for x in 1:n]
    data2 = reverse(data1, dims=1)

    w = [exp(x-n/2) for x in 1:n]

    data = VectorOfSimilarVectors(vcat(data1', data2'))
    m = innersize(data, 1)


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
        mvmeans = Array{BAT.OnlineMvMean{Float64}}(undef, countM)

        for i in axes(mvmeans, 1)
            mvmeans[i] = BAT.OnlineMvMean(m)
        end

        for i in axes(data, 1)
            push!(mvmean, data[i], w[i]);
            push!(mvmeans[(i % countM) + 1], data[i], w[i]);
        end

        @test mvmean ≈ mean(data, Weights(w))

        res = merge(mvmeans...)
        @test res ≈ mean(data, Weights(w))

        mvmean = BAT.OnlineMvMean(m)
        res = append!(deepcopy(mvmean), data)
        @test res ≈ mean(data)
        res = append!(deepcopy(mvmean), data, w)
        @test res ≈ mean(data, Weights(w))
    end

    @testset "BAT.OnlineMvCov" begin
        @test typeof(@inferred BAT.OnlineMvCov(n)) <: AbstractMatrix{Float64}
        @test typeof(@inferred BAT.OnlineMvCov{Float32, ProbabilityWeights}(n)) <: AbstractMatrix{Float32}

        for wKind in [ProbabilityWeights, FrequencyWeights, AnalyticWeights, Weights]
            mvcov = BAT.OnlineMvCov{Float64, wKind}(m)
            for i in eachindex(data)
                push!(mvcov, data[i], w[i]);
            end

            @test mvcov ≈ cov(data, wKind(w); corrected = (wKind != Weights))
        end

        countMvCovs = 3
        mvcovs = Array{BAT.OnlineMvCov{Float64, ProbabilityWeights}}(undef, countMvCovs)
        for i in axes(mvcovs,1)
            mvcovs[i] = BAT.OnlineMvCov(m)
        end
        mvcovc = BAT.OnlineMvCov(m)

        for i in eachindex(data)
            push!(mvcovs[(i % countMvCovs) + 1], data[i], w[i]);
            push!(mvcovc, data[i], w[i]);
        end

        res = merge(mvcovs...)
        @test res ≈ cov(data, ProbabilityWeights(w); corrected = true)
        @test res ≈ mvcovc
        @test Float64(res.sum_w) ≈ Float64(mvcovc.sum_w)
        @test Float64(res.sum_w2) ≈ Float64(mvcovc.sum_w2)
        @test res.n ≈ mvcovc.n
        @test res.Mean_X ≈ mvcovc.Mean_X
        @test res.New_Mean_X ≈ mvcovc.New_Mean_X
        @test push!(res, data[1], zero(Float64)) ≈ mvcovc

        mvcov = BAT.OnlineMvCov(m)
        res = append!(deepcopy(mvcov), data)
        @test res ≈ cov(data)
        res = append!(deepcopy(mvcov), data, w)
        @test res ≈ cov(data, ProbabilityWeights(w); corrected = true)
    end

    @testset "BAT.BasicMvStatistic" begin
        bmvstats = BasicMvStatistics{Float64, ProbabilityWeights}(m)

        countBMS = 3
        bmvs = Array{BAT.BasicMvStatistics{Float64, ProbabilityWeights}}(undef, countBMS)
        for i in axes(bmvs,1)
            bmvs[i] = BasicMvStatistics{Float64, ProbabilityWeights}(m)
        end

        for i in eachindex(data)
            BAT.push!(bmvstats, data[i], w[i]);
            #BAT.push!(bmvs[(i % countBMS) + 1], data[i], w[i]);
        end
    end
end
