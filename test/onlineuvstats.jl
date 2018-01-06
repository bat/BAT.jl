# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Compat.Test
using StatsBase
using DoubleDouble

@testset "onlineuvstats" begin
    n = 10
    data  = [(-1)^x*exp(x-n/2) for x in 1:n]
    w = abs.(data)

    T = Float64
    wK = ProbabilityWeights
    mdata = mean(data, wK(w))
    vdata = var(data, wK(w); corrected=true)
    maxdata = maximum(data)
    mindata = minimum(data)

    @testset "BAT.OnlineUvMean" begin
        @test typeof(@inferred BAT.OnlineUvMean()) <: BAT.OnlineUvMean{Float64}
        @test typeof(@inferred BAT.OnlineUvMean{Float32}()) <: BAT.OnlineUvMean{Float32}
        ouvm = OnlineUvMean()

        res = append!(ouvm, data, w)
        @test res[] ≈ mdata

        numMeans = 3
        means = Array{BAT.OnlineUvMean{T}}(numMeans)
        for i in indices(means, 1)
            means[i] = OnlineUvMean()
        end

        for i in indices(data, 1)
            x = (i % numMeans) + 1
            means[x] = push!(means[x], data[i], w[i]);
        end

        @test merge!(means...)[] ≈ mdata

    end

    @testset "BAT.OnlineUvVar" begin
        @test typeof(@inferred BAT.OnlineUvVar()) <: BAT.OnlineUvVar{Float64, ProbabilityWeights}
        @test typeof(@inferred BAT.OnlineUvVar{Float32, FrequencyWeights}()) <: BAT.OnlineUvVar{Float32, FrequencyWeights}

        for wKind in [ProbabilityWeights, FrequencyWeights, AnalyticWeights, Weights]
            res = @inferred BAT.OnlineUvVar{Float64, wKind}()
            res = append!(res, data, w)
            @test res[] ≈ var(data, wKind(w); corrected=(wKind != Weights))
        end

        numVars = 3
        vars = Array{BAT.OnlineUvVar{T, wK}}(numVars)
        for i in indices(vars, 1)
            vars[i] = OnlineUvVar()
        end

        for i in indices(data, 1)
            x = (i % numVars) + 1
            vars[x] = push!(vars[x], data[i], w[i]);
        end

        merge!(vars...)
        @test vars[1][] ≈ var(data, wK(w); corrected=true)
        @test push!(vars[1], data[1], zero(Float64))[] ≈ var(data, wK(w); corrected=true)
    end

    @testset "BAT.BasicUvStatistics" begin
        @test typeof(@inferred BAT.BasicUvStatistics{Float32, FrequencyWeights}()) <: BAT.BasicUvStatistics{Float32, FrequencyWeights}

        res = BAT.BasicUvStatistics{T, wK}()
        res = append!(res, data, w)

        @test res.mean[] ≈ mdata
        @test res.var[] ≈ vdata # 100
        @test res.maximum ≈ maxdata
        @test res.minimum ≈ mindata

        numStats = 3
        stats = Array{BAT.BasicUvStatistics{T, wK}}(numStats)
        for i in indices(stats, 1)
            stats[i] = BasicUvStatistics{T, wK}()
        end

        for i in indices(data, 1)
            x = (i % numStats) + 1
            stats[x] = push!(stats[x], data[i], w[i]);
        end

        res = merge(stats...)
        @test res.mean[] ≈ mdata
        @test res.var[] ≈ vdata
        @test res.maximum ≈ maxdata
        @test res.minimum ≈ mindata

        res = BAT.BasicUvStatistics{T, wK}()
        res = append!(res, data)
        @test res.mean[] ≈ mean(data)
        @test res.var[] ≈ cov(data)
    end

end
