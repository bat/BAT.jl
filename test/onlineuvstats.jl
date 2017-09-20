# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Base.Test
using StatsBase

@testset "onlineuvstats" begin
    n = 10
    data  = [(-1)^x*exp(x-n/2) for x in 1:n]
    w = abs.(data)
    tdata = tuple(data...)
    
    @testset "BAT.OnlineUvMean" begin
        @test typeof(@inferred BAT.OnlineUvMean()) <: BAT.OnlineUvMean{Float64}
        @test typeof(@inferred BAT.OnlineUvMean{Float32}()) <: BAT.OnlineUvMean{Float32}
        ouvm = OnlineUvMean()

        res = cat(ouvm, data, w)
        @test res[] ≈ mean(data, Weights(w))
        res = cat(res, tdata, w)
        @test res[] ≈ mean(data, Weights(w))
        
        numMeans = 3
        means = Array{BAT.OnlineUvMean{Float64}}(numMeans)
        for i in indices(means, 1)
            means[i] = OnlineUvMean()
        end
        
        for i in indices(data, 1)
            x = (i % numMeans) + 1
            means[x] = cat(means[x], data[i], w[i]);
        end

        @test res[] ≈ merge!(means...)[]

    end
    @testset "BAT.OnlineUvVar" begin
        @test typeof(@inferred BAT.OnlineUvVar()) <: BAT.OnlineUvVar{Float64, ProbabilityWeights}
        @test typeof(@inferred BAT.OnlineUvVar{Float32, FrequencyWeights}()) <: BAT.OnlineUvVar{Float32, FrequencyWeights}

        for wKind in [ProbabilityWeights, FrequencyWeights, AnalyticWeights, Weights]
            res = BAT.OnlineUvVar{Float64, wKind}()
            res = cat(res, data, w)
            @test res[] ≈ var(data, wKind(w); corrected=(wKind != Weights))
        end

        numVars = 3
        vars = Array{BAT.OnlineUvVar{Float64, ProbabilityWeights}}(numVars)
        for i in indices(vars, 1)
            vars[i] = OnlineUvVar()
        end

        for i in indices(data, 1)
            x = (i % numVars) + 1
            vars[x] = cat(vars[x], data[i], w[i]);
        end

        @test merge(vars...)[] ≈ var(data, ProbabilityWeights(w); corrected=true)
    end
end
