# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Base.Test
using StatsBase

@testset "onlineuvstats" begin
    n = 10
    data1  = [(-1)^x*exp(x-n/2) for x in 1:n]
    data2  = 0.1*data1
    w = [2, 3]
    tdata2 = tuple(data2...)
    
    @testset "BAT.OnlineUvMean" begin
        @test typeof(@inferred BAT.OnlineUvMean()) <: BAT.OnlineUvMean{Float64}
        @test typeof(@inferred BAT.OnlineUvMean{Float32}()) <: BAT.OnlineUvMean{Float32}
        ouvm = OnlineUvMean()
        res = ouvm
        numMeans = 3
        means = Array{BAT.OnlineUvMean{Float64}}(numMeans)
        for i in indices(means, 1)
            means[i] = OnlineUvMean()
        end
        for i in indices(data1, 1)
            x = (i % numMeans) + 1
            means[x] = cat(means[x], [data1[i]], w[1]);
            res = cat(res, [data1[i]], w[1])
        end

        tocmp = mean(vcat(data1, data2), Weights(vcat([x*ones(n) for x in w]...)))
        res1 = cat(res, data2, w[2])
        @test res1[] ≈ tocmp
        res2 = cat(res, tdata2, w[2])
        @test res2[] ≈ tocmp

        merMean = merge!(means...)
        @test merMean[] ≈ res[]
    end
    @testset "BAT.OnlineUvVar" begin
        @test typeof(@inferred BAT.OnlineUvVar()) <: BAT.OnlineUvVar{Float64, ProbabilityWeights}
        @test typeof(@inferred BAT.OnlineUvVar{Float32, FrequencyWeights}()) <: BAT.OnlineUvVar{Float32, FrequencyWeights}

        ouvv = OnlineUvVar()
        res = ouvv
        numVars = 3
        vars = Array{BAT.OnlineUvVar{Float64, ProbabilityWeights}}(numVars)
        for i in indices(vars, 1)
            vars[i] = OnlineUvVar()
        end

        for i in indices(data1, 1)
            # x = (i % numVars) + 1
            # vars[x] = cat(vars[x], [data1[i]], w[1]);
            res = cat(res, [data1[i]], w[1])
            
        end
        
    end
end
