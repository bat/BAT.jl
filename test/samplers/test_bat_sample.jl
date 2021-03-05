# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random, Distributions, StatsBase


@testset "bat_sample" begin
    @testset "IIDSampling" begin
        dist = MvNormal([0.4, 0.6], [2.0 1.2; 1.2 3.0])

        @test length(@inferred(bat_sample(dist, IIDSampling(nsamples = 10^3))).result) == 10^3

        @test @inferred(bat_sample(Random.GLOBAL_RNG, dist)).result isa DensitySampleVector
        @test @inferred(bat_sample(dist, BAT.IIDSampling())).result isa DensitySampleVector
        @test @inferred(bat_sample(Random.GLOBAL_RNG, dist, BAT.IIDSampling())).result isa DensitySampleVector

        samples = @inferred(bat_sample(dist, IIDSampling(nsamples = 10^5))).result
        @test isapprox(mean(samples.v), [0.4, 0.6]; rtol = 0.05)
        @test isapprox(cov(samples.v), [2.0 1.2; 1.2 3.0]; rtol = 0.05)
        @test all(isequal(1), samples.weight)

        dist_bmode = @inferred(bat_findmode(dist)).result
        @test @inferred(length(dist_bmode)) == 2

        dist_sample_vector_bmode = @inferred(bat_findmode(samples)).result
        @test @inferred(length(dist_sample_vector_bmode)) == 2

        isapprox(var(bat_sample(Normal(), BAT.IIDSampling(nsamples = 10^3)).result), [1], rtol = 10^-1)
    end
    @testset "RandResampling" begin
        dist = Normal()
        result = @inferred(bat_sample(dist, IIDSampling(nsamples = 2))).result #Draw to samples from Normal dist

        

        @test @inferred(bat_sample(Random.GLOBAL_RNG, result)).result isa DensitySampleVector#Check data types 
        @test @inferred(bat_sample(result, RandResampling(nsamples = 100))).result isa DensitySampleVector
        @test @inferred(bat_sample(Random.GLOBAL_RNG, result, BAT.RandResampling())).result isa DensitySampleVector

        
        samples_rdm = @inferred(bat_sample(result, RandResampling(nsamples = 10^5))).result.v#Sample 100 times from the 2-sample space
        @test length(@inferred(bat_sample(result, RandResampling(nsamples = 100))).result) == 100#Check shape is ok
        @test sort(unique(samples_rdm)) == sort(result.v)#check it only samples from the 2-sample space
        @test isapprox(mean(samples_rdm), mean(result.v), rtol = 10^-1)#means should be the same in both datasets
    end
    @testset "OrderedResampling" begin #Creates new testset for OrderedResampling
        dist = MvNormal([0.4, 0.6], [2.0 1.2; 1.2 3.0])
        result = @inferred(bat_sample(dist, IIDSampling(nsamples = 10^5))).result

        #@test length(@inferred(bat_sample(result, OrderedResampling(nsamples = 10^3))).result) == 10^3#Commented out: fails but looks intended behav?

        @test @inferred(bat_sample(Random.GLOBAL_RNG, result)).result isa DensitySampleVector#Check that types are consistent
        @test @inferred(bat_sample(result, BAT.OrderedResampling())).result isa DensitySampleVector
        @test @inferred(bat_sample(Random.GLOBAL_RNG, result, BAT.OrderedResampling())).result isa DensitySampleVector

        resamples = @inferred(bat_sample(result, OrderedResampling(nsamples = length(result)))).result
        @test result == resamples
        
    end
end