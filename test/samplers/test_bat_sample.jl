# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random, Distributions, StatsBase


@testset "bat_sample" begin
    context = BATContext()

    @testset "IIDSampling" begin
        dist = MvNormal([0.4, 0.6], [2.0 1.2; 1.2 3.0])

        @test length(@inferred(bat_sample(dist, IIDSampling(nsamples = 10^3), context)).result) == 10^3

        @test @inferred(bat_sample(dist, context)).result isa DensitySampleVector
        @test bat_sample(dist, BAT.IIDSampling()).result isa DensitySampleVector
        @test @inferred(bat_sample(dist, BAT.IIDSampling(), context)).result isa DensitySampleVector

        samples = @inferred(bat_sample(dist, IIDSampling(nsamples = 10^5), context)).result
        @test isapprox(mean(samples.v), [0.4, 0.6]; rtol = 0.05)
        @test isapprox(cov(samples.v), [2.0 1.2; 1.2 3.0]; rtol = 0.05)
        @test all(isequal(1), samples.weight)

        dist_bmode = @inferred(bat_findmode(dist, context)).result
        @test @inferred(length(dist_bmode)) == 2

        dist_sample_vector_bmode = @inferred(bat_findmode(samples, context)).result
        @test @inferred(length(dist_sample_vector_bmode)) == 2

        isapprox(var(bat_sample(Normal(), BAT.IIDSampling(nsamples = 10^3), context).result), 1, rtol = 10^-1)
    end

    @testset "RandResampling" begin
        dist = Normal()
        result = @inferred(bat_sample(dist, IIDSampling(nsamples = 2), context)).result #Draw to samples from Normal dist

        @test @inferred(bat_sample(result, context)).result isa DensitySampleVector#Check data types 
        @test bat_sample(result, RandResampling(nsamples = 100)).result isa DensitySampleVector
        @test @inferred(bat_sample(result, BAT.RandResampling(), context)).result isa DensitySampleVector

        samples_rdm = @inferred(bat_sample(result, RandResampling(nsamples = 10^5), context)).result #Sample 100 times from the 2-sample space
        @test length(@inferred(bat_sample(result, RandResampling(nsamples = 100), context)).result) == 100#Check shape is ok
        @test sort(unique(samples_rdm.v)) == sort(result.v)#check it only samples from the 2-sample space
        @test isapprox(mean(samples_rdm), mean(result), rtol = 10^-1)#means should be the same in both datasets
    end

    @testset "OrderedResampling" begin #Creates new testset for OrderedResampling
        dist = MvNormal([0.4, 0.6], [2.0 1.2; 1.2 3.0])
        result = @inferred(bat_sample(dist, IIDSampling(nsamples = 10^5), context)).result

        @test isapprox(mean([length(@inferred(bat_sample(result, OrderedResampling(nsamples = 10), context)).result.v) for i in 1:10^3]), 10, rtol = 10^-1)

        @test @inferred(bat_sample(result, context)).result isa DensitySampleVector#Check that types are consistent
        @test @inferred(bat_sample(result, BAT.OrderedResampling(), context)).result isa DensitySampleVector
        @test bat_sample(result, BAT.OrderedResampling()).result isa DensitySampleVector

        resamples = @inferred(bat_sample(result, OrderedResampling(nsamples = length(result)), context)).result
        @test result == resamples
    end
end
