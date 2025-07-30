# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, MeasureBase, ValueShapes
using HeterogeneousComputing
using ArraysOfArrays

@testset "bat_pwr_measure" begin
    make_ctx() = GenContext(BAT.example_stable_rng())

    dist = MvNormal([0.4, 0.6], [2.0 1.2; 1.2 3.0])

    mu_d = batmeasure(dist)
    nu_d = mu_d^5

    @test rand(make_ctx(), nu_d) isa ArrayOfSimilarArrays
    @test bat_sample(nu_d, IIDSampling(nsamples = 20)).result isa DensitySampleVector


    f_dt = BAT.DistributionTransform(Uniform, dist)
    f_hasladj = Base.Broadcast.BroadcastFunction(exp)
    f_plain = Base.Broadcast.BroadcastFunction(abs)

    mu_f_dt = pushfwd(f_dt, mu_d)
    mu_f_hasladj = pushfwd(f_hasladj, mu_d)
    mu_f_plain = pushfwd(f_plain, mu_d)

    nu_f_dt = mu_f_dt^5
    nu_f_hasladj = mu_f_hasladj^5
    nu_f_plain = mu_f_plain^5

    @test rand(make_ctx(), nu_f_dt) isa ArrayOfSimilarArrays
    @test rand(make_ctx(), nu_f_hasladj) isa ArrayOfSimilarArrays
    @test rand(make_ctx(), nu_f_plain) isa ArrayOfSimilarArrays
    @test bat_sample(nu_f_dt, IIDSampling(nsamples = 20)).result isa DensitySampleVector
    @test bat_sample(nu_f_hasladj, IIDSampling(nsamples = 20)).result isa DensitySampleVector


    mu_w = WeightedMeasure(0.2, mu_d)
    nu_w = mu_w^5

    @test rand(make_ctx(), nu_w) isa ArrayOfSimilarArrays
    @test bat_sample(nu_w, IIDSampling(nsamples = 20)).result isa DensitySampleVector
end
