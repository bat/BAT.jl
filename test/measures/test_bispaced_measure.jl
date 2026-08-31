# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random, Distributions, DensityInterface
using MeasureBase: getdof, massof
using BAT: BispacedMeasure, batmeasure, empiricalof, samplesof

@testset "bispaced_measure" begin
    context = BATContext(rng = Xoshiro(564001))
    prior = distprod(a = Normal(2.0, 1.0), b = Exponential(0.7))
    m = batmeasure(prior)
    n = 20
    xs = rand(Xoshiro(564002), m^n)
    smpls = DensitySampleVector(v = xs, logd = logdensityof(m).(xs))
    dsm = DensitySampleMeasure(smpls, dof = getdof(m))

    f = BAT.transform_function(NormalBased(), m)
    p = BispacedMeasure(f, dsm, context)
    @test samplesof(p) == smpls
    @test empiricalof(p) === dsm
    @test massof(p) == massof(dsm)

    up = unshaped(p, varshape(m))
    @test samplesof(up) == unshaped.(smpls)

    em = EvaluatedMeasure(m, transform_intent = NormalBased(), f_transform = f, empirical = p)
    @test empiricalof(em) === dsm
    @test samplesof(em) == smpls

    em_r = evalmeasure(em, RandResampling(nsamples = 10), context)
    rsmpls = samplesof(em_r)
    @test length(rsmpls) == 10
    @test all(in(smpls.v), rsmpls.v)
end
