# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

import MGVI, ForwardDiff

@testset "MGVI" begin
    context = BATContext(ad = ForwardDiff)

    pstr = BAT.example_posterior()

    @test (@inferred MGVISampling()) isa MGVISampling

    nsteps = 5
    nsmpls = 1000
    algorithm = MGVISampling(
        nsamples = nsmpls,
        schedule = FixedMGVISchedule(range(12, 100, length = nsteps)),
        store_unconverged = true,
    )
    r = bat_sample(pstr, algorithm, context)
    @test r.result isa DensitySampleVector
    @test r.evaluated isa EvaluatedMeasure
    @test first(r.result.info.converged) == false
    @test last(r.result.info.converged) == true
    @test unique(r.result.info.stepno) == 1:nsteps+1
    @test r.ess == nsmpls
    @test r.info.mnlp isa Real

    # ToDo: Test quality of samples
end
