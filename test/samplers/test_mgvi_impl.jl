# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test
using StableRNGs: StableRNG

import MGVI, ForwardDiff

@testset "MGVI" begin
    context = BATContext(rng = StableRNG(564008), ad = ForwardDiff)

    pstr = BAT.example_posterior()

    @test (@inferred MGVISampling()) isa MGVISampling

    nsteps = 5
    nsmpls = 1000
    algorithm = MGVISampling(
        nsamples = nsmpls,
        schedule = FixedMGVISchedule(range(12, 100, length = nsteps)),
        store_unconverged = true,
    )
    em = evalmeasure(pstr, algorithm, context)
    smpls = BAT.samplesof(em)
    @test em isa EvaluatedMeasure
    @test smpls isa DensitySampleVector
    @test first(smpls.info.converged) == false
    @test last(smpls.info.converged) == true
    @test unique(smpls.info.stepno) == 1:nsteps+1
    @test BAT.getess(BAT.empiricalof(em)) == nsmpls
    @test BAT.evalinfo(em).result.mnlp isa Real

    # ToDo: Test quality of samples
end
