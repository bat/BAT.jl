# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test
using StableRNGs: StableRNG

import MGVI, ForwardDiff

@testset "MGVI" begin
    context = BATContext(rng = StableRNG(564008), ad = ForwardDiff)

    pstr = BAT.example_posterior()

    nsteps = 5
    nsmpls = 1000
    algorithm = MGVISampling(
        nsamples = nsmpls,
        schedule = FixedMGVISchedule(range(12, 100, length = nsteps)),
        store_unconverged = true,
    )
    em = evalmeasure(pstr, algorithm, context)
    smpls = BAT.samplesof(em)
    @test BAT.getess(BAT.empiricalof(em)) == nsmpls
    @test !first(smpls.info.converged) && last(smpls.info.converged)
    @test unique(smpls.info.stepno) == 1:nsteps + 1
end
