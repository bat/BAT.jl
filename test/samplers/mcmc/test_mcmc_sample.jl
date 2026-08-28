# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributed, Random
using ArraysOfArrays, Distributions, PDMats, StatsBase
using DensityInterface
using Random123


@testset "mcmc_sample" begin
    mvec = [-0.3, 0.3]
    cmat = [1.0 1.5; 1.5 4.0]
    Σ = @inferred PDMat(cmat)
    mv_dist = MvNormal(mvec, Σ)
    likelihood = logfuncdensity(logdensityof(BAT.BATDistMeasure(mv_dist)))
    prior = product_distribution(Uniform.([-5, -8], [5, 8]))
    nchains = 4
    nwalkers = 1
    nsteps = 10^4

    samplingalg_MW = @inferred(TransformedMCMC(pretransform = DoNotTransform(), nchains = nchains, nwalkers = nwalkers, nsteps = nsteps))

    smplres = BAT.sample_and_verify(PosteriorMeasure(likelihood, prior), samplingalg_MW, mv_dist, BATContext(rng = Philox4x((564, 41))), max_retries = 0)
    samples = smplres.result
    @test smplres.verified
    @test smplres.n_retries == 0
    @test (nchains * nsteps - sum(samples.weight)) < 100

    samplingalg_PW = @inferred TransformedMCMC(proposal = RandomWalk(), pretransform = DoNotTransform(), nwalkers = nwalkers, nsteps = 10^5, sample_weighting = ARPWeighting())

    smplres_pw = BAT.sample_and_verify(mv_dist, samplingalg_PW, mv_dist, BATContext(rng = Philox4x((564, 42))), max_retries = 0)
    @test smplres_pw.verified
    @test smplres_pw.n_retries == 0

    gensamples(context::BATContext) = bat_sample(PosteriorMeasure(logfuncdensity(logdensityof(mv_dist)), prior), samplingalg_PW, context).result

    context = BATContext(rng = Philox4x((564, 43)))
    @test gensamples(context) != gensamples(context)
    @test gensamples(deepcopy(context)) == gensamples(deepcopy(context))

    @test_throws ArgumentError bat_sample(
        Normal(),
        TransformedMCMC(
            nchains = 1,
            init = MCMCChainPoolInit(nsteps_init = 1),
            burnin = MCMCMultiCycleBurnin(max_ncycles = 0)
        )
    )
    
    smplres_normal = BAT.sample_and_verify(Normal(), TransformedMCMC(pretransform = DoNotTransform(), nwalkers = nwalkers, nsteps = 10^4), Normal(), BATContext(rng = Philox4x((564, 44))), max_retries = 0)
    @test smplres_normal.verified
    @test smplres_normal.n_retries == 0
end
