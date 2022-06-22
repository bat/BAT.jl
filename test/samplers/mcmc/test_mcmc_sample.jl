# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributed, Random
using ArraysOfArrays, Distributions, PDMats, StatsBase


@testset "mcmc_sample" begin
    mvec = [-0.3, 0.3]
    cmat = [1.0 1.5; 1.5 4.0]
    Σ = @inferred PDMat(cmat)
    mv_dist = MvNormal(mvec, Σ)
    likelihood = logfuncdensity(logdensityof(BAT.DistMeasure(mv_dist)))
    prior = product_distribution(Uniform.([-5, -8], [5, 8]))
    nchains = 4
    nsteps = 10^4

    algorithmMW = @inferred(MCMCSampling(mcalg = MetropolisHastings(), trafo = DoNotTransform(), nchains = nchains, nsteps = nsteps))

    smplres = BAT.sample_and_verify(PosteriorMeasure(likelihood, prior), algorithmMW, mv_dist)
    samples = smplres.result
    @test smplres.verified
    @test (nchains * nsteps - sum(samples.weight)) < 100


    algorithmPW = @inferred MCMCSampling(mcalg = MetropolisHastings(weighting = ARPWeighting()), trafo = DoNotTransform(), nsteps = 10^5)

    @test BAT.sample_and_verify(mv_dist, algorithmPW).verified

    gensamples(rng::AbstractRNG) = bat_sample(rng, PosteriorMeasure(logfuncdensity(logdensityof(mv_dist)), prior), algorithmPW).result

    rng = bat_rng()
    @test gensamples(rng) != gensamples(rng)
    @test gensamples(deepcopy(rng)) == gensamples(deepcopy(rng))
    
    @test BAT.sample_and_verify(Normal(), MCMCSampling(mcalg = MetropolisHastings(), trafo = DoNotTransform(), nsteps = 10^4)).verified
end
