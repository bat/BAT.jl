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
    likelihood = @inferred BAT.DistributionDensity(mv_dist)
    bounds = @inferred BAT.HyperRectBounds([-5, -8], [5, 8], BAT.reflective_bounds)
    prior = BAT.ConstDensity(LogDVal(0), bounds)
    nsamples = 10^5

    algorithmMW = @inferred(MCMCSampling(sampler = MetropolisHastings()))
    @test BAT.mcmc_compatible(algorithmMW.sampler, BAT.GenericProposalDist(mv_dist), BAT.NoVarBounds(2))

    samples = bat_sample(PosteriorDensity(likelihood, prior), nsamples, algorithmMW).result

    @test length(samples) == nsamples

    cov_samples = cov(flatview(samples.v), FrequencyWeights(samples.weight), 2; corrected=true)
    mean_samples = mean(flatview(samples.v), FrequencyWeights(samples.weight); dims = 2)

    @test isapprox(mean_samples, mvec; rtol = 0.15)
    @test isapprox(cov_samples, cmat; rtol = 0.15)

    algorithmPW = @inferred MCMCSampling(sampler = MetropolisHastings(weighting = ARPWeighting()))

    samples, chains = bat_sample(mv_dist, nsamples, algorithmPW)

    cov_samples = cov(flatview(samples.v), FrequencyWeights(samples.weight), 2; corrected=true)
    mean_samples = mean(flatview(samples.v), FrequencyWeights(samples.weight); dims = 2)

    @test isapprox(mean_samples, mvec; rtol = 0.15)
    @test isapprox(cov_samples, cmat; rtol = 0.15)

    gensamples(rng::AbstractRNG) = bat_sample(rng, PosteriorDensity(mv_dist, prior), nsamples, algorithmPW).result

    rng = bat_rng()
    @test gensamples(rng) != gensamples(rng)
    @test gensamples(deepcopy(rng)) == gensamples(deepcopy(rng))

    @test isapprox(var(bat_sample(Normal(), 10^4, MCMCSampling(sampler = MetropolisHastings())).result), [1], rtol = 10^-1)
end
