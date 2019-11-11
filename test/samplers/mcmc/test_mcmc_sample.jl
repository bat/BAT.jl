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
    density = @inferred DistributionDensity(mv_dist)
    bounds = @inferred HyperRectBounds([-5, -8], [5, 8], reflective_bounds)
    nsamples_per_chain = 50000
    nchains = 4

    # algorithmMW = @inferred MetropolisHastings() TODO: put back the @inferred
    algorithmMW = MetropolisHastings()
    @test BAT.mcmc_compatible(algorithmMW, GenericProposalDist(mv_dist), NoParamBounds(2))

    samples, stats, chains = bat_sample(
        PosteriorDensity(density, bounds), (nsamples_per_chain, nchains), algorithmMW
    )

    # ToDo: Should be able to make this exact, for MH sampler:
    @test length(samples) == nchains * nsamples_per_chain

    @test begin
        stats = @inferred bat_stats(samples)
        stats2 = @inferred bat_stats(MCMCBasicStats(samples))
        stats.mode == stats2.mode && stats.mean ≈ stats2.mean && stats.cov ≈ stats2.cov
    end

    cov_samples = cov(flatview(samples.params), FrequencyWeights(samples.weight), 2; corrected=true)
    mean_samples = mean(flatview(samples.params), FrequencyWeights(samples.weight); dims = 2)

    @test isapprox(mean_samples, mvec; rtol = 0.15)
    @test isapprox(cov_samples, cmat; rtol = 0.15)

    algorithmPW = @inferred MetropolisHastings(ARPWeights())

    samples, stats, chains = bat_sample(
        PosteriorDensity(mv_dist, bounds), (nsamples_per_chain, nchains), algorithmPW
    )

    @test begin
        stats = @inferred bat_stats(samples)
        stats2 = @inferred bat_stats(MCMCBasicStats(samples))
        stats.mode == stats2.mode && stats.mean ≈ stats2.mean && stats.cov ≈ stats2.cov
    end

    @test samples.params[findmax(samples.logdensity)[2]] == stats.mode

    cov_samples = cov(flatview(samples.params), FrequencyWeights(samples.weight), 2; corrected=true)
    mean_samples = mean(flatview(samples.params), FrequencyWeights(samples.weight); dims = 2)

    @test isapprox(mean_samples, mvec; rtol = 0.15)
    @test isapprox(cov_samples, cmat; rtol = 0.15)
end
