# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT, BAT.Logging
using Test

using Distributed, Random
using ArraysOfArrays, Distributions, PDMats, StatsBase


@testset "mcmc_rand" begin
    @testset "rand" begin
        set_log_level!(BAT, LOG_WARNING)

        mvec = [-0.3, 0.3]
        cmat = [1.0 1.5; 1.5 4.0]
        Σ = @inferred PDMat(cmat)
        mv_dist = MvNormal(mvec, Σ)
        density = @inferred MvDistDensity(mv_dist)
        bounds = @inferred HyperRectBounds([-5, -8], [5, 8], reflective_bounds)
        nsamples_per_chain = 20000
        nchains = 4

        # algorithmMW = @inferred MetropolisHastings() TODO: put back the @inferred
        algorithmMW = MetropolisHastings()
        @test BAT.mcmc_compatible(algorithmMW, GenericProposalDist(mv_dist), NoParamBounds(2))
#        samples, sampleids, stats = @inferred rand( TODO: put back the @inferred
        samples, sampleids, stats = rand(
            MCMCSpec(algorithmMW, BayesianModel(density, bounds)),
            nsamples_per_chain,
            nchains,
            max_time = Inf,
            granularity = 1
        )

        @test length(samples) == length(sampleids)
        @test length(samples) == nchains * nsamples_per_chain
        @test samples.params[findmax(samples.log_posterior)[2]] == stats.mode

        cov_samples = cov(flatview(samples.params), FrequencyWeights(samples.weight), 2; corrected=true)
        mean_samples = mean(flatview(samples.params), FrequencyWeights(samples.weight); dims = 2)

        @test isapprox(mean_samples, mvec; rtol = 0.1)
        @test isapprox(cov_samples, cmat; rtol = 0.1)

        algorithmPW = @inferred MetropolisHastings(MHAccRejProbWeights())
        # samples, sampleids, stats = @inferred rand(
        samples, sampleids, stats = rand(
            MCMCSpec(algorithmPW, BayesianModel(mv_dist, bounds)),
            nsamples_per_chain,
            nchains,
            max_time = Inf,
            granularity = 1
        )

        @test length(samples) == length(sampleids)
        @test samples.params[findmax(samples.log_posterior)[2]] == stats.mode

        cov_samples = cov(flatview(samples.params), FrequencyWeights(samples.weight), 2; corrected=true)
        mean_samples = mean(flatview(samples.params), FrequencyWeights(samples.weight); dims = 2)

        @test isapprox(mean_samples, mvec; rtol = 0.1)
        @test isapprox(cov_samples, cmat; rtol = 0.1)
    end
end
