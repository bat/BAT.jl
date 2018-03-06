# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT, BAT.Logging
using Compat.Test
using Distributions, PDMats, StatsBase

@testset "mcmc_rand" begin
    @testset "rand" begin
        set_log_level!(BAT, LOG_WARNING)

        mvec = [-0.3, 0.3]
        cmat = [1.0 1.5; 1.5 4.0]
        Σ = @inferred PDMat(cmat)
        density = @inferred MvDistDensity(MvNormal(mvec, Σ))
        bounds = @inferred HyperRectBounds([-5, -8], [5, 8], reflective_bounds)
        nsamples_per_chain = 2000
        nchains = 4

        algorithmMW = @inferred MetropolisHastings(MHMultiplicityWeights())
        samples, sampleids, stats = @inferred rand(
            MCMCSpec(algorithmMW, density, bounds),
            nsamples_per_chain,
            nchains,
            max_time = Inf,
            granularity = 1
        )

        @test length(samples) == length(sampleids)
        @test length(samples) == nchains * nsamples_per_chain
        @test samples.params[:, findmax(samples.log_value)[2]] == stats.mode

        cov_samples = cov(samples.params, FrequencyWeights(samples.weight), 2; corrected=true)
        mean_samples = mean(Array(samples.params), FrequencyWeights(samples.weight), 2)

        @test isapprox(mean_samples, mvec; atol = 0.2)
        @test isapprox(cov_samples, cmat; atol = 0.5)

        algorithmPW = @inferred MetropolisHastings(MHAccRejProbWeights())
        samples, sampleids, stats = @inferred rand(
            MCMCSpec(algorithmPW, density, bounds),
            nsamples_per_chain,
            nchains,
            max_time = Inf,
            granularity = 1
        )

        @test length(samples) == length(sampleids)
        @test samples.params[:, findmax(samples.log_value)[2]] == stats.mode

        cov_samples = cov(samples.params, FrequencyWeights(samples.weight), 2; corrected=true)
        mean_samples = mean(Array(samples.params), FrequencyWeights(samples.weight), 2)

        @test isapprox(mean_samples, mvec; atol = 0.2)
        @test isapprox(cov_samples, cmat; atol = 0.5)

        algorithmFW = @inferred MetropolisHastings(MHPosteriorFractionWeights())
        
        samples, sampleids, stats = @inferred rand(
            MCMCSpec(algorithmFW, density, bounds),
            nsamples_per_chain,
            nchains,
            max_time = Inf,
            granularity = 1
        )

        @test length(samples) == length(sampleids)
        @test samples.params[:, findmax(samples.log_value)[2]] == stats.mode

        cov_samples = cov(samples.params, FrequencyWeights(samples.weight), 2; corrected=true)
        mean_samples = mean(Array(samples.params), FrequencyWeights(samples.weight), 2)

        @test isapprox(mean_samples, mvec; atol = 0.2)
        @test isapprox(cov_samples, cmat; atol = 0.5)
        
    end
end
