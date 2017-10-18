# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT, BAT.Logging
using Base.Test

using Distributions, PDMats, StatsBase
using Base.Test

@testset "mcmc_rand" begin
    @testset "rand" begin
        set_log_level!(BAT, LOG_WARNING)

        mvec = [-0.3, 0.3]
        cmat = [1.0 1.5; 1.5 4.0]
        Σ = @inferred PDMat(cmat)
        tdensity = @inferred MvDistTargetDensity(MvNormal(mvec, Σ))
        algorithm = @inferred MetropolisHastings()
        bounds = @inferred HyperRectBounds([-5, -8], [5, 8], reflective_bounds)
        nsamples_per_chain = 2000
        nchains = 4

        samples = @inferred rand(
            MCMCSpec(algorithm, tdensity, bounds),
            nsamples_per_chain,
            nchains,
            max_time = Inf,
            granularity = 1
        )

        @test length(samples) == nchains * nsamples_per_chain

        cov_samples = cov(samples.params, FrequencyWeights(samples.weight), 2; corrected=true)
        mean_samples = mean(Array(samples.params), FrequencyWeights(samples.weight), 2)

        @test isapprox(mean_samples, mvec; atol = 0.1)
        @test isapprox(cov_samples, cmat; atol = 0.3)
    end
end
