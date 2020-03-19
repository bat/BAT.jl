# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using ArraysOfArrays, Distributions, PDMats, StatsBase, IntervalSets

@testset "ahmc_sample" begin
    likelihood = let D = 5
        params -> begin
            r = logpdf(MvNormal(zeros(D), ones(D)), params.θ)
            LogDVal(r)
        end
    end;

    prior = BAT.NamedTupleDist(
        #θ = [-500..500, -500..500, -500..500, -500..500, -500..500]
        θ = [-5..5, -5..5, -5..5, -5..5, -5..5]
    )

    posterior = PosteriorDensity(likelihood, prior);

    algorithm = AHMC()
    nsamples_per_chain = 100_000
    nchains = 2
    samples, chains = bat_sample(posterior, (nsamples_per_chain, nchains), algorithm)

    #@test length(samples) == nchains * nsamples_per_chain

    #cov_samples = cov(flatview(samples.v), FrequencyWeights(samples.weight), 2; corrected=true)
    #mean_samples = mean(flatview(samples.v), FrequencyWeights(samples.weight); dims = 2)

    #@test isapprox(mean_samples, mvec; rtol = 0.15)
    #@test isapprox(cov_samples, cmat; rtol = 0.15)

end


@testset "ahmc_sample_options" begin
    likelihood = let D = 5
        params -> begin
            r = logpdf(MvNormal(zeros(D), ones(D)), params.θ)
            LogDVal(r)
        end
    end;

    prior = BAT.NamedTupleDist(
        #θ = [-5..5, -5..5, -5..5, -5..5, -5..5]
        θ = MvNormal(zeros(5), ones(5))
    )

    posterior = PosteriorDensity(likelihood, prior);

    algorithm = AHMC()
    nsamples_per_chain = 100_000
    nchains = 1

    for metric in [DiagEuclideanMetric()]#, UnitEuclideanMetric(), DenseEuclideanMetric()]
        for integrator in [Leapfrog()]#, JitteredLeapfrog(), TemperedLeapfrog()]
            @inferred bat_sample(posterior, (nsamples_per_chain, nchains), algorithm; metric=metric, integrator=integrator)
        end
    end


end
