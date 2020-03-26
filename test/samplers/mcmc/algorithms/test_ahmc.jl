# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using ArraysOfArrays, Distributions, PDMats, StatsBase, IntervalSets, LinearAlgebra

@testset "AHMC: sample" begin

    dims = 5
    likelihood = let D = dims
        params -> begin
            r = logpdf(MvNormal(zeros(D), ones(D)), params.θ)
            LogDVal(r)
        end
    end;

    prior = BAT.NamedTupleDist(
        θ = MvNormal(zeros(5), ones(5))
    )

    posterior = PosteriorDensity(likelihood, prior);

    algorithm = AHMC()
    nsamples_per_chain = 10_000
    nchains = 2
    samples, chains = bat_sample(posterior, (nsamples_per_chain, nchains), algorithm)

    # number of samples smaller then requested because of weighting,
    @test isapprox(length(samples), nchains * nsamples_per_chain; rtol=0.2)

    cov_samples = cov(BAT.unshaped.(samples.v), FrequencyWeights(samples.weight))
    mean_samples = mean(BAT.unshaped.(samples.v), FrequencyWeights(samples.weight))

    @test isapprox(mean_samples, zeros(dims); atol = 0.05)
    @test isapprox(cov_samples, 0.5*Matrix{Int}(I, dims, dims); atol = 0.05)

end



# running AHMC with uniform prior will give the warning:
# "Warning: The current proposal will be rejected due to numerical error(s).
#  isfinite.((θ, r, ℓπ, ℓκ)) = (true, true, false, true)"

# @testset "AHMC: sample with uniform prior" begin
#
#     dims = 5
#     likelihood = let D = dims
#         params -> begin
#             r = logpdf(MvNormal(zeros(D), ones(D)), params.θ)
#             LogDVal(r)
#         end
#     end;
#
#     prior = BAT.NamedTupleDist(
#         θ = [-5..5, -5..5, -5..5, -5..5, -5..5]
#     )
#
#     posterior = PosteriorDensity(likelihood, prior);
#
#     algorithm = AHMC()
#     nsamples_per_chain = 10_000
#     nchains = 2
#     samples, chains = bat_sample(posterior, (nsamples_per_chain, nchains), algorithm)
#
#     @test isapprox(length(samples), nchains * nsamples_per_chain; rtol=0.2)
#
#     cov_samples = cov(BAT.unshaped.(samples.v), FrequencyWeights(samples.weight))
#     mean_samples = mean(BAT.unshaped.(samples.v), FrequencyWeights(samples.weight))
#
#     @test isapprox(mean_samples, zeros(dims); atol = 0.5)
#     @test isapprox(cov_samples, Matrix{Int}(I, dims, dims); atol = 0.05)
#
# end
