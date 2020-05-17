using BAT
using StatsBase
using Distributions
using Test


@testset "BATHist from StatsBase.Hist" begin
    statsbase_hist_1d = fit(Histogram, randn(10_000), closed=:left)
    statsbase_hist_2d = fit(Histogram, (randn(10_000), randn(10_000)), closed=:left)

    bathist_1d = BATHistogram(statsbase_hist_1d)
    bathist_2d = BATHistogram(statsbase_hist_2d)

    @test isa(bathist_1d, BATHistogram)
    @test isa(bathist_2d, BATHistogram)
    @test isa(bathist_2d.h, StatsBase.Histogram)
end


likelihood = let D = 5
    params -> begin
        r = logpdf(MvNormal(zeros(D), ones(D)), params.θ)
        r2 = logpdf(Normal(5, 1.5), params.ϕ)
        LogDVal(r+r2)
    end
end;

prior = BAT.NamedTupleDist(
    θ = MvNormal(zeros(5), ones(5)),
    ϕ = Normal(1, 0.2)
)

posterior = PosteriorDensity(likelihood, prior);
algorithm = MetropolisHastings()
nsamples_per_chain = 10_000
nchains = 2

shaped_samples = bat_sample(posterior, (nsamples_per_chain, nchains), algorithm).result
unshaped_samples = BAT.unshaped.(shaped_samples)

#TODO: what about :θ ? it refers to a multidimensional parameter
@testset "asindex" begin
    @test BAT.asindex(prior, :θ) == 1
    @test BAT.asindex(prior, 1) == 1

    @test BAT.asindex(shaped_samples, :θ) == 1
    @test BAT.asindex(shaped_samples, 1) == 1

    @test_throws ArgumentError BAT.asindex(unshaped_samples, :θ) == 1
    @test BAT.asindex(unshaped_samples, 1) == 1
end



@testset "BATHist from samples" begin
    @test isa(BATHistogram(shaped_samples, :ϕ), BATHistogram)
    @test isa(BATHistogram(shaped_samples, 2), BATHistogram)

    @test_throws ArgumentError BATHistogram(unshaped_samples, :ϕ), BATHistogram
    @test isa(BATHistogram(unshaped_samples, 2), BATHistogram)

    @test isa(BATHistogram(shaped_samples, (:θ, :ϕ)), BATHistogram)
    @test isa(BATHistogram(shaped_samples, (1, 2)), BATHistogram)
end


@testset "BATHist from prior" begin
    @test isa(BATHistogram(prior, :ϕ), BATHistogram)
    @test isa(BATHistogram(prior, 2), BATHistogram)

    @test isa(BATHistogram(prior, (:θ, :ϕ)), BATHistogram)
    @test isa(BATHistogram(prior, (1, 2)), BATHistogram)
end
