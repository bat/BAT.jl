# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, StatsBase, ValueShapes
using EmpiricalDistributions: UvBinnedDist, MvBinnedDist

using BAT: MarginalDist, get_bin_centers, find_marginalmodes, asindex

@testset "MarginalDist" begin
    prior = NamedTupleDist(
        θ = MvNormal(zeros(2), [1.0 0.5; 0.5 2.0]),
        ϕ = Normal(1, 0.2)
    )

    context = BATContext()
    shaped_samples = bat_sample(prior, IIDSampling(nsamples = 10^4), context).result
    unshaped_samples = BAT.unshaped.(shaped_samples)

    @testset "asindex" begin
        @test asindex(prior, :θ) == [1, 2]
        @test asindex(prior, :ϕ) == 3

        @test asindex(shaped_samples, :ϕ) == 3
        @test asindex(shaped_samples, 3) == 3

        @test_throws ArgumentError asindex(unshaped_samples, :ϕ)
        @test asindex(unshaped_samples, 3) == 3
    end

    @testset "from samples" begin
        marg = MarginalDist(shaped_samples, :ϕ, bins = 40)
        @test marg isa MarginalDist
        @test marg.dist isa UvBinnedDist
        @test isapprox(mean(marg.dist), 1, atol = 0.05)

        @test MarginalDist(shaped_samples, 3, bins = 40).dist isa UvBinnedDist
        @test MarginalDist(unshaped_samples, 3, bins = 40).dist isa UvBinnedDist
        @test_throws ArgumentError MarginalDist(unshaped_samples, :ϕ)

        marg_2d = MarginalDist(unshaped_samples, (1, 2), bins = (30, 30))
        @test marg_2d isa MarginalDist
        binned_2d = marg_2d.dist isa BAT.ReshapedDist ? marg_2d.dist.dist : marg_2d.dist
        @test binned_2d isa MvBinnedDist
        @test isapprox(mean(binned_2d), [0, 0], atol = 0.1)
    end

    @testset "from distribution" begin
        marg = MarginalDist(prior, :ϕ, bins = 40, nsamples = 10^4)
        @test marg isa MarginalDist
        @test marg.dist isa UvBinnedDist
        @test isapprox(mean(marg.dist), 1, atol = 0.05)
    end

    @testset "bin centers and marginal modes" begin
        data1 = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 10]
        hist_1d = fit(Histogram, data1, nbins = 10, closed = :left)
        marg_1d = MarginalDist(UvBinnedDist(hist_1d))

        @test get_bin_centers(marg_1d) == [collect(1.5:1:10.5)]
        @test find_marginalmodes(marg_1d) == [[5.5], [9.5]]

        data2 = [10, 20, 30, 40, 50, 50, 60, 70, 80, 90, 90, 100]
        hist_2d = fit(Histogram, (data1, data2), nbins = 10, closed = :left)
        marg_2d = MarginalDist(MvBinnedDist(hist_2d))

        @test get_bin_centers(marg_2d) == [collect(1.5:1:10.5), collect(15:10:105)]
        @test find_marginalmodes(marg_2d) == [[5.5, 55.0], [9.5, 95.0]]
    end
end
