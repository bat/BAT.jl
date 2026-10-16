# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, StatsBase, ValueShapes, LinearAlgebra, ArraysOfArrays
using EmpiricalDistributions: UvBinnedDist, MvBinnedDist

using BAT: MarginalDist, get_bin_centers, find_marginalmodes, asindex, bat_marginalize

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

        low_weight_samples = DensitySampleVector(
            [[0.0], [1.0e15], [1.0], [2.0]], zeros(4);
            weight = [1.0, 1.0e-6, 5.0e-6, 1.0],
        )
        filtered = MarginalDist(low_weight_samples, 1; bins = 4, filter = true)
        @test maximum(abs, only(get_bin_centers(filtered))) < 10

        # Non-1-based flat storage selects by position among the free parameters:
        offset_data = VectorOfSimilarVectors(view([0.0 0.0; 10.0 30.0; 20.0 40.0], Base.IdentityUnitRange(2:3), :))
        offset_samples = DensitySampleVector(offset_data, zeros(2))
        shaped_offset_samples = NamedTupleShape(a = ScalarShape{Real}(), b = ScalarShape{Real}()).(offset_samples)
        @test flatview(bat_marginalize(offset_samples, 2).result.v) == [20.0 40.0]
        @test bat_marginalize(shaped_offset_samples, :b).result.v.b == [20.0, 40.0]
    end

    @testset "from distribution" begin
        marg = MarginalDist(prior, :ϕ, bins = 40, nsamples = 10^4)
        @test marg isa MarginalDist
        @test marg.dist isa UvBinnedDist
        @test isapprox(mean(marg.dist), 1, atol = 0.05)
    end

    @testset "nested named tuples" begin
        nested_prior = distprod(
            x = (a = Uniform(0, 1), b = (c = Normal(1, 0.2), d = MvNormal(zeros(2), I(2)), k = 7.0)),
            e = Uniform(0, 1)
        )
        nested_samples = bat_sample(nested_prior, IIDSampling(nsamples = 10^3), context).result

        @test asindex(nested_samples, :(x.b.c)) == 2
        @test asindex(nested_samples, :(x.b.d[2])) == 4
        @test asindex(nested_samples, :x) == [1, 2, 3, 4]
        @test_throws ArgumentError asindex(nested_samples, :(x[2]))
        @test_throws ArgumentError asindex(nested_samples, :(x.b.k))
        @test_throws ArgumentError asindex(nested_samples, :(x.nope))
        @test_throws ArgumentError MarginalDist(nested_samples, :(x(e)))

        marg = MarginalDist(nested_samples, :(x.b.c), bins = 40)
        @test marg.dist isa UvBinnedDist
        @test isapprox(mean(marg.dist), 1, atol = 0.05)
        @test MarginalDist(nested_samples, 2, bins = 40).dist isa UvBinnedDist

        marg_2d = MarginalDist(nested_samples, (:(x.b.c), :e), bins = (20, 20))
        @test keys(varshape(marg_2d.dist)) == (Symbol("x.b.c"), :e)
        @test keys(varshape(MarginalDist(nested_samples, :([x.b.c, e]), bins = (20, 20)).dist)) == (Symbol("x.b.c"), :e)
        @test varshape(MarginalDist(nested_samples, :x, bins = 5).dist) == NamedTupleShape(x = varshape(nested_prior).x.shape)
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
