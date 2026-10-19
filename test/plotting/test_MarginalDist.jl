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

        low_weight_samples = DensitySampleVector(
            [[0.0], [1.0e15], [1.0], [2.0]], zeros(4);
            weight = [1.0, 1.0e-6, 5.0e-6, 1.0],
        )
        filtered = MarginalDist(low_weight_samples, 1; bins = 4, filter = true)
        @test maximum(abs, only(get_bin_centers(filtered))) < 10
    end

    @testset "from distribution" begin
        marg = MarginalDist(prior, :ϕ, bins = 40, nsamples = 10^4)
        @test marg isa MarginalDist
        @test marg.dist isa UvBinnedDist
        @test isapprox(mean(marg.dist), 1, atol = 0.05)
    end

    @testset "Representable histogram edges" begin
        for lo in (0.0, 1e-300, 1.0, 1e8, 1e16, 1e100, 1e300),
                nsteps in (1, 2, 10), closed in (:left, :right)
            hi = lo
            for _ in 1:nsteps
                hi = nextfloat(hi)
            end
            values = [lo, hi]
            edges = BAT._get_edges(values, 100, closed)
            hist = fit(Histogram, values, FrequencyWeights([2.0, 3.0]), edges; closed)
            @test all(diff(collect(edges)) .> 0)
            @test all(w -> isapprox(w, edges[2] - edges[1]; atol=0), diff(collect(edges)))
            @test length(edges) <= 101
            @test first(edges) < lo
            @test last(edges) > hi
            @test sum(hist.weights) == 5
            # Subnormal widths need not have a representable PDF height.
            if lo >= 1
                density = StatsBase.normalize(hist; mode=:pdf)
                @test all(isfinite, density.weights)
                @test sum(density.weights .* diff(collect(edges))) ≈ 1
            end
        end
        for first_constant in (true, false), closed in (:left, :right)
            data = first_constant ? (fill(3.0, 3), [1.0, 2.0, 3.0]) :
                ([1.0, 2.0, 3.0], fill(3.0, 3))
            edges = BAT._get_edges(data, (100,), closed)
            @test length(edges) == 2
            hist = fit(Histogram, data, FrequencyWeights(ones(3)), edges; closed)
            @test sum(hist.weights) == 3
            samples = DensitySampleVector([collect(v) for v in zip(data...)], zeros(3))
            @test MarginalDist(samples, (1, 2); bins=100, closed) isa MarginalDist
        end
        data = ([-3.0, -1.0, 0.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0])
        for closed in (:left, :right), bins in ((30,), (30, 40))
            @test BAT._get_edges(data, bins, closed) ==
                StatsBase.histrange(data, StatsBase._nbins_tuple(data, bins), closed)
        end
        for scale in (1e-40, 1e-300)
            @test length(BAT._get_edges(scale .* [-2, -1, 0, 1, 2], 100, :left)) == 101
        end
        values = [1e16, nextfloat(nextfloat(1e16))]
        edges = BAT._get_edges(values, 100, :left)
        hist = StatsBase.normalize(fit(Histogram, values,
            FrequencyWeights([3.0, 2.0]), edges); mode=:pdf)
        regions, reported = BAT.get_smallest_intervals(hist, [0.5])
        actual = sum(only(regions).weights .* diff(edges))
        @test actual >= 0.5
        @test only(reported) ≈ actual
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
