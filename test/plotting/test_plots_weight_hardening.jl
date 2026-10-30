# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Optional Plots lane: the core test environment need not load a plotting backend.
module TestPlotsWeightHardening
using BAT, Test, Statistics, StatsBase, Plots, ValueShapes

fixture(data, weights; logd=zeros(length(weights))) = DensitySampleVector(
    v=[collect(c) for c in eachcol(data)], logd=logd, weight=weights)
function recipe(samples, vsel; kwargs...)
    attrs = Dict{Symbol,Any}(:marginalmode => false, :seriescolor => :green, :bins => 8)
    merge!(attrs, Dict{Symbol,Any}(kwargs))
    return BAT.RecipesBase.apply_recipe(attrs, samples, vsel)
end
with_label(series, name) = only(filter(s -> get(s.plotattributes, :label, nothing) == name, series))

@testset "Legacy Plots weight arithmetic and selected overlays" begin
    @testset "Frequency correction and ordinary accumulator parity" begin
        data = [0.0 1.0 2.0; 1.0 3.0 5.0; 4.0 -1.0 3.0]
        weights = [1, 2, 3]
        samples = fixture(data, weights)
        before = deepcopy(samples)
        old = BAT.MCMCBasicStats(samples)
        for idxs in ((1,), (2,), (1, 2), (3, 1))
            actual = BAT._legacy_plot_moments(samples, idxs; spread=true)
            @test actual.mean == [old.param_stats.mean[i] for i in idxs]
            @test actual.std == [sqrt(old.param_stats.cov[i, i]) for i in idxs]
        end
        result = BAT._legacy_plot_moments(samples, (1, 2); spread=true)
        # Independent frequency-weight oracle: S=6, A_x=10/3, divide by S-1.
        @test result.mean ≈ [4/3, 11/3]
        @test result.std ≈ [sqrt(2/3), sqrt(8/3)]
        scaled = BAT._legacy_plot_moments(fixture(data, 2weights), (1,); spread=true)
        @test only(scaled.mean) ≈ 4/3
        @test only(scaled.std) ≈ sqrt(20/33)
        @test only(scaled.std) != result.std[1]

        one = recipe(samples, 1; mean=true, std=true)
        meanline = with_label(one, "mean")
        @test first(meanline.args) == BAT.line(result.mean[1], last(first(meanline.args))[2])
        band = only(with_label(one, "std. dev.").args)
        @test collect(extrema(band.x)) ≈ [result.mean[1]-result.std[1], result.mean[1]+result.std[1]]
        two = recipe(samples, (1, 2); mean=true, std=true)
        point = with_label(two, "mean")
        @test point.args == ([result.mean[1]], [result.mean[2]])
        @test point.plotattributes[:xerror] == result.std[1]
        @test point.plotattributes[:yerror] == result.std[2]
        @test samples == before
    end

    @testset "Unused statistics do not control admission" begin
        data = [0.0 1.0 2.0; 2.0 1.0 0.0]
        huge = fixture(data, 5e307 .* [1.0, 2.0, 3.0])
        before = deepcopy(huge)
        for vsel in (1, (1, 2))
            @test !isempty(recipe(huge, vsel))
            @test !isempty(recipe(huge, vsel; marginalmode=true))
            @test_throws OverflowError recipe(huge, vsel; mean=true)
        end
        @test huge == before
        # Pure mean does not need a covariance whose intermediate squares overflow.
        wide = fixture([-1e200 1e200; 0.0 1.0], [1, 1])
        meanonly = recipe(wide, (1, 2); seriestype=:scatter, mean=true)
        @test with_label(meanonly, "mean").args == ([0.0], [0.5])
        @test_throws ArgumentError recipe(wide, (1, 2); seriestype=:scatter, mean=true, std=true)
        # Nonselected dimensions and unused log-density values are irrelevant.
        unused = fixture([0.0 1.0 2.0; 2.0 1.0 0.0; NaN Inf -Inf], [1, 2, 3]; logd=fill(NaN, 3))
        @test BAT._legacy_plot_moments(unused, (1, 2); spread=true).mean ≈ [4/3, 2/3]
        @test !isempty(recipe(unused, 1; mean=true))
        @test !isempty(recipe(unused, (1, 2); seriestype=:scatter, mean=true, std=true))
        @test_throws ArgumentError BAT._legacy_plot_moments(unused, (3,))
        filtered_bad = fixture([NaN 10.0 0.0 1.0; 0.0 1.0 2.0 3.0], [1e-8, 1e-6, 1.0, 1.0])
        @test_throws ArgumentError recipe(filtered_bad, (1, 2); filter=true)
        # An unused 1.03*peak overlay calculation must not invalidate the marginal.
        # H22 now rejects subnormal cached coordinates, even without overlays.
        narrow = fixture(reshape([2.8e-309, 2.8e-309], 1, :), [1, 1])
        @test_throws ArgumentError recipe(narrow, 1; bins=range(0.0, 5.6e-309; length=2))
        @test_throws ArgumentError recipe(narrow, 1; bins=range(0.0, 5.6e-309; length=2), mean=true)
        # Translate the same high-density case into the admitted coordinate
        # range; its finite PDF still overflows only in the requested overlay.
        width = ldexp(65.0, -1030)
        center = 128width
        translated = fixture(reshape([center, center], 1, :), [1, 1])
        bins = range(center - width/2, center + width/2; length=2)
        @test !isempty(recipe(translated, 1; bins))
        @test_throws ArgumentError recipe(translated, 1; bins, mean=true)
    end

    @testset "Used coordinates, spread and final geometry" begin
        zeros_ignored = fixture([NaN 1.0 3.0; Inf 2.0 4.0], [0, 1, 2])
        positive = fixture([1.0 3.0; 2.0 4.0], [1, 2])
        @test BAT._legacy_plot_moments(zeros_ignored, (1, 2); spread=true) ==
            BAT._legacy_plot_moments(positive, (1, 2); spread=true)
        for bad in ([1.0, -1.0], [1.0, NaN], [1.0, Inf])
            @test_throws ArgumentError BAT._legacy_plot_moments(fixture([0.0 1.0], bad), (1,))
        end
        @test_throws ArgumentError BAT._legacy_plot_moments(fixture([0.0 1.0], [0, 0]), (1,))
        tiny_count = fixture([0.0 1.0], [0.1, 0.2])
        @test only(BAT._legacy_plot_moments(tiny_count, (1,)).mean) ≈ 2/3
        @test_throws ArgumentError BAT._legacy_plot_moments(tiny_count, (1,); spread=true)
        constant = BAT._legacy_plot_moments(fixture([2.0 2.0], [1, 2]), (1,); spread=true)
        @test constant == (mean=[2.0], std=[0.0])
        @test_throws ArgumentError BAT._legacy_plot_moments(fixture([-1e-200 1e-200], [1, 1]), (1,); spread=true)
        @test_throws ArgumentError BAT._legacy_plot_moments(fixture([2.0 3.0], [1e308, 1.0]), (1,))
        @test_throws ArgumentError BAT._legacy_plot_interval(1.7e308, 1e308)
        @test_throws ArgumentError BAT._legacy_plot_interval(1e20, 1.0)
        @test_throws OverflowError BAT._legacy_plot_moments(
            fixture([0.0 1.0], [typemax(Int), typemax(Int)]), (1,))
    end

    @testset "Global mode keeps first ties and does not need count arithmetic" begin
        data = [10.0 20.0 30.0; 4.0 5.0 6.0; Inf Inf Inf]
        samples = fixture(data, [0, 2, 1]; logd=[3.0, 3.0, 1.0])
        @test BAT._legacy_plot_mode(samples, (1, 2)) == [10.0, 4.0]
        mode_reference = fixture(data[1:2, :], samples.weight; logd=samples.logd)
        @test BAT._legacy_plot_mode(samples, (1, 2)) == BAT.MCMCBasicStats(mode_reference).mode
        @test_throws ArgumentError BAT._legacy_plot_mode(samples, (3,))
        huge = fixture(data[1:2, :], 5e307 .* [1.0, 2.0, 3.0]; logd=[3.0, 3.0, 1.0])
        @test with_label(recipe(huge, (1, 2); globalmode=true), "global mode").args == ([10.0], [4.0])
        all_negative_inf = fixture(data[1:2, :], [1, 2, 1]; logd=fill(-Inf, 3))
        @test !isempty(recipe(all_negative_inf, (1, 2)))
        @test_throws ArgumentError recipe(all_negative_inf, (1, 2); globalmode=true)
        # The original strict comparison retains a prior winner after a NaN score.
        nan_score = fixture(data[1:2, :], [1, 2, 1]; logd=[1.0, NaN, 3.0])
        @test BAT._legacy_plot_mode(nan_score, (1, 2)) == BAT.MCMCBasicStats(nan_score).mode
    end

    @testset "Standalone scatter preserves rejected-weight encoding" begin
        data = [0.0 1.0 2.0 3.0; 4.0 5.0 6.0 7.0]
        signed = fixture(data, [-2.0, 0.0, 0.25, 4.0]; logd=[2.0, 2.0, 0.0, 0.0])
        before = deepcopy(signed)
        out = recipe(signed, (1, 2); seriestype=:scatter)
        accepted = with_label(out, "accepted")
        rejected = with_label(out, "rejected")
        @test accepted.args == ([2.0, 3.0], [6.0, 7.0])
        @test rejected.args == ([0.0, 1.0], [4.0, 5.0])
        @test accepted.plotattributes[:markersize] == [1.5, 3.0]
        @test getproperty.(accepted.plotattributes[:markercolor], :alpha) == [0.25, 1.0]
        mode = with_label(recipe(signed, (1, 2); seriestype=:scatter, globalmode=true), "global mode")
        @test mode.args == ([0.0], [4.0])
        @test_throws ArgumentError recipe(signed, (1, 2); seriestype=:scatter, marginalmode=true)
        @test_throws ArgumentError recipe(signed, (1, 2); seriestype=:scatter, mean=true)
        @test signed == before
        # Historically std without a mean marker is unused in the 2D recipe.
        @test length(recipe(signed, (1, 2); seriestype=:scatter, std=true)) == length(out)
        @test_throws ArgumentError recipe(fixture([0.0 Inf; 0.0 1.0], [-1, 0]), (1, 2); seriestype=:scatter)
        @test_throws ArgumentError recipe(fixture([0.0 1.0; 0.0 1.0], [1.0, Inf]), (1, 2); seriestype=:scatter)
        @test_throws ArgumentError recipe(fixture([0.0 1.0; 0.0 1.0], [1.0, 4.0]), (1, 2);
            seriestype=:scatter, markersize=1e308)
    end

    @testset "Public Plots integration" begin
        samples = fixture([0.0 1.0 2.0; 2.0 1.0 0.0], 5e307 .* [1.0, 2.0, 3.0])
        @test Plots.plot(samples, 1; marginalmode=false, bins=8) isa Plots.Plot
        plot2d = Plots.plot(samples, (1, 2); seriestype=:histogram2d, marginalmode=false, bins=(8, 8))
        @test plot2d isa Plots.Plot
        mktempdir() do dir
            path = joinpath(dir, "relative-histogram.png")
            Plots.savefig(plot2d, path)
            @test filesize(path) > 0
        end
        @test_throws OverflowError Plots.plot(samples, 1; mean=true, marginalmode=false, bins=8)
        ordinary = fixture([0.0 1.0 2.0; 2.0 1.0 0.0], [1, 2, 3])
        @test Plots.plot(ordinary, 1; mean=true, std=true, bins=8) isa Plots.Plot
        @test Plots.plot(ordinary, (1, 2); mean=true, std=true, bins=(8, 8)) isa Plots.Plot
    end
end
end
