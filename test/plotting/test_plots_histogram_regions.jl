# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Optional Plots lane: inspect converted series as well as actual GR rendering.
module TestPlotsHistogramRegions
using BAT, Test, StatsBase, Plots
using Printf: @sprintf

function marginal(edges, weights)
    h = Histogram(map(collect, edges), copy(weights), :left, true)
    dist = ndims(weights) == 1 ? BAT.EmpiricalDistributions.UvBinnedDist(h) :
        BAT.EmpiricalDistributions.MvBinnedDist(h)
    return BAT.MarginalDist(dist)
end

function recipe(marg, vsel=nothing; kwargs...)
    attrs = Dict{Symbol,Any}(kwargs)
    return vsel === nothing ? BAT.RecipesBase.apply_recipe(attrs, marg) :
        BAT.RecipesBase.apply_recipe(attrs, marg, vsel)
end

# Exercise renderer admission separately from EmpiricalDistributions' constructor
# (which cannot construct a zero-mass distribution). This test-owned distribution
# supplies only the same Histogram conversion used by the real MarginalDist recipe.
struct HistogramDisplayFixture{H} <: BAT.Distributions.ContinuousMultivariateDistribution
    hist::H
end
Base.convert(::Type{Histogram}, d::HistogramDisplayFixture) = deepcopy(d.hist)
raw_marginal(hist) = BAT.MarginalDist(HistogramDisplayFixture(deepcopy(hist)))

series_of(plot, type) = filter(s -> s[:seriestype] == type, plot.series_list)
recipe_of(series, type) = filter(s -> s.plotattributes[:seriestype] == type, series)
polygon_area(x, y) = abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in 1:length(x)-1)) / 2
masked_surface(w) = permutedims(map(v -> iszero(v) ? NaN : v, w))

function check_heatmap(series, edges, weights)
    @test series[:seriestype] == :heatmap
    @test series[:x] == edges[1]
    @test series[:y] == edges[2]
    @test isequal(series[:z].surf, masked_surface(weights))
    x, y = Plots.heatmap_edges(series[:x], :identity, series[:y], :identity, size(series[:z].surf))
    @test x == edges[1]
    @test y == edges[2]
end

@testset "Legacy Plots histogram region mass and geometry" begin
    Plots.gr()

    @testset "Complete edge paths and baseline polygons in both orientations" begin
        edges = [0.0, 2.0, 6.0, 8.0]
        marg = marginal((edges,), [0.05, 0.1, 0.25])
        hist = convert(Histogram, marg.dist)
        before = deepcopy(hist)
        path_x = [0.0, 2.0, 2.0, 6.0, 6.0, 8.0]
        path_y = repeat(hist.weights; inner=2)
        polygon_x = [0.0; path_x; 8.0; 0.0]
        polygon_y = [0.0; path_y; 0.0; 0.0]
        for orientation in (:vertical, :horizontal)
            swap = orientation == :horizontal
            for style in (:stephist, :steppost)
                p = Plots.plot(marg; seriestype=style, orientation=orientation,
                    linewidth=3, linestyle=:dash, label="density")
                s = only(p.series_list)
                @test s[:seriestype] == :path
                @test s[:x] == (swap ? path_y : path_x)
                @test s[:y] == (swap ? path_x : path_y)
                @test s[:linewidth] == 3
                @test s[:linestyle] == :dash
                @test s[:label] == "density"
            end
            p = Plots.plot(marg; seriestype=:histogram, orientation=orientation)
            s = only(p.series_list)
            @test s[:seriestype] == :shape
            @test s[:x] == (swap ? polygon_y : polygon_x)
            @test s[:y] == (swap ? polygon_x : polygon_y)
            @test first(s[:x]) == last(s[:x])
            @test first(s[:y]) == last(s[:y])
            @test polygon_area(s[:x], s[:y]) ≈ 1
        end
        after = convert(Histogram, marg.dist)
        @test after.edges == before.edges
        @test after.weights == before.weights

        for (e, w) in (([3.0, 7.0], [1.0]),
                       ([0.0, 2.0, 6.0, 8.0], [0.0, 0.0, 1.0]),
                       ([0.0, 1.0, 3.0, 4.0], [1.0, 0.0, 2.0]))
            m = marginal((e,), w)
            for orientation in (:vertical, :horizontal)
                p = Plots.plot(m; seriestype=:smallest_intervals, intervals=[1.0],
                    colors=[:green], orientation=orientation)
                s = only(series_of(p, :shape))
                coord = orientation == :vertical ? s[:x] : s[:y]
                height = orientation == :vertical ? s[:y] : s[:x]
                @test extrema(coord) == (first(e), last(e))
                @test polygon_area(s[:x], s[:y]) ≈ 1
                @test s[:label] == "whole-bin highest-density 100.00% region"
                # The zero-height gap is retained, rather than bridged above zero.
                if w == [1.0, 0.0, 2.0]
                    @test any(i -> coord[i] == 1 && coord[i+1] == 3 &&
                        height[i] == height[i+1] == 0, 1:length(coord)-1)
                end
            end
        end

        # An odd-ULP bin cannot be reconstructed reliably from center + width.
        odd_edges = [1e16, 1e16 + 6, 1e16 + 14]
        odd = marginal((odd_edges,), [1.0, 2.0])
        s = only(Plots.plot(odd; seriestype=:stephist).series_list)
        @test s[:x] == [odd_edges[1], odd_edges[2], odd_edges[2], odd_edges[3]]
    end

    @testset "Achieved content, stable labels/colors, and central tails" begin
        m = marginal(([0.0, 2.0, 6.0, 8.0],), [0.0, 0.15, 0.2])
        p = Plots.plot(m; seriestype=:smallest_intervals, intervals=[0.5], colors=[:green])
        s = only(series_of(p, :shape))
        @test s[:label] == "whole-bin highest-density 100.00% region"
        @test polygon_area(s[:x], s[:y]) ≈ 1
        requests = [0.4, 0.9, 0.4]
        colors = [:blue, :red, :green]
        labels = ["first tight", "broad", "second tight"]
        for style in (:smallest_intervals, :HDR, :central_intervals)
            out = recipe(m; seriestype=style, intervals=requests,
                colors=colors, interval_labels=labels)
            shapes = recipe_of(out, :shape)
            @test getindex.(getproperty.(shapes, :plotattributes), :label) == labels[[2, 1, 3]]
            @test getindex.(getproperty.(shapes, :plotattributes), :fillcolor) == colors[[2, 1, 3]]
        end
        @test requests == [0.4, 0.9, 0.4]
        @test colors == [:blue, :red, :green]
        @test labels == ["first tight", "broad", "second tight"]
        central = marginal(([0.0, 1.0, 3.0, 4.0],), [0.2, 0.3, 0.2])
        c = only(series_of(Plots.plot(central; seriestype=:central_intervals,
            intervals=[0.5], colors=[:blue]), :shape))
        @test c[:label] == "whole-bin central 60.00% region"
        @test polygon_area(c[:x], c[:y]) ≈ 0.6
        empty = only(recipe_of(recipe(m; seriestype=:smallest_intervals,
            intervals=[0.0], colors=[:blue]), :shape))
        @test empty.plotattributes[:label] == "whole-bin highest-density 0.00% region"
        @test all(iszero, empty.args[2])
        for style in (:smallest_intervals, :central_intervals)
            @test_throws ArgumentError recipe(m; seriestype=style, intervals=[0.2, 0.8], colors=[:red])
            @test_throws ArgumentError recipe(m; seriestype=style, intervals=[0.2, 0.8],
                interval_labels=["missing"], colors=[:red, :blue])
            @test_throws ArgumentError recipe(m; seriestype=style, intervals=[NaN])
        end
    end

    @testset "Actual edge heatmaps, matrix orientation, and region labels" begin
        edges = ([0.0, 2.0, 6.0, 8.0], [-2.0, -1.0, 3.0])
        weights = [0.0 6.0; 2.0 4.0; 5.0 1.0]
        m = marginal(edges, weights)
        hist = convert(Histogram, m.dist)
        p = Plots.plot(m, (1, 2); seriestype=:histogram2d)
        check_heatmap(only(p.series_list), edges, hist.weights)
        @test Plots.xlims(p)[1] <= first(edges[1]) < last(edges[1]) <= Plots.xlims(p)[2]
        @test Plots.ylims(p)[1] <= first(edges[2]) < last(edges[2]) <= Plots.ylims(p)[2]
        limited = Plots.plot(m, (1, 2); seriestype=:histogram2d, xlims=(1, 7), ylims=(-1, 2))
        @test Plots.xlims(limited) == (1, 7)
        @test Plots.ylims(limited) == (-1, 2)
        trans = marginal(reverse(edges), permutedims(weights))
        tp = Plots.plot(trans, (2, 1); seriestype=:histogram2d)
        check_heatmap(only(tp.series_list), reverse(edges), permutedims(hist.weights))
        @test convert(Histogram, m.dist).weights == hist.weights

        single = marginal(([0.0, 2.0], [0.0, 1.0, 4.0]), reshape([1.0, 2.0], 1, 2))
        single_hist = convert(Histogram, single.dist)
        single_plot = Plots.plot(single, (1, 2); seriestype=:histogram2d)
        check_heatmap(only(single_plot.series_list), single_hist.edges, single_hist.weights)

        # Independent mass table: [8 1; 12 9], total 30. Density prefix for
        # p=.6 retains all cells except (1,2), achieving 29/30.
        asymmetric = marginal(([0.0, 1.0, 4.0], [0.0, 2.0, 3.0]), [4.0 1.0; 2.0 3.0])
        ah = convert(Histogram, asymmetric.dist)
        rp = Plots.plot(asymmetric, (1, 2); seriestype=:smallest_intervals,
            intervals=[0.6], colors=[:orange])
        expected = copy(ah.weights)
        expected[1, 2] = 0
        check_heatmap(only(series_of(rp, :heatmap)), ah.edges, expected)
        @test only(series_of(rp, :shape))[:label] == "whole-bin highest-density 96.67% region"
        custom = recipe(asymmetric, (1, 2); seriestype=:smallest_intervals,
            intervals=[0.4, 0.9, 0.4], colors=[:blue, :red, :green],
            interval_labels=["first tight", "broad", "second tight"])
        shapes = recipe_of(custom, :shape)
        @test getindex.(getproperty.(shapes, :plotattributes), :label) == ["broad", "first tight", "second tight"]
        @test getindex.(getproperty.(shapes, :plotattributes), :fillcolor) == [:red, :blue, :green]
    end

    @testset "Paired contour field, duplicate ownership, and actual GR bands" begin
        m = marginal(([0.0, 1.0, 4.0], [0.0, 2.0, 3.0]), [4.0 1.0; 2.0 3.0])
        requests = [0.4, 0.9, 0.5, 0.0]
        colors = [:blue, :red, :green, :black]
        labels = ["tight", "broad", "coincident", "empty"]
        field = [1.0 0.25; 0.5 0.75]
        figures = Plots.Plot[]
        for style in (:smallest_intervals_contour, :smallest_intervals_contourf)
            p = Plots.plot(m, (1, 2); seriestype=style, intervals=requests,
                colors=colors, interval_labels=labels)
            push!(figures, p)
            @test length(p.series_list) == 2
            for (s, cutoff, color, label) in zip(p.series_list, [0.5, 0.75], [:red, :blue], ["broad", "tight"])
                @test s[:seriestype] == :contour
                @test s[:z].surf ≈ permutedims(field)
                @test s[:label] == label
                @test s[:linecolor] == Plots.plot_color(color)
                @test s[:x] == [0.5, 2.5]
                @test s[:y] == [1.0, 2.5]
                if style == :smallest_intervals_contourf
                    @test s[:levels] == [cutoff, 2.0]
                    @test s[:fillrange] !== nothing
                    @test s[:linewidth] == 0
                    @test s[:seriescolor][0.0] == Plots.plot_color(color)
                    @test s[:seriescolor][1.0] == Plots.plot_color(color)
                    @test Plots.gr_contour_levels(s, extrema(field)) == [cutoff]
                else
                    @test s[:levels] == [cutoff]
                    @test s[:fillrange] === nothing
                end
            end
            automatic = recipe(m, (1, 2); seriestype=style, intervals=[0.9, 0.4], colors=[:red, :blue])
            @test getindex.(getproperty.(automatic, :plotattributes), :label) ==
                ["whole-bin highest-density 96.67% region", "whole-bin highest-density 56.67% region"]
        end
        mktempdir() do dir
            path = joinpath(dir, "histogram-region-contours.png")
            Plots.savefig(Plots.plot(figures...; layout=(1, 2), size=(900, 400)), path)
            @test filesize(path) > 0
        end

        # Independent two-by-two Gaussian sum. Kernel normalization cancels
        # when rescaling the displayed field to max=1; all cells lie inside
        # the chosen kernel's support. No BAT convolution/region helper is used.
        sigma = 0.8
        smooth = [sum(field[a, b] * exp(-((i-a)^2 + (j-b)^2)/(2 * sigma^2))
            for a in 1:2, b in 1:2) for i in 1:2, j in 1:2]
        smooth ./= maximum(smooth)
        volumes = [2.0 1.0; 6.0 3.0]
        masses = smooth .* volumes
        cutoff = first(tau for tau in sort(unique(vec(smooth)); rev=true)
            if sum(masses[smooth .>= tau]) >= 0.6 * sum(masses))
        content = sum(masses[smooth .>= cutoff]) / sum(masses)
        smoothed = Plots.plot(m, (1, 2); seriestype=:smallest_intervals_contour,
            intervals=[0.6], colors=[:blue], smoothing=sigma)
        s = only(smoothed.series_list)
        @test s[:z].surf ≈ permutedims(smooth)
        @test s[:levels] ≈ [cutoff]
        @test s[:label] == "whole-bin highest-density $(@sprintf("%.2f", 100 * content))% region"
        @test !isapprox(s[:z].surf, permutedims(field))

        flat = marginal(([0.0, 1.0, 4.0], [0.0, 2.0, 3.0]), ones(2, 2))
        flat_plot = Plots.plot(flat, (1, 2); seriestype=:smallest_intervals_contourf,
            intervals=[0.9, 0.1], colors=[:red, :green])
        @test length(flat_plot.series_list) == 1
        @test only(flat_plot.series_list)[:levels] == [1.0, 2.0]
        @test only(flat_plot.series_list)[:label] == "whole-bin highest-density 100.00% region"
        @test only(flat_plot.series_list)[:linecolor] == Plots.plot_color(:green)
        mktempdir() do dir
            path = joinpath(dir, "flat-histogram-region.png")
            Plots.savefig(flat_plot, path)
            @test filesize(path) > 0
        end
    end

    @testset "Contour admission, empty regions, and unsupported geometry" begin
        edges = ([0.0, 1.0, 4.0], [0.0, 2.0, 3.0])
        m = marginal(edges, [4.0 1.0; 2.0 3.0])
        for style in (:smallest_intervals_contour, :smallest_intervals_contourf)
            @test isempty(recipe(m, (1, 2); seriestype=style, intervals=Float64[], colors=Symbol[]))
            @test isempty(recipe(m, (1, 2); seriestype=style, intervals=[0.0], colors=[:red]))
            zero = raw_marginal(Histogram(edges, zeros(2, 2), :left, true))
            @test isempty(recipe(zero, (1, 2); seriestype=style, intervals=[0.5]))
            @test_throws ArgumentError recipe(zero, (1, 2); seriestype=style, intervals=[NaN])
            @test_throws ArgumentError recipe(m, (1, 2); seriestype=style, intervals=[0.4, 0.8], colors=[:red])
            @test_throws ArgumentError recipe(m, (1, 2); seriestype=style,
                intervals=[0.4, 0.8], interval_labels=["missing"])
            @test_throws ArgumentError recipe(m, (1, 2); seriestype=style, smoothing=-1)
            @test_throws ArgumentError recipe(m, (1, 2); seriestype=style, smoothing=NaN)
            @test_throws ArgumentError recipe(m, (1, 2); seriestype=style, contour_labels=true)
            @test_throws ArgumentError Plots.plot(m, (1, 2); seriestype=style, clabels=true)
            narrow = marginal(([0.0, 1.0], [0.0, 1.0, 2.0]), ones(1, 2))
            @test_throws ArgumentError recipe(narrow, (1, 2); seriestype=style)
            @test isempty(recipe(narrow, (1, 2); seriestype=style, intervals=[0.0]))
            collapsed = raw_marginal(Histogram(
                ([1e16 + 2, 1e16 + 4, 1e16 + 6], [0.0, 1.0, 2.0]),
                ones(2, 2), :left, true))
            @test_throws ArgumentError recipe(collapsed, (1, 2); seriestype=style)
            @test isempty(recipe(collapsed, (1, 2); seriestype=style, intervals=[0.0]))

            # Count masses are benign; their relative density field is not
            # representable. A zero from conversion must be rejected before
            # smoothing could conceal that half the original mass was lost.
            extreme = raw_marginal(Histogram(([0.0, 1e-308, 1e308], [0.0, 1.0, 2.0]),
                ones(2, 2), :left, false))
            @test_throws ArgumentError recipe(extreme, (1, 2); seriestype=style, smoothing=0)
            @test_throws ArgumentError recipe(extreme, (1, 2); seriestype=style, smoothing=1)
        end

        # Native Int64 subtraction overflows for the first width, although
        # these endpoints, widths and centers are exactly representable in
        # the core's Float64 working type. The old center helper produced
        # -3*2^62, outside the first cell, while remaining increasing.
        integer_grid = Histogram((Int64[typemin(Int64), 0, Int64(2)^62], [0.0, 1.0, 2.0]),
            [4.0 1.0; 2.0 3.0], :left, true)
        expected_x = Float64[-Int64(2)^62, Int64(2)^61]
        expected_y = [0.5, 1.5]
        @test BAT._histogram_contour_centers(integer_grid) == (expected_x, expected_y)
        integer_marg = raw_marginal(integer_grid)
        # These rounded centers would be increasing, but the first is on
        # its left edge: there is no interior Float64 point in that cell.
        boundary_grid = Histogram(([1e16, 1e16 + 2, 1e16 + 6], [0.0, 1.0, 2.0]),
            ones(2, 2), :left, true)
        @test_throws ArgumentError BAT._histogram_contour_centers(boundary_grid)
        boundary_marg = raw_marginal(boundary_grid)
        for style in (:smallest_intervals_contour, :smallest_intervals_contourf)
            displayed = only(recipe(integer_marg, (1, 2); seriestype=style,
                intervals=[0.5], colors=[:blue]))
            @test displayed.args[1] == expected_x
            @test displayed.args[2] == expected_y
            @test_throws ArgumentError recipe(boundary_marg, (1, 2); seriestype=style,
                intervals=[0.5], colors=[:blue])
            @test isempty(recipe(boundary_marg, (1, 2); seriestype=style,
                intervals=[0.0], colors=[:blue]))
        end

        # Relative field/threshold scaling is independent of an absolute
        # density multiplier. This adapter fixture avoids a separate
        # distribution constructor's absolute normalization arithmetic.
        for scale in (1e-300, 1.0, 1e300)
            scaled = raw_marginal(Histogram(edges, scale .* [4.0 1.0; 2.0 3.0], :left, true))
            out = only(recipe(scaled, (1, 2); seriestype=:smallest_intervals_contour,
                intervals=[0.6], colors=[:blue]))
            @test out.args[3] ≈ [1.0 0.5; 0.25 0.75]
            @test out.plotattributes[:levels] ≈ [0.5]
            @test out.plotattributes[:label] == "whole-bin highest-density 96.67% region"
        end
    end

    @testset "Public marginal composition and rendered cell regions" begin
        m = marginal(([0.0, 1.0, 4.0], [0.0, 2.0, 3.0]), [4.0 1.0; 2.0 3.0])
        cell_plot = Plots.plot(m, (1, 2); seriestype=:smallest_intervals,
            intervals=[0.6, 0.9], colors=[:green, :red])
        one = marginal(([0.0, 2.0, 6.0, 8.0],), [0.0, 0.15, 0.2])
        horizontal = Plots.plot(one; seriestype=:smallest_intervals,
            orientation=:horizontal, intervals=[0.5], colors=[:green])
        mktempdir() do dir
            path = joinpath(dir, "histogram-region-cells.png")
            Plots.savefig(Plots.plot(cell_plot, horizontal; layout=(1, 2)), path)
            @test filesize(path) > 0
        end
        # The marginal composite delegates the right panel to the same
        # horizontal path. Inspect its recipe before random remarginalization.
        composite = recipe(m, (1, 2); seriestype=:marginal,
            right=Dict("seriestype" => :smallest_intervals, "intervals" => [0.5]))
        right = only(filter(s -> get(s.plotattributes, :subplot, nothing) == 4, composite))
        @test right.plotattributes[:orientation] == :horizontal
        @test right.plotattributes[:seriestype] == :smallest_intervals
        @test right.plotattributes[:intervals] == [0.5]
    end

    @testset "Bounded tuple smoothing through samples and marginals" begin
        bins = (0.:1.:3., 0.:2.:8.)
        weights = [1. 4. 2. 3.; 5. 2. 9. 1.; 2. 6. 1. 7.]
        centers = map(e -> (e[1:end-1] .+ e[2:end]) ./ 2, bins)
        samples = DensitySampleVector(v=[[centers[1][i],centers[2][j]]
            for j in 1:4 for i in 1:3], logd=zeros(12), weight=vec(copy(weights)))
        before = deepcopy(samples)
        m = marginal(map(collect,bins), weights)
        figures = Plots.Plot[]
        for style in (:smallest_intervals_contour, :smallest_intervals_contourf),
            sigmas in ((0.5,1.), (0.,1.), (1.,0.), (0.,0.))
            # Direct finite-support sum, including a zero axis and edge crop.
            factor(d,s) = iszero(s) ? Float64(d == 0) : exp(-d^2/(2s^2))
            smooth = [sum(weights[a,b]*factor(i-a,sigmas[1])*factor(j-b,sigmas[2])
                for a in 1:3, b in 1:4 if abs(i-a) <= 2ceil(Int,sigmas[1]) &&
                    abs(j-b) <= 2ceil(Int,sigmas[2])) for i in 1:3, j in 1:4]
            smooth ./= maximum(smooth)
            cutoff = first(t for t in sort(unique(vec(smooth)); rev=true)
                if sum(smooth[smooth .>= t]) >= 0.6sum(smooth))
            for input in (m, samples)
                p = Plots.plot(input, (1,2); seriestype=style, bins=bins,
                    smoothing=sigmas, intervals=[0.6], colors=[:blue],
                    interval_labels=["selected"], marginalmode=false, mean=false, std=false)
                s = only(series_of(p,:contour))
                @test s[:z].surf ≈ permutedims(smooth) rtol=5e-14 atol=1e-15
                @test first(s[:levels]) ≈ cutoff rtol=5e-14
                @test s[:x] == centers[1]
                @test s[:y] == centers[2]
                @test s[:label] == "selected"
                @test s[:linecolor] == Plots.plot_color(:blue)
                @test (s[:fillrange] !== nothing) == (style == :smallest_intervals_contourf)
                push!(figures,p)
            end
        end
        for style in (:smallest_intervals_contour, :smallest_intervals_contourf)
            for input in (m,samples), sigma in (1e6, ldexp(1.,62), (0.,1e6), (-1.,0.), (0.,Inf))
                @test_throws ArgumentError Plots.plot(input, (1,2); seriestype=style,
                    bins=bins, smoothing=sigma, marginalmode=false)
            end
            for intervals in (Float64[], [0.])
                @test isempty(recipe(m, (1,2); seriestype=style,
                    intervals=intervals, smoothing=(1e6,0.)))
            end
            zero = raw_marginal(Histogram(map(collect,bins), zeros(3,4), :left, true))
            @test isempty(recipe(zero, (1,2); seriestype=style, smoothing=(0.,1e6)))
        end
        @test samples == before
        mktempdir() do dir
            for (i,p) in enumerate(figures)
                path = joinpath(dir,"tuple-contour-$i.png")
                Plots.savefig(p,path)
                @test filesize(path) > 0
            end
        end
    end
end
end
