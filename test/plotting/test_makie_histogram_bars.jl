# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestMakieHistogramBars
using BAT, Makie, CairoMakie, Test

const EXT = Base.get_extension(BAT, :BATMakieExt)
const AUTO = Makie.automatic
# An independent statement of the documented bound, not the production constant.
const EPS_GEOM = 1 // (BigInt(1) << 18)

function nativebar(p)
    p isa Makie.BarPlot && return p
    for child in p.plots
        found = nativebar(child)
        isnothing(found) || return found
    end
    nothing
end
function expected_edges(edges, gap, dodge, n_dodge, dodge_gap)
    # Independent exact endpoint/area oracle, not the native layout helpers.
    nd = n_dodge === AUTO ? (dodge === AUTO ? 1 : maximum(dodge)) : n_dodge
    d = dodge === AUTO ? fill(1, length(edges) - 1) : dodge
    g = Rational{BigInt}(gap)
    dg = Rational{BigInt}(dodge_gap)
    fraction = (1 - (nd - 1) * dg) / nd
    map(1:length(edges)-1) do i
        a, b = Rational{BigInt}(edges[i]), Rational{BigInt}(edges[i+1])
        group = (b - a) * (1 - g)
        offset = (fraction - 1) / 2 + (d[i] - 1) * (fraction + dg)
        mid = (a + b) / 2 + group * offset
        half = abs(group * fraction) / 2
        (mid - half, mid + half, b - a)
    end
end
@testset "H37 independent styled rectangle oracle" begin
    fixtures = [
        [-0.05, 0.05],
        [1.9, 2.1],
        [9.5e7, 1.05e8],
        collect(range(0.0, 1.0; length = 101)),
        collect(range(1e8, 1e8 + 1; length = 101)),
        [-2.0, -1.1, 0.0, 0.03, 5.0],
    ]
    for edges in fixtures, gap in (0.0, 0.2), dodged in (false, true)
        n = length(edges) - 1
        d = dodged ? [isodd(i) ? 1 : 2 for i = 1:n] : AUTO
        nd = dodged ? 2 : AUTO
        dg = 0.03
        heights = collect(range(0.1; step = 0.9 / max(n - 1, 1), length = n))
        f = Figure()
        ax = Axis(f[1, 1])
        try
            centers = edges[1:end-1] .+ diff(edges) ./ 2
            ref = barplot!(
                ax,
                centers,
                heights;
                width = diff(edges),
                gap,
                dodge = d,
                n_dodge = nd,
                dodge_gap = dg,
            )
            oracle = expected_edges(edges, gap, d, nd, dg)
            reference_rects = deepcopy(ref.plots[1][1][])
            errors = map(zip(reference_rects, oracle)) do (rect, (lo, hi, original_width))
                al = Rational{BigInt}(rect.origin[1])
                ar = Rational{BigInt}(rect.origin[1] + rect.widths[1])
                max(abs(al - lo), abs(ar - hi)) / (hi == lo ? original_width : hi - lo)
            end
            # This specific style/offset combination exceeds the proposed bound;
            # retain it as a rejection test, without loosening the bound.
            expected_rejection = first(edges) == 1e8 && n == 100 && gap == 0.2 && dodged
            @test (maximum(errors) > EPS_GEOM) == expected_rejection
            if expected_rejection
                @test_throws ArgumentError EXT._checked_hist1d_bar_layout(
                    edges,
                    gap,
                    d,
                    nd,
                    dg,
                    AUTO,
                )
            else
                p = EXT.batcheckedhistbars!(
                    ax,
                    edges,
                    heights;
                    gap,
                    dodge = d,
                    n_dodge = nd,
                    dodge_gap = dg,
                )
                b = nativebar(p)
                actual = b.plots[1][1][]
                @test actual == reference_rects
                for (rect, (lo, hi, original_width)) in zip(actual, oracle)
                    al = Rational{BigInt}(rect.origin[1])
                    ar = Rational{BigInt}(rect.origin[1] + rect.widths[1])
                    scale = hi == lo ? original_width : hi - lo
                    @test abs(al - lo) <= EPS_GEOM * scale
                    @test abs(ar - hi) <= EPS_GEOM * scale
                    @test abs((ar - al) - (hi - lo)) <= 2EPS_GEOM * scale
                end
                @test last.(b[1][]) == heights
                @test b.width[] == diff(edges)
            end
        finally
            empty!(f)
        end
    end
end
@testset "H37 themes, layouts and failure ordering" begin
    theme = Theme(
        BarPlot = (
            color = :purple,
            alpha = 0.4,
            gap = 0.15,
            strokecolor = :orange,
            strokewidth = 2,
            bar_labels = :y,
            label_color = :red,
            label_size = 17,
        ),
    )
    with_theme(theme) do
        f = Figure()
        ax = Axis(f[1, 1])
        control = Axis(f[1, 2])
        try
            p = EXT.batcheckedhistbars!(ax, [0.0, 0.2, 1.0], [0.4, 0.8])
            b = nativebar(p)
            ref = barplot!(
                control,
                [0.0, 0.2] .+ diff([0.0, 0.2, 1.0]) ./ 2,
                [0.4, 0.8];
                width = diff([0.0, 0.2, 1.0]),
            )
            for key in (
                :color,
                :alpha,
                :gap,
                :strokecolor,
                :strokewidth,
                :bar_labels,
                :label_color,
                :label_size,
            )
                @test b[key][] == ref[key][]
            end
            @test length(b.plots) == length(ref.plots) == 2
            @test b.plots[2][1][] == ref.plots[2][1][]
            @test !isempty(Makie.colorbuffer(f))
            # y-only updates must not recompute exact bin layout.
            old = p.checked_centers[]
            Makie.update!(p; arg2 = [0.7, 0.9])
            @test p.checked_centers[] === old
            @test last.(b[1][]) == [0.7, 0.9]
            Makie.update!(p; gap = 0.3)
            @test b.gap[] == 0.3
        finally
            empty!(f)
        end
    end
    # A valid canonical near-ULP bin can become invalid when a gap is requested.
    f = Figure()
    ax = Axis(f[1, 1])
    try
        p = EXT.batcheckedhistbars!(ax, [1e16, 1e16 + 4], [0.25]; gap = 0.0)
        b = nativebar(p)
        before = deepcopy(b.plots[1][1][])
        err = try
            Makie.update!(p; gap = 0.5)
            p.checked_centers[]
            nothing
        catch e
            e
        end
        @test err !== nothing && occursin("Hist1D bin geometry", sprint(showerror, err))
        # The failing validation publishes no new native geometry controls.
        gap_error = try
            b.gap[]
            nothing
        catch e
            e
        end
        @test gap_error !== nothing && occursin("Hist1D bin geometry", sprint(showerror, gap_error))
        @test b.plots[1][1][] == before
    finally
        empty!(f)
    end
    for edges in ([1e16 + 2, 1e16 + 4], [1.0, 1.0], [0.0, Inf]), gap in (0.0, 0.2)
        @test_throws ArgumentError EXT._checked_hist1d_bar_layout(
            edges,
            gap,
            AUTO,
            AUTO,
            0.03,
            AUTO,
        )
    end
end
@testset "H37 public sample paths and repeated empty windows" begin
    for ordinary in (false, true), point in (0.0, 2.0, 1e8)
        source = DensitySampleVector([[point]], zeros(1))
        rec = (upper = Hist2D, diagonal = Hist1D, lower = Hist2D)
        f = ordinary ? Figure() : bat_makie_plot(source, rec, [1], 1; display = false)
        try
            ordinary && with_theme(Theme(BarPlot = (gap = 0.0, width = 123.0))) do
                Makie.plot(f[1, 1], source; recipes = rec, vsel = [1], N_max = 1)
            end
            ax = only(filter(x -> x isa Axis, contents(f[1, 1][1, 1])))
            b = only(filter(!isnothing, nativebar.(ax.scene.plots)))
            r = only(b.plots[1][1][])
            pad = max(abs(point), 1.0) * 0.05
            @test r.origin[1] == point - pad
            @test r.origin[1] + r.widths[1] == point + pad
            @test r.widths[1] * only(b[1][])[2] ≈ 1.0
            @test !isempty(Makie.colorbuffer(f))
        finally
            empty!(f)
        end
    end
    source = DensitySampleVector([[0.1], [0.3], [0.6], [0.9]], zeros(4))
    graph, _ = EXT._setup_static_graph(
        source,
        (upper = Hist2D, diagonal = Hist1D, lower = Hist2D),
        [1],
        1,
        BAT._default_makie_triagonal_config(),
        BAT._default_makie_diagonal_config(),
    )
    f = Figure()
    try
        EXT._bind_interactive_grid!(f, EXT._init_gridlayout!(graph, 1))
        ax = only(filter(x -> x isa Axis, contents(f[1, 1][1, 1])))
        for _ = 1:3
            Makie.update!(graph; window_steps = (Int64(20), Int64(30)))
            @test isempty(filter(!isnothing, nativebar.(ax.scene.plots)))
            Makie.update!(graph; window_steps = nothing)
            @test length(filter(!isnothing, nativebar.(ax.scene.plots))) == 1
        end
        @test !isempty(Makie.colorbuffer(f))
    finally
        empty!(f)
    end
end

@testset "H37 native attribute forwarding" begin
    for kw in (
        (direction = :x, fillto = -0.4, offset = 0.1),
        (stack = [1, 2],),
        (gap = 1.0,),
        (gap = -0.1,),
        (gap = 1.2,),
        (gap = 0.2f0, dodge_gap = 0.03f0),
        (gap = 1 // 5, dodge_gap = 3 // 100),
        (dodge = 2, n_dodge = 3, dodge_gap = 0.1),
    )
        f = Figure()
        try
            p = EXT.batcheckedhistbars!(Axis(f[1, 1]), [0.0, 0.5, 1.0], [0.2, 0.3]; kw...)
            b = only(p.plots)
            ref = barplot!(Axis(f[1, 2]), [0.25, 0.75], [0.2, 0.3]; width = [0.5, 0.5], kw...)
            @test b.plots[1][1][] == ref.plots[1][1][]
            @test Makie.data_limits(p) == Makie.data_limits(b)
            @test Makie.boundingbox(p) == Makie.boundingbox(b)
            @test !isempty(Makie.colorbuffer(f))
        finally
            empty!(f)
        end
    end
    for (gap, d, nd, dg, w) in (
        (NaN, AUTO, AUTO, 0.03, AUTO),
        (0.0, [0], AUTO, 0.03, AUTO),
        (0.0, [1.0], AUTO, 0.03, AUTO),
        (0.0, AUTO, 0, 0.03, AUTO),
        (0.0, AUTO, 2.0, 0.03, AUTO),
        (0.0, AUTO, AUTO, Inf, AUTO),
        (0.0, AUTO, AUTO, 0.03, 1.0),
    )
        @test_throws ArgumentError EXT._checked_hist1d_bar_layout([0.0, 1.0], gap, d, nd, dg, w)
    end
end
@testset "H37 public rejection and height-only SpecApi reuse" begin
    src = DensitySampleVector([[1e16], [nextfloat(1e16)]], zeros(2))
    rec = (upper = Hist2D, diagonal = Hist1D, lower = Hist2D)
    f = Figure()
    try
        err = try
            Makie.plot(f[1, 1], src; recipes = rec, vsel = [1], N_max = 1)
            nothing
        catch e
            e
        end
        @test err !== nothing && occursin("Hist1D bin geometry", sprint(showerror, err))
    finally
        empty!(f)
    end
    f = bat_makie_plot(
        src,
        (upper = Hist2D, diagonal = QuantileHist1D, lower = Hist2D),
        [1],
        1;
        display = false,
    )
    try
        @test !isempty(Makie.colorbuffer(f))
    finally
        empty!(f)
    end
    src = DensitySampleVector([[0.1], [0.3], [0.6], [0.9]], zeros(4))
    graph, _ = EXT._setup_static_graph(
        src,
        rec,
        [1],
        1,
        BAT._default_makie_triagonal_config(),
        BAT._default_makie_diagonal_config(),
    )
    f = Figure()
    try
        EXT._bind_interactive_grid!(f, EXT._init_gridlayout!(graph, 1))
        ax = only(filter(x -> x isa Axis, contents(f[1, 1][1, 1])))
        p = only(filter(x -> x isa EXT.BATCheckedHistBars, ax.scene.plots))
        cached = p.checked_centers[]
        oldheights = copy(p.bin_heights[])
        Makie.update!(graph; window_steps = (Int64(1), Int64(1)))
        @test only(filter(x -> x isa EXT.BATCheckedHistBars, ax.scene.plots)) === p
        @test p.checked_centers[] === cached
        @test p.bin_heights[] != oldheights
    finally
        empty!(f)
    end
end
function axis_spec(edges)
    weights = fill(0.25, length(edges) - 1)
    prim = (centers = edges[1:end-1] .+ diff(edges) ./ 2, weights = weights, edges = edges)
    Makie.SpecApi.Axis(
        plots = EXT.compose_plotspecs(prim, Hist1D(), BAT._default_makie_diagonal_config()),
        limits = (1e16 - 8, 1e16 + 12, 0.0, 1.0),
        xlabel = "x",
        ylabel = "y",
        xlabelvisible = true,
        ylabelvisible = true,
    )
end
@testset "H37 real live owner fails closed on unsafe bin geometry" begin
    vis = BATVisualizer(BATMakieVisualization())
    EXT._claim_live_visualizer!(vis)
    f = Figure()
    ax = Axis(f[1, 1])
    obs = Observable(Makie.PlotSpec[])
    Makie.update_plotspecs!(ax.scene, obs)
    good = axis_spec([1e16, 1e16 + 4])
    bad = axis_spec([1e16 + 2, 1e16 + 4])
    # Force the canonical gap rather than the ordinary Makie default.
    good.plots[1].kwargs[:gap] = 0.0
    bad.plots[1].kwargs[:gap] = 0.0
    try
        @test EXT._with_live_view(vis) do
            EXT._attach_live_view!(vis, f)
            EXT._update_grid_axis!(ax, obs, good, nothing)
        end
        EXT._start_live_listener!(vis, () -> nothing, () -> nothing)
        @test !EXT._with_live_view(vis) do
            EXT._update_grid_axis!(ax, obs, bad, good.kwargs)
        end
        BAT.finish_visualizer!(vis; success = true)
        status = BAT.visualizer_status(vis)
        @test status.outcome === :failed
        @test !status.presentation_valid && !status.data_complete
        @test status.quiesced && status.view_disposed && !status.cleanup_pending
        @test isempty(f.content)
        @test isempty(Makie.Observables.listeners(obs))
    finally
        BAT.request_visualizer_stop!(vis)
        BAT.finish_visualizer!(vis; success = false)
        empty!(f)
    end
end

@testset "Histogram bars retain closure, normalization and zero-height bins" begin
    coordinates, weights = [0.0, 0.25, 1.0, 2.0], [1.0, 2.0, 0.0, 3.0]
    samples = DensitySampleVector([[x] for x in coordinates], zeros(4); weight = weights)
    for closed in (:left, :right), normalization in (:none, :probability, :density, :pdf)
        config = merge(BAT._default_makie_diagonal_config(), (nbins = 3, closed, normalization))
        graph, _ = EXT._setup_static_graph(
            samples,
            (upper = Hist2D, diagonal = Hist1D, lower = Hist2D),
            [1],
            1,
            BAT._default_makie_triagonal_config(),
            config,
        )
        prim = graph[EXT.primitive_symbol(Hist1D, (1, 1))][]
        edges, heights = copy(prim.edges), copy(prim.weights)
        counts = [
            sum(
                (
                    w for (x, w) in zip(coordinates, weights) if
                    closed === :left ? edges[i] <= x < edges[i+1] : edges[i] < x <= edges[i+1]
                );
                init = 0.0,
            ) for i in eachindex(heights)
        ]
        expected =
            normalization === :none ? counts :
            normalization === :probability ? counts ./ sum(counts) :
            normalization === :density ? counts ./ diff(edges) :
            counts ./ sum(counts) ./ diff(edges)
        @test heights ≈ expected
        @test any(iszero, heights)
        fig = Figure()
        try
            with_theme(Theme(BarPlot = (gap = 0.0,))) do
                EXT._bind_interactive_grid!(fig, EXT._init_gridlayout!(graph, 1))
            end
            ax = only(filter(x -> x isa Axis, contents(fig[1, 1][1, 1])))
            bar = only(filter(!isnothing, nativebar.(ax.scene.plots)))
            @test last.(bar[1][]) == heights
            @test bar.width[] == diff(edges)
            rectangles = bar.plots[1][1][]
            @test length(rectangles) == length(heights)
            for i in eachindex(heights)
                @test rectangles[i].origin[1] == edges[i]
                @test rectangles[i].origin[1] + rectangles[i].widths[1] == edges[i+1]
                @test rectangles[i].widths[2] == heights[i]
            end
            stairs = only(filter(x -> x isa Makie.Stairs, ax.scene.plots))
            @test first.(stairs[1][]) == edges
            @test last.(stairs[1][]) == vcat(heights, heights[end])
            @test prim.edges == edges && prim.weights == heights
            @test !isempty(Makie.colorbuffer(fig))
        finally
            empty!(fig)
        end
    end
end

@testset "Histogram bin count updates publish matching native geometry" begin
    fig = Figure()
    try
        plot =
            EXT.batcheckedhistbars!(Axis(fig[1, 1]), [0.0, 1.0], [0.0]; gap = 0.0, bar_labels = :y)
        bar = only(plot.plots)
        for edges in ([0.0, 0.25, 0.5, 1.0], [-1.0, 0.0, 1.0], [0.0, 1.0], [0.0, 0.125, 0.5, 1.0])
            heights = Float64.(0:(length(edges)-2))
            before = copy(edges)
            Makie.update!(plot; arg1 = edges, arg2 = heights)
            rectangles = bar.plots[1][1][]
            @test length(rectangles) == length(heights)
            @test [(r.origin[1], r.origin[1] + r.widths[1]) for r in rectangles] == collect(zip(edges[1:end-1], edges[2:end]))
            @test last.(bar[1][]) == heights
            @test length(bar.plots[2][1][]) == length(heights)
            @test edges == before
        end
        @test !isempty(Makie.colorbuffer(fig))
    finally
        empty!(fig)
    end
end
end # module
