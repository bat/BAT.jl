# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestMakieHistogramViewport
using BAT, Makie, CairoMakie, Test

const EXT = Base.get_extension(BAT, :BATMakieExt)
const PAIR = BAT._default_makie_triagonal_config()
const DIAG = BAT._default_makie_diagonal_config()
encloses(limits, edges) = isempty(edges) || (limits[1] <= minimum(edges) && maximum(edges) <= limits[2])
function fixtures()
    return [([0.1 0.9; 0.1 0.9], (2, 2), 2),
        ([-0.9 -0.1; 0.1 0.9; 2.1 2.9], (2, 7), 3),
        ([1.0 1.0; 0.1 0.9], (3, 2), 2),
        ([1e8 1e8+1; 0.1 0.9], (3, 2), 3),
        ([1e16 nextfloat(1e16); 0.1 0.9], (3, 2), 2),
        ([1e-3 9e-3; 0.1 0.9], (3, 2), 2)]
end

@testset "Fixed-domain histogram viewport contracts" begin
    for (data, bins, dbins) in fixtures(), closed in (:left, :right),
            diagtype in (Hist1D, QuantileHist1D),
            pairtypes in ((Hist2D, QuantileHist2D), (QuantileHist2D, Scatter2D), (Scatter2D, Hist2D))
        sample = DensitySampleVector([copy(v) for v in eachcol(data)], zeros(size(data, 2)))
        before = deepcopy(sample)
        n = size(data, 1)
        cfg = merge(PAIR, (nbins=bins, closed=closed, levels=[1.0]))
        dcfg = merge(DIAG, (nbins=dbins, closed=closed, levels=[1.0]))
        recipes = (upper=pairtypes[1], diagonal=diagtype, lower=pairtypes[2])
        # Reverse selection distinguishes actual variables from display slots.
        idxs = reverse(collect(1:n))
        graph, _ = EXT._setup_static_graph(sample, recipes, idxs, n, cfg, dcfg)
        limits = [graph[Symbol("axis_limits_$i")][] for i in 1:n]
        domain = [extrema(data[v, :]) for v in idxs]
        @test all(i -> encloses(limits[i], domain[i]), 1:n)
        for i in 1:n
            prim = graph[EXT.primitive_symbol(diagtype, (i, i))][]
            edges = diagtype === Hist1D ? prim.edges : first.(prim.stairs_data)
            @test encloses(limits[i], edges)
        end
        for i in 1:n, j in i+1:n, recipe in pairtypes
            prim = graph[EXT.primitive_symbol(recipe, (j, i))][]
            if recipe in (Hist2D, QuantileHist2D)
                @test encloses(limits[j], prim.xedges)
                @test encloses(limits[i], prim.yedges)
            end
        end
        cached = graph[:axis_limits][]
        # Even an empty visible window must retain the fixed-domain envelope.
        for window in ((Int64(1), Int64(1)), (Int64(20), Int64(30)), nothing)
            Makie.update!(graph; window_steps=window)
            @test [graph[Symbol("axis_limits_$i")][] for i in 1:n] == limits
            @test graph[:axis_limits][] === cached
        end
        @test sample == before
    end
end

@testset "Inactive histogram settings and domain replacement" begin
    badpair = merge(PAIR, (nbins=(0, 0),))
    baddiag = merge(DIAG, (nbins=0,))
    limits = EXT._histogram_axis_limits([0.0], [1.0], [1], 3,
        Mean1D, Scatter2D, Scatter2D, baddiag, badpair)
    @test limits == [(-0.05, 1.05), (0.0, 1.0), (0.0, 1.0)]
    @test EXT._histogram_axis_limits(Float64[], Float64[], [1], 3,
        Hist1D, Hist2D, Hist2D, DIAG, PAIR) == fill((0.0, 1.0), 3)
    @test EXT._histogram_axis_limits([0.0], [1.0], Int[], 3,
        Hist1D, Hist2D, Hist2D, DIAG, PAIR) == fill((0.0, 1.0), 3)
    @test_throws OverflowError EXT._histogram_axis_limits([0.0], [1.0], [1], 1,
        Hist1D, Scatter2D, Scatter2D, merge(DIAG, (nbins=typemax(Int),)), PAIR)
    @test EXT._histogram_axis_limits([0.0], [1.0], Int[], 3,
        Hist1D, Hist2D, Hist2D, baddiag, badpair) == fill((0.0, 1.0), 3)
    # Hist1D already interprets a configured zero as one bin; preserve it.
    @test all(isfinite, only(EXT._histogram_axis_limits([0.1], [0.9], [1], 1,
        Hist1D, Scatter2D, Scatter2D, baddiag, PAIR)))
    @test_throws ArgumentError EXT._histogram_axis_limits([0.0], [1.0], [1, 1], 1,
        Mean1D, Scatter2D, Scatter2D, DIAG, PAIR)
    # Bounds follow current domain inputs, not a historical maximum.
    graph, _ = EXT._setup_static_graph(DensitySampleVector([[0.1, 0.1], [0.9, 0.9]], zeros(2)),
        (upper=Hist2D, diagonal=QuantileHist1D, lower=Hist2D), [1, 2], 2,
        merge(PAIR, (nbins=(2, 2),)), merge(DIAG, (nbins=2,)))
    old = copy(graph[:axis_limits][])
    Makie.update!(graph; domain_lo=[-2.0, -3.0], domain_hi=[4.0, 5.0])
    @test all(i -> encloses(graph[:axis_limits][][i], old[i]), 1:2)
    Makie.update!(graph; domain_lo=[0.1, 0.1], domain_hi=[0.9, 0.9])
    @test graph[:axis_limits][] == old
end

cell(fig, i, j) = only(filter(x -> x isa Axis, contents(fig[1, 1][i, j])))
function heatmap(axis)
    plots = Makie.Heatmap[]
    function visit(p)
        p isa Makie.Heatmap && push!(plots, p)
        foreach(visit, p.plots)
    end
    foreach(visit, axis.scene.plots)
    return only(plots)
end

@testset "Generated histogram bounds and actual navigation" begin
    samples = DensitySampleVector([[0.1, 0.1], [0.9, 0.9]], zeros(2))
    graph, _ = EXT._setup_static_graph(samples,
        (upper=Hist2D, diagonal=QuantileHist1D, lower=QuantileHist2D), [1, 2], 2,
        merge(PAIR, (nbins=(2, 2), levels=[1.0])), merge(DIAG, (nbins=2,)))
    fig = Figure()
    try
        EXT._bind_interactive_grid!(fig, EXT._init_gridlayout(graph, 2))
        axes = [cell(fig, i, j) for i in 1:2, j in 1:2]
        # Independent regression oracle for the original clipped [0, 0.5, 1] bins.
        @test !isempty(Makie.colorbuffer(fig))
        for axis in (axes[1, 2], axes[2, 1])
            p = heatmap(axis)
            @test p[1][] == p[2][] == [0.0, 0.5, 1.0]
            @test axis.limits[] == ((0.0, 1.0), (0.0, 1.0))
            @test all(axis.finallimits[].origin .<= 0)
            @test all(axis.finallimits[].origin + axis.finallimits[].widths .>= 1)
        end
        ax = axes[1, 2]
        ax.title = "caller title"
        ax.backgroundcolor = :lavender
        limits!(ax, 0.2, 0.7, 0.3, 0.8)
        manual, target = deepcopy(ax.limits[]), ax.targetlimits[]
        cached = graph[:axis_limits][]
        for window in ((Int64(1), Int64(1)), (Int64(20), Int64(30)), nothing)
            Makie.update!(graph; window_steps=window)
            @test ax.limits[] == manual
            @test ax.targetlimits[] == target
            @test graph[:axis_limits][] === cached
        end
        Makie.update!(graph; upper_recipe=Scatter2D)
        @test ax.limits[] == manual && ax.targetlimits[] == target
        # Removing all histograms restores exactly the original padded domain.
        Makie.update!(graph; diagonal_recipe=Mean1D, lower_recipe=Scatter2D)
        @test all(l -> l[1] ≈ 0.06 && l[2] ≈ 0.94, ax.limits[])
        @test ax.targetlimits[] != target
        @test ax.title[] == "caller title"
        @test ax.backgroundcolor[] == Makie.to_color(:lavender)
        Makie.update!(graph; upper_recipe=Hist2D)
        @test ax.limits[] == ((0.0, 1.0), (0.0, 1.0))
        @test all(cell(fig, i, j) === axes[i, j] for i in 1:2, j in 1:2)
        # Configuration and selection changes invalidate the envelope; samples
        # and empty windows above must not. Compare with actual new primitives.
        Makie.update!(graph; triagonal_config=merge(PAIR, (nbins=(1, 3),)))
        p = heatmap(ax)
        @test encloses(graph[:axis_limits_2][], p[1][])
        @test encloses(graph[:axis_limits_1][], p[2][])
        Makie.update!(graph; triagonal_config=PAIR)
        @test all(l -> l[1] ≈ 0.06 && l[2] ≈ 0.94, ax.limits[])
        Makie.update!(graph; triagonal_config=merge(PAIR, (nbins=(2, 2),)))
        @test ax.limits[] == ((0.0, 1.0), (0.0, 1.0))
        EXT._apply_vsel_to_graph!(graph, 2, 2, Int[])
        @test graph[:axis_limits][] == [(0.0, 1.0), (0.0, 1.0)]
        EXT._apply_vsel_to_graph!(graph, 2, 2, [2, 1])
        @test length(filter(x -> x isa Axis, fig.content)) == 4
        @test !isempty(Makie.colorbuffer(fig))
    finally
        empty!(fig)
    end

    # Public defaults are a separate control, not evidence that users can
    # configure coarse bins through the public static API.
    for ordinary in (false, true)
        public_fig = ordinary ? Figure() : bat_makie_plot(samples,
            (upper=Hist2D, diagonal=QuantileHist1D, lower=Hist2D), [1, 2], 2; display=false)
        try
            ordinary && Makie.plot(public_fig[1, 1], samples;
                recipes=(upper=Hist2D, diagonal=QuantileHist1D, lower=Hist2D), vsel=[1, 2], N_max=2)
            ax = cell(public_fig, 1, 2)
            p = heatmap(ax)
            @test ax.limits[][1] == (0.06, 0.9400000000000001)
            @test encloses(ax.limits[][1], p[1][])
            @test encloses(ax.limits[][2], p[2][])
            @test !isempty(Makie.colorbuffer(public_fig))
        finally
            empty!(public_fig)
        end
end
end
end # module
