# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestMakieHistogramRegions

using BAT, Test, StatsBase, Makie, CairoMakie, Distributions, Random

const EXT = Base.get_extension(BAT, :BATMakieExt)
const DIAG = merge(BAT._default_makie_diagonal_config(), (nbins=4,))
const PAIR = merge(BAT._default_makie_triagonal_config(), (nbins=(4, 3),))
const QUANTILES = (upper=QuantileHist2D, diagonal=QuantileHist1D, lower=QuantileHist2D)
const NUMERIC = (upper=Hist2D, diagonal=QuantileHist1D, lower=Hist2D)
const RED = EXT._quantile_level_color(1)
const YELLOW = EXT._quantile_level_color(2)
const GREEN = EXT._quantile_level_color(3)

hist1d(edges, weights; density=true) = Histogram((copy(edges),), copy(weights), :left, density)
samples(data; weights=ones(size(data, 2))) = DensitySampleVector(
    [collect(v) for v in eachcol(data)], zeros(size(data, 2)); weight=copy(weights))
q1(hist, levels) = EXT._quantilehist1d_output(hist, merge(DIAG, (normalization=:none, levels)))
q2(hist, levels) = EXT._quantilehist2d_output(hist, merge(PAIR, (normalization=:none, levels)))
cell(fig, i, j) = only(filter(x -> x isa Axis, contents(fig[1, 1][i, j])))

function descendants(axis, T)
    found = Any[]
    function visit(plot)
        plot isa T && push!(found, plot)
        foreach(visit, plot.plots)
    end
    foreach(visit, axis.scene.plots)
    return found
end

function converted_heat(fig, i, j)
    plot = only(descendants(cell(fig, i, j), Makie.Heatmap))
    return (x=copy(plot[1][]), y=copy(plot[2][]), z=copy(plot[3][]))
end

function check_heat_geometry(upper, lower)
    @test length(upper.x) == size(upper.z, 1) + 1
    @test length(upper.y) == size(upper.z, 2) + 1
    @test all(>(0), diff(upper.x)) && all(>(0), diff(upper.y))
    @test upper.x == lower.y && upper.y == lower.x
    @test isequal(upper.z, permutedims(lower.z))
end

function check_stair_geometry(plot, edges, heights)
    @test plot.step[] === :post
    path = plot.steppoints[]
    @test length(path) == 2 * length(heights) + 1
    # Independent cell interpretation: each height spans its original complete
    # interval, including zero gaps and the last bin; connecting jumps are vertical.
    expected = [((edges[i], heights[i]), (edges[i+1], heights[i])) for i in eachindex(heights)]
    actual = [(Tuple(path[i]), Tuple(path[i+1])) for i in 1:2:(length(path)-1)]
    @test actual == expected
    @test Tuple(last(path)) == (last(edges), last(heights))
end

@testset "Histogram regions use cell mass and original geometry" begin
    @testset "Independent unequal-bin mass, plateau and color oracles" begin
        hist = hist1d([0.0, 2.0, 6.0, 8.0], [0.0, 0.15, 0.2])
        before = deepcopy(hist)
        result = q1(hist, [0.5])
        # Densities .2/.15 have masses .4/.6: both bins are needed for .5.
        @test [extrema(first.(poly)) for poly in result.polys] == [(2.0, 6.0), (6.0, 8.0)]
        @test result.fill_colors == [RED, RED]
        @test sum((maximum(first.(poly))-minimum(first.(poly))) * maximum(last.(poly))
            for poly in result.polys) ≈ 1.0
        @test first.(result.stairs_data) == hist.edges[1]
        @test last.(result.stairs_data) == [0.0, 0.15, 0.2, 0.2]
        @test EXT._diag_y_extent(result, QuantileHist1D()) == 0.2
        nested = q1(hist, [0.3, 1.0])
        @test nested.fill_colors == [RED, YELLOW]

        tied = hist1d([0.0, 1.0, 2.0, 3.0], [2.0, 2.0, 1.0])
        plateau = q1(tied, [0.3])
        @test [extrema(first.(poly)) for poly in plateau.polys] == [(0.0, 1.0), (1.0, 2.0)]
        duplicate = q1(tied, [0.3, 0.0, 1.0, 0.3])
        @test duplicate.fill_colors == [GREEN, GREEN, RED]
        @test isempty(q1(tied, [0.0]).polys)
        @test length(q1(tied, [1.0]).polys) == 3
        @test isempty(q1(tied, Float64[]).polys)
        @test hist.weights == before.weights && hist.edges == before.edges
        @test result.polys !== nested.polys
        @test result.polys[1] !== nested.polys[1]
        @test result.stairs_data !== nested.stairs_data

        # Actual areas [2 1; 6 3], density ranks 4,3,2,1 and masses 8,9,12,1.
        edges = ([0.0, 1.0, 4.0], [0.0, 2.0, 3.0])
        counts = [8.0 1.0; 12.0 9.0]
        expected_colors = [GREEN RED; YELLOW YELLOW]
        for mode in (:none, :probability, :density, :pdf)
            raw = Histogram(deepcopy(edges), copy(counts), :left, false)
            h = mode === :none ? raw : StatsBase.normalize(raw; mode)
            colored = q2(h, [0.2, 1.0, 0.6])
            @test colored.color_grid == expected_colors
            selected = q2(h, [0.6]).color_grid .!= RGBAf(0, 0, 0, 0)
            @test selected == Bool[1 0; 1 1]
            @test sum(counts[selected]) / sum(counts) == 29/30
            @test colored.xedges == edges[1] && colored.yedges == edges[2]
            @test raw.weights == counts
        end
        # Density-field ties include every cell at the cutoff regardless of index.
        h = Histogram(deepcopy(edges), [2.0 1.0; 2.0 1.0], :left, true)
        @test (q2(h, [0.1]).color_grid .!= RGBAf(0, 0, 0, 0)) == Bool[1 0; 1 0]
        @test all(c -> iszero(c.alpha), q2(h, [0.0]).color_grid)
        @test all(c -> c == RED, q2(h, [1.0]).color_grid)
        left, right = q2(h, [1.0]), q2(h, [1.0])
        @test left.xedges !== right.xedges && left.yedges !== right.yedges
        @test left.color_grid !== right.color_grid
    end

    @testset "Level validation precedes active empty and dead-cell returns" begin
        invalid = ([-0.1], [1.1], [NaN], [Inf], [0.1, 0.2, 0.3, 1.0],
            [0.5, 0.5, 0.5, 0.5], [big"1e-400"], ["not a probability"])
        for (recipe, dims, config) in ((QuantileHist1D(), 1, DIAG), (QuantileHist2D(), 2, PAIR))
            for levels in invalid, n in (0, 2), state in (EXT.LiveCell(), EXT.DeadCell())
                coordinates, weights = zeros(dims, n), zeros(n)
                cfg = merge(config, (; levels))
                @test_throws ArgumentError EXT.compute_plotting_primitives(view(coordinates, :, :),
                    view(weights, :), recipe, EXT.LiveRecipe(), state, cfg)
                # An unused absolute/region recipe must not poison another recipe.
                @test EXT.compute_plotting_primitives(view(coordinates, :, :), view(weights, :),
                    recipe, EXT.DeadRecipe(), state, cfg) isa NamedTuple
            end
            for levels in (Float64[], [0.0, 0.0], [0.0, 0.3, 0.6, 1.0], Float32[0, 1])
                @test EXT.compute_plotting_primitives(view(zeros(dims, 0), :, :), view(Float64[], :),
                    recipe, EXT.LiveRecipe(), EXT.LiveCell(), merge(config, (; levels))) isa NamedTuple
            end
        end
    end

    @testset "Real heatmap conversion retains unequal and single-cell edges" begin
        for edges in (([0.0, 2.0, 6.0, 8.0], [-2.0, -1.0, 3.0]),
                ([1e-7, 3e-7], [5.0, 5.1]))
            weights = reshape(Float64.(1:((length(edges[1])-1)*(length(edges[2])-1))),
                length(edges[1])-1, length(edges[2])-1)
            length(weights) > 1 && (weights[1] = 0)
            hist = Histogram(deepcopy(edges), weights, :left, true)
            before = deepcopy(hist)
            for recipe in (Hist2D(), QuantileHist2D()), transposed in (false, true)
                prim = recipe isa Hist2D ? EXT._hist2d_output(hist, :none) : q2(hist, [1.0])
                spec = only(EXT.compose_plotspecs(prim, recipe, PAIR; transposed))
                fig = Figure()
                try
                    plot = heatmap!(Axis(fig[1, 1]), spec.args...; spec.kwargs...)
                    @test plot[1][] == edges[transposed ? 2 : 1]
                    @test plot[2][] == edges[transposed ? 1 : 2]
                    values = recipe isa Hist2D ? prim.weights : prim.color_grid
                    @test isequal(plot[3][], transposed ? permutedims(values) : values)
                    if length(weights) > 1
                        @test (recipe isa Hist2D ? isnan(plot[3][][1]) : iszero(plot[3][][1].alpha))
                    end
                    @test !isempty(Makie.colorbuffer(fig))
                finally
                    empty!(fig)
                end
            end
            @test hist.weights == before.weights && hist.edges == before.edges
        end
        for height in (1e50, 1e-50)
            hist = Histogram(([0.0, 1.0], [0.0, 1.0]), fill(height, 1, 1), :left, true)
            @test_throws "Float32 color values" EXT.compose_plotspecs(
                EXT._hist2d_output(hist, :none), Hist2D(), PAIR)
            @test length(EXT.compose_plotspecs(q2(hist, [1.0]), QuantileHist2D(), PAIR)) == 1
        end
        # Finite endpoints whose conversion collapses are not valid rendered cells.
        bad = Histogram(([big"1.0", big"1.0"+big"1e-30"], [0.0, 1.0]), ones(1, 1), :left, true)
        @test_throws ArgumentError EXT._hist2d_output(bad, :none)
        @test_throws ArgumentError q2(bad, [1.0])
    end

    @testset "Polygon mesh endpoints, zero gaps and odd-ULP cells" begin
        for (edges, heights) in (([0.0, 2.0, 6.0, 8.0], [1.0, 0.0, 2.0]),
                ([0.0, 2.0, 6.0, 8.0], [0.0, 0.0, 2.0]),
                ([1e16-2, 1e16+4], [2.0]), ([2.0, 3.0], [1e50]))
            hist = hist1d(edges, heights)
            prim = q1(hist, [1.0])
            selected = findall(>(0), heights)
            @test length(prim.polys) == length(selected)
            @test first.(prim.stairs_data) == edges
            @test last(last(prim.stairs_data)) == last(heights)
            specs = EXT.compose_plotspecs(prim, QuantileHist1D(), DIAG)
            fig = Figure()
            try
                axis = Axis(fig[1, 1])
                polygon = poly!(axis, specs[1].args...; specs[1].kwargs...)
                outline = stairs!(axis, specs[2].args...; specs[2].kwargs...)
                check_stair_geometry(outline, edges, heights)
                ordinary_prim = EXT._hist1d_output(hist, :none)
                ordinary_spec = last(EXT.compose_plotspecs(ordinary_prim, Hist1D(), DIAG))
                ordinary_outline = stairs!(axis, ordinary_spec.args...; ordinary_spec.kwargs...)
                check_stair_geometry(ordinary_outline, edges, heights)
                @test polygon[1][] == prim.polys
                @test polygon.color[] == prim.fill_colors
                meshes = polygon.attributes[:meshes][]
                @test length(meshes) == length(selected)
                for (mesh, index) in zip(meshes, selected)
                    points = Makie.GeometryBasics.coordinates(mesh)
                    @test Set(Tuple.(points)) == Set([(edges[index], 0.0), (edges[index+1], 0.0),
                        (edges[index+1], heights[index]), (edges[index], heights[index])])
                    @test all(p -> eltype(p) == Float64, points)
                    @test !isempty(Makie.GeometryBasics.faces(mesh))
                end
                @test !isempty(Makie.colorbuffer(fig))
            finally
                empty!(fig)
            end
        end
        zero = q1(hist1d([0.0, 1.0, 3.0], zeros(2)), [1.0])
        @test isempty(zero.polys) && isempty(zero.fill_colors)
        @test length(zero.stairs_data) == 3
        @test length(EXT.compose_plotspecs(zero, QuantileHist1D(), DIAG)) == 1
        # No polygon shares another cell's writable point storage.
        two = q1(hist1d([0.0, 1.0, 2.0], ones(2)), [1.0])
        @test two.polys[1] !== two.polys[2]
    end

    @testset "Public ordinary and interactive geometry" begin
        source = samples([3.0 3.0; -2.0 -2.0])
        before = deepcopy(source)
        for recipes in (NUMERIC, QUANTILES), ordinary in (false, true)
            fig = ordinary ? Figure() : bat_makie_plot(source, recipes, [1, 2], 2; display=false)
            try
                ordinary && Makie.plot(fig[1, 1], source; recipes, vsel=[1, 2], N_max=2)
                upper, lower = converted_heat(fig, 1, 2), converted_heat(fig, 2, 1)
                check_heat_geometry(upper, lower)
                @test upper.x ≈ [-2.1, -1.9]
                @test upper.y ≈ [2.85, 3.15]
                polygon = only(descendants(cell(fig, 1, 1), Makie.Poly))
                @test collect(extrema(first.(only(polygon[1][])))) ≈ [2.85, 3.15]
                @test !isempty(Makie.colorbuffer(fig))
            finally
                empty!(fig)
            end
        end
        @test source == before
    end

    @testset "Public histogram outlines retain cell heights under caller themes" begin
        values, weights = [0.1, 0.1, 0.5, 0.9], [1.0, 2.0, 1.0, 4.0]
        source = samples(reshape(values, 1, :); weights)
        before = deepcopy(source)
        # Ordinary composition inherits the caller's theme. Bin interpretation
        # must therefore be explicit in the recipe, not supplied by BAT's theme.
        with_theme(Theme(Stairs=(step=:pre,))) do
            for diagonal in (Hist1D, QuantileHist1D), ordinary in (false, true)
                recipes = (upper=Hist2D, diagonal=diagonal, lower=Hist2D)
                fig = ordinary ? Figure() : bat_makie_plot(source, recipes, [1], 1; display=false)
                try
                    ordinary && Makie.plot(fig[1, 1], source; recipes, vsel=[1], N_max=1)
                    outline = only(descendants(cell(fig, 1, 1), Makie.Stairs))
                    original_points = outline[1][]
                    edges, heights = first.(original_points), last.(original_points)[1:end-1]
                    counts = [sum((weights[j] for j in eachindex(values)
                        if edges[i] <= values[j] < edges[i+1]); init=0.0)
                        for i in 1:length(edges)-1]
                    @test sum(counts) == sum(weights)
                    expected_heights = counts ./ (sum(weights) .* diff(edges))
                    @test count(>(0), counts) == 3
                    @test length(unique(filter(>(0), expected_heights))) > 1
                    @test heights ≈ expected_heights
                    check_stair_geometry(outline, edges, heights)
                    @test !isempty(Makie.colorbuffer(fig))
                finally
                    empty!(fig)
                end
            end
        end
        @test source == before
    end

    @testset "Graph empty, positive, zero, mode, window and selection transitions" begin
        source = samples([0.1 0.3 0.9; 0.8 0.2 0.5])
        before = deepcopy(source)
        graph, _ = EXT._setup_static_graph(source[1:0], QUANTILES, [1, 2], 2, PAIR, DIAG)
        diagonal_node = EXT.primitive_symbol(QuantileHist1D(), (1, 1))
        pair_node = EXT.primitive_symbol(QuantileHist2D(), (2, 1))
        empty_diagonal, empty_pair = graph[diagonal_node][], graph[pair_node][]
        fig = Figure()
        try
            EXT._bind_interactive_grid!(fig, EXT._init_gridlayout!(graph, 2))
            @test isempty(descendants(cell(fig, 1, 1), Makie.Poly))
            Makie.update!(graph; samples=[[source]], current_idxs=[[length(source)]])
            first_pair = graph[pair_node][]
            snapshot = deepcopy(first_pair)
            for mode in (:none, :probability, :density, :pdf)
                Makie.update!(graph; diagonal_config=merge(DIAG, (normalization=mode, levels=[0.0, 0.5, 1.0],)),
                    triagonal_config=merge(PAIR, (normalization=mode, levels=[0.0, 0.5, 1.0],)))
                @test typeof(graph[diagonal_node][]) == typeof(empty_diagonal)
                @test typeof(graph[pair_node][]) == typeof(empty_pair)
                check_heat_geometry(converted_heat(fig, 1, 2), converted_heat(fig, 2, 1))
            end
            @test isequal(first_pair, snapshot)
            @test graph[pair_node][].xedges !== first_pair.xedges
            zero = samples([0.1 0.3 0.9; 0.8 0.2 0.5]; weights=zeros(3))
            Makie.update!(graph; samples=[[zero]], current_idxs=[[length(zero)]])
            @test isempty(graph[diagonal_node][].polys)
            @test all(c -> iszero(c.alpha), graph[pair_node][].color_grid)
            Makie.update!(graph; samples=[[source]], current_idxs=[[length(source)]])
            EXT._apply_vsel_to_graph!(graph, 2, 2, [2, 1])
            check_heat_geometry(converted_heat(fig, 1, 2), converted_heat(fig, 2, 1))
            for window in ((2, 3), (100, 101), nothing)
                Makie.update!(graph; window_steps=window)
                @test typeof(graph[diagonal_node][]) == typeof(empty_diagonal)
                @test typeof(graph[pair_node][]) == typeof(empty_pair)
                if window == (100, 101)
                    @test isempty(descendants(cell(fig, 1, 1), Makie.Poly))
                    @test isempty(descendants(cell(fig, 1, 2), Makie.Heatmap))
                else
                    check_heat_geometry(converted_heat(fig, 1, 2), converted_heat(fig, 2, 1))
                end
            end
            @test !isempty(Makie.colorbuffer(fig))
            @test source == before
        finally
            empty!(fig)
        end
    end

    @testset "Completed public live run uses the same region geometry" begin
        algorithm = TransformedMCMC(nchains=2, nsteps=32, strict=false,
            init=MCMCChainPoolInit(nsteps_init=20), convergence=AssumeConvergence(),
            adaptive_transform=BAT.NoAdaptiveTransform(), transform_tuning=NoMCMCTransformTuning(),
            burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=12, max_ncycles=1, nsteps_final=3))
        vis = BATVisualizer(BATMakieVisualization(QUANTILES, [1, 2], 2,
            8, 10^6, false, 1.2, 0.01, false, PAIR, DIAG))
        old_suppress = EXT._SUPPRESS_DISPLAY[]
        EXT._SUPPRESS_DISPLAY[] = true
        try
            result = bat_sample(MvNormal(zeros(2), ones(2)), algorithm,
                BATContext(rng=BAT.Philox4x((165, 2026)), visualizer=vis))
            @test !isempty(result.result)
            status = BAT.visualizer_status(vis)
            @test status.quiesced && status.data_complete
            @test isnothing(status.error)
            fig = BAT.visualizer_figure(vis)
            @test !isnothing(fig)
            if !isnothing(fig)
                check_heat_geometry(converted_heat(fig, 1, 2), converted_heat(fig, 2, 1))
                @test !isempty(descendants(cell(fig, 1, 1), Makie.Poly))
                @test !isempty(Makie.colorbuffer(fig))
            end
        finally
            try
                BAT.request_visualizer_stop!(vis; reason=:disabled)
                BAT.finish_visualizer!(vis; success=false)
                BAT.dispose_visualizer!(vis)
                task = lock(() -> vis.content.lifecycle.disposal_task, vis.content.buffer_lock)
                @test task isa Task
                if task isa Task
                    ready = timedwait(() -> istaskdone(task), 30.0; pollint=0.005) === :ok
                    @test ready
                    ready && wait(task)
                end
            finally
                EXT._SUPPRESS_DISPLAY[] = old_suppress
            end
        end
    end
end
end
