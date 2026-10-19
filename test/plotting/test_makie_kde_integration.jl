# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestMakieKDEIntegration

using BAT, Test, Makie, CairoMakie, Distributions, Random, ValueShapes

const EXT = Base.get_extension(BAT, :BATMakieExt)
const SUPPORT = [(-2.0, 4.0), (10.0, 18.0)]
const DATA = [-1.7 -0.7 0.2 1.3 2.0 3.5; 11.0 15.0 12.0 17.0 13.0 16.0]
const RECIPES = (upper=KDE2D, diagonal=KDE1D, lower=KDE2D)
const QUANTILES = (upper=QuantileKDE2D, diagonal=QuantileKDE1D, lower=QuantileKDE2D)

samples(data=DATA; elastic=false) = DensitySampleVector(
    v=EXT.VectorOfSimilarVectors(elastic ? EXT.ElasticArray(data) : copy(data)),
    logd=zeros(size(data, 2)), weight=ones(size(data, 2)))
cell(fig, i, j) = only(filter(x -> x isa Axis, contents(fig[1, 1][i, j])))

function descendants(axis, T)
    result = Any[]
    function visit(plot)
        plot isa T && push!(result, plot)
        foreach(visit, plot.plots)
    end
    foreach(visit, axis.scene.plots)
    return result
end

function heat(axis)
    plot = only(descendants(axis, Makie.Heatmap))
    return (x=copy(plot[1][]), y=copy(plot[2][]), z=copy(plot[3][]))
end

function check_heatmaps(fig, support=SUPPORT, selection=[1, 2])
    upper, lower = heat(cell(fig, 1, 2)), heat(cell(fig, 2, 1))
    # Check the converted renderer inputs: center-only PlotSpecs can appear
    # bounded while Makie extrapolates cells beyond their hard support.
    @test length(upper.x) == size(upper.z, 1) + 1
    @test length(upper.y) == size(upper.z, 2) + 1
    @test all(>(0), diff(upper.x)) && all(>(0), diff(upper.y))
    @test (first(upper.x), last(upper.x)) == support[selection[2]]
    @test (first(upper.y), last(upper.y)) == support[selection[1]]
    @test upper.x == lower.y && upper.y == lower.x
    @test isequal(upper.z, permutedims(lower.z))
    return upper
end

function setup_graph(s; recipes=RECIPES, support=SUPPORT)
    graph, _ = EXT._setup_static_graph(s, recipes, [1, 2], 2,
        BAT._default_makie_triagonal_config(), BAT._default_makie_diagonal_config(); support)
    return graph
end

@testset "Bounded KDE public integration" begin
    @testset "Hard support admission and original measure forwarding" begin
        prior = product_distribution([Uniform(-2, 4), Uniform(10, 18)])
        wrapped = EvaluatedMeasure(prior; empirical=samples())
        @test EXT._support_bounds(prior, 2) == ([-2.0, 10.0], [4.0, 18.0])
        @test EXT._support_bounds(wrapped, 2) == EXT._support_bounds(prior, 2)
        @test EXT._support_vectors(wrapped, 2) == EXT._support_vectors(SUPPORT, 2)
        @test EXT._support_bounds(nothing, 2) == (fill(-Inf, 2), fill(Inf, 2))
        @test EXT._support_vectors([(-Inf, Inf), (0.0, Inf)], 2) == ([-Inf, 0.0], [Inf, Inf])
        @test_throws ArgumentError EXT._support_vectors([(-1.0, 1.0)], 2)
        @test_throws ArgumentError EXT._support_vectors([nothing], 1)
        invalid = ((0.0, 0.0), (1.0, 0.0), (NaN, 1.0), (0.0, NaN),
            (Inf, Inf), (-Inf, -Inf), (0.0, 1.0, 2.0),
            (-big"1e400", 1.0), (0.0, big"1e400"))
        for entry in invalid
            @test_throws ArgumentError EXT._support_vectors([entry], 1)
        end
        # Validate known distribution endpoints before Float64 conversion can
        # turn a finite hard bound into an apparently unbounded coordinate.
        wide_prior = Uniform(-big"1e400", big"1e400")
        @test_throws ArgumentError EXT._support_vectors(wide_prior, 1)
        @test_throws ArgumentError EXT._support_vectors(product_distribution([wide_prior]), 1)
        bad = samples(reshape([1.1], 1, 1))
        @test_throws "inside hard support" bat_makie_plot(bad, RECIPES, [1], 1;
            support=[(0.0, 1.0)], display=false)
        @test_throws "inside hard support" Makie.convert_arguments(Makie.Plot{Makie.plot}, bad;
            recipes=RECIPES, vsel=[1], N_max=1, support=[(0.0, 1.0)])
    end

    @testset "Actual heatmap edges and both triangle orientations" begin
        source = samples()
        before = deepcopy(source)
        prior = EvaluatedMeasure(product_distribution([Uniform(-2, 4), Uniform(10, 18)]))
        for recipes in (RECIPES, QUANTILES), ordinary in (false, true)
            fig = ordinary ? Figure() : bat_makie_plot(source, recipes, [2, 1], 2;
                support=prior, display=false)
            try
                ordinary && Makie.plot(fig[1, 1], source;
                    recipes, vsel=[2, 1], N_max=2, support=prior)
                check_heatmaps(fig, SUPPORT, [2, 1])
                @test !isempty(Makie.colorbuffer(fig))
            finally
                empty!(fig)
            end
        end
        @test source == before

        # A single center must not become Makie's default unit-width cell.
        tiny_support = [(-1e-7, 2e-7), (10.0 - 3e-7, 10.0 + 4e-7)]
        constant = samples([0.0 0.0; 10.0 10.0])
        for recipes in (RECIPES, QUANTILES)
            fig = bat_makie_plot(constant, recipes, [1, 2], 2;
                support=tiny_support, display=false)
            try
                rendered = check_heatmaps(fig, tiny_support)
                @test size(rendered.z) == (1, 1)
                if recipes === QUANTILES
                    @test only(rendered.z).alpha > 0
                else
                    @test isfinite(only(rendered.z)) && only(rendered.z) > 0
                end
            finally
                empty!(fig)
            end
        end
    end

    @testset "Numeric heatmap colors reject lossy range conversion" begin
        for scale in (1e-30, 1e30)
            data = scale == 1e-30 ? fill(0.5scale, 2, 2) :
                scale .* [0.25 0.75; 0.75 0.25]
            support = [(0.0, scale), (0.0, scale)]
            cfg = merge(BAT._default_makie_triagonal_config(), (support=Tuple(support),))
            weights = ones(2)
            prim = EXT.compute_plotting_primitives(view(data, :, :), view(weights, :),
                KDE2D(), EXT.LiveRecipe(), EXT.LiveCell(), cfg)
            visible = filter(isfinite, vec(prim.density))
            @test !isempty(visible) && all(>(0), visible)
            @test any(d -> !isfinite(Float32(d)) || iszero(Float32(d)), visible)
            @test_throws "Float32 color values" EXT.compose_plotspecs(prim, KDE2D(), cfg)
            @test_throws "Float32 color values" EXT.compose_plotspecs(prim, KDE2D(), cfg; transposed=true)
            @test_throws "Float32 color values" bat_makie_plot(samples(data),
                (diagonal=Hist1D, upper=KDE2D, lower=KDE2D), [1, 2], 2;
                support, display=false)

            # Region colors do not carry physical density magnitudes and remain
            # usable without changing the numerical estimate's density units.
            quantile = EXT.compute_plotting_primitives(view(data, :, :), view(weights, :),
                QuantileKDE2D(), EXT.LiveRecipe(), EXT.LiveCell(), cfg)
            spec = only(EXT.compose_plotspecs(quantile, QuantileKDE2D(), cfg))
            fig = Figure()
            try
                plot = heatmap!(Axis(fig[1, 1]), spec.args...; spec.kwargs...)
                @test all(c -> all(isfinite, (c.r, c.g, c.b, c.alpha)), plot[3][])
                @test any(c -> c.alpha > 0, plot[3][])
                @test first(plot[1][]) == 0 && last(plot[1][]) == scale
                @test first(plot[2][]) == 0 && last(plot[2][]) == scale
                @test !isempty(Makie.colorbuffer(fig))
            finally
                empty!(fig)
            end
        end
    end

    @testset "Empty, support, selection and window transitions preserve ownership" begin
        for elastic in (false, true), recipes in (RECIPES, QUANTILES)
            source = samples(; elastic)
            before = deepcopy(source)
            graph = setup_graph(source[1:0]; recipes)
            grid = EXT._init_gridlayout(graph, 2)
            fig = Figure()
            try
                EXT._bind_interactive_grid!(fig, grid)
                @test isempty(descendants(cell(fig, 1, 2), Makie.Heatmap))
                node = EXT.primitive_symbol(recipes.upper(), (2, 1))
                empty_type = typeof(graph[node][])
                Makie.update!(graph; samples=[[source]], current_idxs=[[length(source)]])
                first_result = graph[node][]
                snapshot = deepcopy(first_result)
                @test typeof(first_result) == empty_type
                bounded = check_heatmaps(fig)
                Makie.update!(graph; support_lo=Float64[], support_hi=Float64[])
                @test typeof(graph[node][]) == empty_type
                unbounded = heat(cell(fig, 1, 2))
                @test first(unbounded.x) < SUPPORT[2][1] || last(unbounded.x) > SUPPORT[2][2]
                Makie.update!(graph; support_lo=first.(SUPPORT), support_hi=last.(SUPPORT))
                @test isequal(check_heatmaps(fig), bounded)
                EXT._apply_vsel_to_graph!(graph, 2, 2, [2, 1])
                check_heatmaps(fig, SUPPORT, [2, 1])
                Makie.update!(graph; window_steps=(2, 4))
                @test graph[:flat_weights][] == source.weight[2:4]
                @test graph[:flat_samples][] == source[2:4]
                check_heatmaps(fig, SUPPORT, [2, 1])
                Makie.update!(graph; window_steps=(100, 101))
                @test typeof(graph[node][]) == empty_type
                @test isempty(descendants(cell(fig, 1, 2), Makie.Heatmap))
                Makie.update!(graph; window_steps=nothing)
                check_heatmaps(fig, SUPPORT, [2, 1])
                @test isequal(first_result, snapshot)
                @test source == before
                @test graph[node][].xedges !== first_result.xedges
                @test graph[node][].yedges !== first_result.yedges
            finally
                empty!(fig)
            end
        end
    end

    @testset "Real bounded live target forwards original support" begin
        prior = NamedTupleDist(a=Uniform(-2, 4), b=Uniform(10, 18))
        algorithm = TransformedMCMC(nchains=2, nsteps=60, strict=false,
            init=MCMCChainPoolInit(nsteps_init=20),
            adaptive_transform=BAT.NoAdaptiveTransform(),
            transform_tuning=NoMCMCTransformTuning(),
            convergence=GelmanRubinConvergence(),
            burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=20, max_ncycles=1, nsteps_final=5))
        vis = BATVisualizer(BATMakieVisualization(RECIPES, [1, 2], 2,
            50, 10^6, false, 1.2, 0.1, false,
            BAT._default_makie_triagonal_config(), BAT._default_makie_diagonal_config()))
        old_suppress = EXT._SUPPRESS_DISPLAY[]
        EXT._SUPPRESS_DISPLAY[] = true
        try
            result = bat_sample(prior, algorithm, BATContext(rng=BAT.Philox4x((564, 84)), visualizer=vis))
            @test !isempty(result.result)
            @test BAT.visualizer_status(vis).quiesced
            @test BAT.visualizer_status(vis).data_complete
            graph = EXT._live_graph(vis)
            @test graph[:support_lo][] == first.(SUPPORT)
            @test graph[:support_hi][] == last.(SUPPORT)
            fig = BAT.visualizer_figure(vis)
            @test !isnothing(fig)
            if !isnothing(fig)
                check_heatmaps(fig)
                EXT._live_graph_update!(vis; diagonal_recipe=QuantileKDE1D,
                    upper_recipe=QuantileKDE2D, lower_recipe=QuantileKDE2D)
                check_heatmaps(fig)
                @test BAT.visualizer_figure(vis) === fig
                @test !isempty(Makie.colorbuffer(fig))
            end
        finally
            BAT.request_visualizer_stop!(vis; reason=:disabled)
            BAT.finish_visualizer!(vis; success=false)
            BAT.dispose_visualizer!(vis)
            EXT._SUPPRESS_DISPLAY[] = old_suppress
        end
    end
end
end
