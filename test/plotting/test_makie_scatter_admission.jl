# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestMakieScatterAdmission
using BAT, Makie, Test
const EXT = Base.get_extension(BAT, :BATMakieExt)
const CONFIG = BAT._default_makie_triagonal_config()
const XY = [0.137 0.473 0.819; 0.211 0.397 0.733]
const IDS = Int32[1, 1, 2]
const INFO = [BAT.MCMCSampleID(IDS[i], Int32(1), Int32(1), Int64(i), Int32(1), true) for i in 1:3]

function primitive(weights, recipe)
    args = (view(XY, :, :), view(weights, :))
    recipe isa ChainScatter2D && (args = (args..., IDS))
    return EXT.compute_plotting_primitives(args..., recipe, EXT.LiveRecipe(), EXT.LiveCell(), CONFIG)
end

function expected_sizes(weights)
    # Independent arithmetic on the original values, before Float64 narrowing.
    wide = [w isa BAT.ULogarithmic ? exp(BigFloat(log(w))) : BigFloat(w) for w in weights]
    return Float64.(CONFIG.markersize .* sqrt.(length(wide) .* wide ./ sum(wide)))
end

@testset "Scatter relative weight admission" begin
    families = ([1., 2., 3.], big"1e400" .* [1, 2, 3],
        big"1e-400" .* [1, 2, 3], big(10)^400 .* [1, 2, 3],
        exp.(BAT.ULogarithmic, 1000 .+ log.([1., 2., 3.])))
    for weights in families, recipe in (Scatter2D(), ChainScatter2D())
        before = copy(weights)
        p = primitive(weights, recipe)
        specs = EXT.compose_plotspecs(p, recipe, CONFIG)
        actual = reduce(vcat, [s.kwargs[:markersize] for s in specs])
        @test actual ≈ expected_sizes(weights) rtol=1e-12
        @test weights == before
    end
    for weights in families, recipe in (Scatter2D, ChainScatter2D), ordinary in (false, true)
        samples = DensitySampleVector(v=[collect(v) for v in eachcol(XY)],
            logd=zeros(3), weight=weights, info=INFO)
        recipes = (diagonal=QuantileHist1D, upper=recipe, lower=recipe)
        fig = ordinary ? Figure() : bat_makie_plot(samples, recipes, [1, 2], 2; display=false)
        try
            ordinary && Makie.plot(fig[1, 1], samples; recipes, vsel=[1, 2], N_max=2)
            axes = filter(x -> x isa Axis, fig.content)
            pairs = filter(ax -> any(p -> p isa Makie.Scatter, ax.scene.plots), axes)
            @test length(pairs) == 2
            for ax in pairs
                scatter = filter(p -> p isa Makie.Scatter, ax.scene.plots)
                actual = reduce(vcat, [first.(p.markersize[]) for p in scatter])
                @test actual ≈ expected_sizes(weights) rtol=2e-6
            end
        finally
            empty!(fig)
        end
    end
    for recipe in (Scatter2D(), ChainScatter2D())
        @test_throws ArgumentError primitive(BigFloat[1, big"1e-400", 2], recipe)
        for weights in (zeros(3), ones(3), Float16[1,2,3], Float32[1,2,3], [0.,2.,3.])
            specs = EXT.compose_plotspecs(primitive(weights, recipe), recipe, CONFIG)
            sizes = reduce(vcat, [s.kwargs[:markersize] isa Number ?
                fill(s.kwargs[:markersize], length(s.args[1])) : s.kwargs[:markersize] for s in specs])
            expected = all(iszero, weights) ? zeros(3) : expected_sizes(weights)
            @test sizes ≈ expected rtol=1e-12
        end
    end
    samples = DensitySampleVector(v=[collect(v) for v in eachcol(XY)], logd=zeros(3),
        weight=big"1e400" .* [1,2,3], info=INFO)
    recipes = (diagonal=Hist1D, upper=Scatter2D, lower=ChainScatter2D)
    graph, _ = EXT._setup_static_graph(samples, recipes, [1,2], 2, CONFIG,
        BAT._default_makie_diagonal_config())
    layout = EXT._init_gridlayout!(graph, 2)
    initial = graph[EXT.primitive_symbol(Scatter2D(), (2,1))][]
    Makie.update!(graph; window_steps=(Int64(10),Int64(12)))
    layout[]
    @test isempty(graph[EXT.primitive_symbol(Scatter2D(), (2,1))][].x)
    Makie.update!(graph; window_steps=nothing)
    layout[]
    @test graph[EXT.primitive_symbol(Scatter2D(), (2,1))][] == initial
    Makie.update!(graph; idxs=[1])
    layout[]
    @test isempty(graph[EXT.primitive_symbol(Scatter2D(), (2,1))][].x)
    Makie.update!(graph; idxs=[1,2])
    layout[]
    @test graph[EXT.primitive_symbol(Scatter2D(), (2,1))][] == initial
end

function check_scatter_sizes(fig, weights)
    axes = filter(x -> x isa Axis && any(p -> p isa Makie.Scatter, x.scene.plots), fig.content)
    @test length(axes) == 2
    expected = all(iszero, weights) ? zeros(length(weights)) : expected_sizes(weights)
    for axis in axes
        plots = filter(p -> p isa Makie.Scatter, axis.scene.plots)
        @test sum(p -> length(p[1][]), plots) == length(weights)
        sizes = reduce(vcat, [p.markersize[] isa Makie.Vec2f ?
            fill(first(p.markersize[]), length(p[1][])) : first.(p.markersize[]) for p in plots])
        @test sizes ≈ expected rtol=2e-6
    end
end

@testset "Zero-mass scatter populations stay invisible" begin
    coords = [collect(v) for v in eachcol(hcat(XY, XY .+ 1))]
    info = [BAT.MCMCSampleID(Int32(chain), Int32(1), Int32(1), Int64(step), Int32(1), true)
        for chain in 1:2 for step in 1:3]
    for recipe in (Scatter2D, ChainScatter2D), ordinary in (false, true),
            weights in (zeros(6), repeat([1.0, 0.0, 0.0], 2), ones(6))
        samples = DensitySampleVector(v=coords, logd=zeros(6), weight=copy(weights), info=info)
        before = deepcopy(samples)
        recipes = (diagonal=Hist1D, upper=recipe, lower=recipe)
        fig = ordinary ? Figure() : bat_makie_plot(samples, recipes, [1, 2], 2; display=false)
        try
            ordinary && Makie.plot(fig[1, 1], samples; recipes, vsel=[1, 2], N_max=2)
            check_scatter_sizes(fig, weights)
            if !ordinary
                slider = only(filter(x -> x isa IntervalSlider, fig.content))
                apply = only(filter(x -> x isa Button && x.label[] == "Apply", fig.content))
                all_button = only(filter(x -> x isa Button && x.label[] == "All", fig.content))
                slider.interval[] = (2, 3)
                check_scatter_sizes(fig, weights)
                notify(apply.clicks)
                check_scatter_sizes(fig, weights[[2, 3, 5, 6]])
                notify(all_button.clicks)
                check_scatter_sizes(fig, weights)
            end
            @test samples == before
        finally
            empty!(fig)
        end
    end
end

@testset "Chain scatter requires chain identity" begin
    samples = DensitySampleVector(v=[collect(v) for v in eachcol(XY)], logd=zeros(3))
    recipes = (diagonal=Hist1D, upper=ChainScatter2D, lower=Hist2D)
    @test_throws ArgumentError EXT.compute_plotting_primitives(view(XY,:,:),
        view(ones(3),:), Int32[], ChainScatter2D(), EXT.LiveRecipe(), EXT.LiveCell(), CONFIG)
    error = try
        Makie.convert_arguments(Makie.Plot{Makie.plot}, samples;
            recipes, vsel=[1,2], N_max=2)
        nothing
    catch err
        err
    end
    @test !isnothing(error)
    @test occursin("ChainScatter2D requires one chain identity per sample", sprint(showerror, error))
    # An empty population remains an empty plot even without chain metadata.
    p = EXT.compute_plotting_primitives(view(zeros(2,0),:,:), view(Float64[],:),
        Int32[], ChainScatter2D(), EXT.LiveRecipe(), EXT.LiveCell(), CONFIG)
    @test isempty(EXT.compose_plotspecs(p, ChainScatter2D(), CONFIG))
end
end
