# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestMakieFailureOwnership
using BAT, Makie, Test
const EXT = Base.get_extension(BAT, :BATMakieExt)
const INFO = [BAT.MCMCSampleID(Int32(1), Int32(1), Int32(1), Int64(i), Int32(1), true) for i in 1:3]
const SAMPLES = DensitySampleVector(v=[[0.2,0.3],[0.5,0.8],[0.9,0.6]], logd=zeros(3), info=INFO)
const RECIPES = (upper=ChainScatter2D, diagonal=Hist1D, lower=Hist2D)
const DISPLAY_ERROR = ErrorException("injected static display failure")
mutable struct FailingDisplay <: AbstractDisplay
    fig::Union{Nothing,Figure}
end
Base.displayable(::FailingDisplay, ::MIME"image/png") = true
Base.display(d::FailingDisplay, fig::Figure) = (d.fig = fig; throw(DISPLAY_ERROR))

@testset "Failed static figures release owned content" begin
    unrelated = Figure()
    ax = Axis(unrelated[1,1])
    scatter!(ax, [0.,1.], [1.,0.])
    try
        graph, _ = EXT._setup_static_graph(SAMPLES, RECIPES, [1,2], 2,
            BAT._default_makie_triagonal_config(), BAT._default_makie_diagonal_config())
        layout = EXT._init_gridlayout(graph,2)
        source = Makie.ComputePipeline.get_observable!(layout; use_deepcopy=false)
        listeners_before = length(Makie.Observables.listeners(source))
        # The data really has chain IDs. This controlled UI mismatch throws
        # after axes and graph subscriptions have been installed.
        @test_throws ArgumentError EXT._build_fig(graph, layout; has_chain_info=false)
        failed = Makie.current_figure()
        @test failed !== unrelated
        @test isempty(failed.content)
        @test isempty(failed.scene.deregister_callbacks)
        @test length(Makie.Observables.listeners(source)) == listeners_before
        @test length(unrelated.content) == 1
        @test length(ax.scene.plots) == 1
        empty!(failed)

        display = FailingDisplay(nothing)
        pushdisplay(display)
        error = try
            bat_makie_plot(SAMPLES, RECIPES, [1,2], 2)
            nothing
        catch err
            err
        finally
            popdisplay(display)
        end
        @test error === DISPLAY_ERROR
        @test display.fig isa Figure
        if display.fig isa Figure
            @test isempty(display.fig.content)
            @test isempty(display.fig.scene.deregister_callbacks)
            empty!(display.fig)
        end
        @test length(ax.scene.plots) == 1
        @test !isempty(Makie.colorbuffer(unrelated))
        repaired = bat_makie_plot(SAMPLES, RECIPES, [1,2], 2; display=false)
        try
            @test length(filter(x -> x isa Axis, repaired.content)) == 4
        finally
            empty!(repaired)
        end
    finally
        empty!(unrelated)
    end
end

@testset "Grid and histogram dimensions accept integer widths" begin
    for T in (Int32, Int64, BigInt)
        pair = (T(1), T(2))
        @test EXT.marg_symbol(pair) == :marg_1_2
        @test EXT.primitive_symbol(Hist2D, pair) == EXT.primitive_symbol(Hist2D, (1,2))
        @test EXT.primitive_symbol(Hist2D(), pair) == EXT.primitive_symbol(Hist2D(), (1,2))
        for coords in ([0.2 0.6 0.8], [0.2 0.6 0.8; 0.3 0.4 0.9])
            bins = size(coords,1) == 1 ? T(3) : (T(3),T(4))
            native = size(coords,1) == 1 ? 3 : (3,4)
            actual = EXT._marginal_view_dist(view(coords,:,:),view(ones(3),:),false,bins,:left,:pdf)
            expected = EXT._marginal_view_dist(view(coords,:,:),view(ones(3),:),false,native,:left,:pdf)
            @test actual.edges == expected.edges
            @test actual.weights == expected.weights
        end
    end
end
end
