# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Run with a native display and an external process deadline.
using BAT, GLMakie, Makie, Distributions, Test, Logging
include("../../plotting/makie_control_testutils.jl")
const EXT = Base.get_extension(BAT, :BATMakieExt)
@test realpath(pkgdir(BAT)) == realpath(joinpath(@__DIR__, "../../.."))
GLMakie.activate!(visible = false)
Makie.inline!(false)

function pointer!(fig, position)
    events(fig).mouseposition[] = Tuple(Float64.(position))
end

function click!(fig, button)
    sleep(0.25) # Keep separate clicks outside Makie's double-click interval.
    box = button.layoutobservables.computedbbox[]
    pointer!(fig, box.origin + box.widths / 2)
    events(fig).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.press)
    events(fig).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.release)
end

function sliderpoint(slider, fraction)
    box = slider.layoutobservables.computedbbox[]
    padding = box.widths[2] / 2
    return box.origin + Vec2f(padding + fraction * (box.widths[1] - 2padding), box.widths[2] / 2)
end

function drag!(fig, slider, from, to)
    pointer!(fig, (-10, -10))
    # Enter, then hover so Makie selects the handle or interval before pressing.
    pointer!(fig, sliderpoint(slider, from) + Vec2f(0, 1))
    pointer!(fig, sliderpoint(slider, from))
    events(fig).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.press)
    pointer!(fig, sliderpoint(slider, (from + to) / 2))
    pointer!(fig, sliderpoint(slider, to))
end

release!(fig) = (events(fig).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.release))

function check_controls(fig, graph)
    slider = only(filter(x -> x isa IntervalSlider, fig.content))
    live_drag = only(filter(x -> x isa Button && x.label[] == "Live drag", fig.content))
    @test !any(x -> x isa Button && x.label[] == "All", fig.content)
    @test !isempty(Makie.colorbuffer(fig))
    initial = graph[:window_steps][]

    drag!(fig, slider, 0.0, 0.2)
    @test first(slider.interval[]) > 1
    @test graph[:window_steps][] == initial
    # Let the on-demand renderer settle while held. Release must finish even
    # without another motion event or a test-injected render tick.
    sleep(0.3)
    @test graph[:window_steps][] == initial
    release!(fig)
    @test timedwait(() -> graph[:window_steps][] == slider.interval[], 5; pollint=0.02) == :ok

    click!(fig, live_drag)
    @test live_drag.label[] == "Live drag"
    n = last(slider.range[])
    range_text!(fig, (round(Int, 0.2n), round(Int, 0.6n)); submit=true)
    before = slider.interval[]
    drag!(fig, slider, (first(before) - 1) / (n - 1), 0.3)
    @test timedwait(() -> graph[:window_steps][] == slider.interval[], 5; pollint = 0.02) == :ok
    @test first(graph[:window_steps][]) > first(before)
    release!(fig)
    @test timedwait(() -> graph[:window_steps][] == slider.interval[], 5; pollint=0.02) == :ok

    before = slider.interval[]
    drag!(fig, slider, (last(before) - 1) / (n - 1), 0.7)
    @test timedwait(() -> graph[:window_steps][] == slider.interval[], 5; pollint = 0.02) == :ok
    @test last(graph[:window_steps][]) > last(before)
    @test first(graph[:window_steps][]) == first(before)
    release!(fig)
    @test timedwait(() -> graph[:window_steps][] == slider.interval[], 5; pollint=0.02) == :ok

    before = slider.interval[]
    middle = (sum(before) / 2 - 1) / (n - 1)
    drag!(fig, slider, middle, middle + 0.1)
    @test timedwait(() -> graph[:window_steps][] == slider.interval[], 5; pollint = 0.02) == :ok
    @test first(graph[:window_steps][]) > first(before)
    @test last(graph[:window_steps][]) - first(graph[:window_steps][]) == last(before) - first(before)
    release!(fig)
    @test timedwait(() -> graph[:window_steps][] == slider.interval[], 5; pollint=0.02) == :ok
    if haskey(ENV, "BAT_LIVE_DRAG_SCREENSHOT")
        save(ENV["BAT_LIVE_DRAG_SCREENSHOT"], fig)
    end

    click!(fig, live_drag)
    @test live_drag.label[] == "Live drag"
    applied = graph[:window_steps][]
    range_text!(fig, (2,3))
    @test graph[:window_steps][] == applied
    range_key!(fig)
    @test graph[:window_steps][] == (2, 3)
end

@test_logs min_level=Logging.Warn @testset "Native release application and live dragging" begin
    samples = DensitySampleVector(v=[[Float64(i), sin(i)] for i in 1:100], logd=zeros(100))
    recipes = (upper=Hist2D, diagonal=Hist1D, lower=Hist2D)
    graph = first(EXT._setup_static_graph(samples, recipes, [1,2], 2,
        BAT._default_makie_triagonal_config(), BAT._default_makie_diagonal_config()))
    built = with_theme(bat_theme()) do
        EXT._build_fig(graph, EXT._init_gridlayout!(graph, 2))
    end
    screen = GLMakie.Screen(; visible=false)
    try
        display(screen, built.fig)
        check_controls(built.fig, graph)
    finally
        close(screen)
        empty!(built.fig)
    end

    config = BATMakieVisualization()
    vis = BATVisualizer(BATMakieVisualization(recipes, [1,2], 2, 50, 100_000, false,
        config.batch_growth_rate, 0.01, false, config.triagonal_config, config.diagonal_config))
    try
        bat_sample(MvNormal(zeros(2), ones(2)), TransformedMCMC(nchains=2, nsteps=100,
            strict=false, init=MCMCChainPoolInit(nsteps_init=30), convergence=AssumeConvergence(),
            adaptive_transform=BAT.NoAdaptiveTransform(), transform_tuning=NoMCMCTransformTuning(),
            burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=20, max_ncycles=1, nsteps_final=2)),
            BATContext(rng=BAT.Philox4x((131, 32)), visualizer=vis))
        @test BAT.visualizer_status(vis).outcome === :complete
        check_controls(BAT.visualizer_figure(vis), EXT._live_graph(vis))
        @test BAT.visualizer_status(vis).presentation_valid
    finally
        BAT.dispose_visualizer!(vis)
        @test timedwait(() -> BAT.visualizer_status(vis).history_released, 15; pollint=0.02) == :ok
    end
end
