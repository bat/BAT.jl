# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieRangeEvents
using BAT, Makie, Test
const EXT = Base.get_extension(BAT, :BATMakieExt)
include("makie_control_testutils.jl")

function fixture()
    samples = DensitySampleVector(v=[[Float64(i),sin(i)] for i in 1:100], logd=zeros(100))
    graph = first(EXT._setup_static_graph(samples, (upper=Hist2D,diagonal=Hist1D,lower=Hist2D),
        [1,2], 2, BAT._default_makie_triagonal_config(), BAT._default_makie_diagonal_config()))
    built = with_theme(bat_theme()) do
        EXT._build_fig(graph, EXT._init_gridlayout!(graph,2); message_duration=60)
    end
    @test !isempty(Makie.colorbuffer(built.fig))
    return built, graph
end

function check_handles(slider, interval)
    # Inspect rendered native handle geometry, not the widget's private drag flag.
    points = last(filter(p -> p isa Makie.Scatter, slider.blockscene.plots))[1][]
    limit = last(slider.range[])
    @test points[1] ≈ range_point(slider, (interval[1]-1)/(limit-1))
    @test points[2] ≈ range_point(slider, (interval[2]-1)/(limit-1))
end

@testset "Track clicks, interval dragging, and handle crossover" begin
    built, graph = fixture()
    fig, c = built.fig, built.range_controls[]
    changes = Any[]
    listener = on(value -> push!(changes,value), graph[:window_steps])
    try
        range_text!(fig,(20,60); submit=true)
        empty!(changes)
        range_press!(fig,c.slider,9/99)
        @test c.slider.interval[] == (10,60)
        @test isempty(changes)
        range_release!(fig)
        @test changes == [(10,60)]
        range_press!(fig,c.slider,34/99)
        range_move!(fig,c.slider,44/99)
        @test length(changes) == 1
        range_release!(fig)
        @test changes == [(10,60),(20,70)]
        range_press!(fig,c.slider,19/99)
        range_move!(fig,c.slider,89/99)
        @test length(changes) == 2
        range_release!(fig)
        @test changes == [(10,60),(20,70),(70,90)]
        check_handles(c.slider,(70,90))
    finally
        off(listener)
        empty!(fig)
    end
end

@testset "Native reset ordering and unrelated releases" begin
    built, graph = fixture()
    fig, c = built.fig, built.range_controls[]
    try
        range_text!(fig,(20,80); submit=true)
        range_press!(fig,c.slider,19/99)
        range_release!(fig; tick=false)
        # The second release resets natively before BAT reads the final interval.
        range_press!(fig,c.slider,19/99; advance=false)
        range_release!(fig; tick=false)
        @test c.slider.interval[] == (1,100)
        @test graph[:window_steps][] == (20,80)
        range_tick!(fig)
        @test graph[:window_steps][] == (1,100)
        check_handles(c.slider,(1,100))

        range_text!(fig,(30,70))
        range_pointer!(fig,(-10,-10))
        range_release!(fig)
        @test graph[:window_steps][] == (1,100)
        range_key!(fig)
        @test graph[:window_steps][] == (30,70)
        check_handles(c.slider,(30,70))
    finally
        empty!(fig)
    end
end

@testset "Notice owns covered presses but permits native dragstop" begin
    built, graph = fixture()
    fig, c, notice = built.fig, built.range_controls[], built.controls.notice
    forwarded = Any[]
    listener = on(events(fig).mousebutton; priority=84) do event
        push!(forwarded,(event.button,event.action))
        Consume(false)
    end
    try
        range_text!(fig,(20,80); submit=true)
        notice.show!(EXT._marginal_capacity_error(3))
        range_press!(fig,c.slider,19/99)
        range_move!(fig,c.slider,29/99)
        tile = notice.bounds[]
        range_pointer!(fig,tile.origin + tile.widths/2)
        final = c.slider.interval[]
        count_before = length(forwarded)
        range_release!(fig)
        @test length(forwarded) == count_before + 1
        @test graph[:window_steps][] == final
        notice.dismiss!()
        range_text!(fig,(12,40); submit=true)
        check_handles(c.slider,(12,40)) # Fails if the notice swallowed native dragstop.
        range_pointer!(fig,range_point(c.slider,0.9))
        @test c.slider.interval[] == (12,40)

        # A notice-consumed press cannot leak its release after hide/dismiss.
        for dismiss in (false,true,true)
            notice.show!(EXT._marginal_capacity_error(3))
            range_pointer!(fig,notice.bounds[].origin + notice.bounds[].widths/2)
            count_before = length(forwarded)
            events(fig).mousebutton[] = Makie.MouseButtonEvent(Mouse.left,Mouse.press)
            dismiss && notice.dismiss!()
            range_release!(fig)
            @test length(forwarded) == count_before
            @test graph[:window_steps][] == (12,40)
        end
        notice.show!(EXT._marginal_capacity_error(3))
        close_bounds = notice.close.layoutobservables.computedbbox[]
        range_pointer!(fig,close_bounds.origin + close_bounds.widths/2)
        count_before = length(forwarded)
        events(fig).mousebutton[] = Makie.MouseButtonEvent(Mouse.left,Mouse.press)
        @test !notice.clock.active
        range_release!(fig)
        @test length(forwarded) == count_before
        check_handles(c.slider,(12,40))
    finally
        off(listener)
        empty!(fig)
    end
end

@testset "Cancellation and no-op respect newer programmatic windows" begin
    built, graph = fixture()
    fig, c = built.fig, built.range_controls[]
    try
        events(fig).hasfocus[] = true
        range_press!(fig,c.slider,0)
        EXT._apply_sample_window!(graph,(10,60))
        range_release!(fig)
        @test graph[:window_steps][] == (10,60)
        @test c.draft[] == (10,60)
        check_handles(c.slider,(10,60))
        range_press!(fig,c.slider,9/99)
        range_move!(fig,c.slider,19/99)
        EXT._apply_sample_window!(graph,(30,80))
        events(fig).hasfocus[] = false
        range_release!(fig)
        @test graph[:window_steps][] == (30,80)
        @test c.draft[] == (30,80)
        check_handles(c.slider,(30,80))
        # Pending completion is cancelled by real window closure.
        events(fig).window_open[] = true
        range_press!(fig,c.slider,29/99)
        range_move!(fig,c.slider,39/99)
        range_release!(fig; tick=false)
        events(fig).window_open[] = false
        range_tick!(fig)
        @test graph[:window_steps][] == (30,80)
    finally
        empty!(fig)
    end
end
end
