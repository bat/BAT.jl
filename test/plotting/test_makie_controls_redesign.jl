# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieControlsRedesign

using BAT, CairoMakie, Makie, Test

const EXT = Base.get_extension(BAT, :BATMakieExt)
include("makie_control_testutils.jl")
const RECIPES = (upper = Hist2D, diagonal = Hist1D, lower = Hist2D)

function fixture(; n = 12, selection = [11, 2], capacity = 3, dark = false)
    samples = DensitySampleVector(v = [[Float64(i + j / 10) for i in 1:n] for j in 1:20], logd = zeros(20))
    graph = first(EXT._setup_static_graph(samples, RECIPES, selection, capacity,
        BAT._default_makie_triagonal_config(), BAT._default_makie_diagonal_config()))
    built = with_theme(dark ? bat_theme_dark() : bat_theme()) do
        info = EXT.PickerInfo(n, capacity, selection,
            s -> EXT._apply_vsel_to_graph!(graph, n, capacity, s))
        EXT._build_fig(graph, EXT._init_gridlayout!(graph, capacity), info)
    end
    return built, graph, samples
end

@testset "Marginal controls preserve selection and construction reset" begin
    for dark in (false, true)
        built, graph, samples = fixture(; dark)
        fig, controls = built.fig, built.controls
        picker = controls.picker
        original = deepcopy(samples)
        axes = filter(x -> x isa Axis, fig.content)
        try
            @test !controls.expanded[]
            @test all(h -> !h.box.blockscene.visible[], values(picker.headers))
            Makie.notify(controls.collapse.clicks)
            @test controls.expanded[]
            @test length(picker.checks) == 21
            @test all(cb -> cb.roundness[] > 0, values(picker.checks))

            header = picker.headers[(2, :row)]
            header.replace!("9")
            @test picker.headers[(2, :column)].box.displayed_string[] == "9"
            @test graph[:idxs][] == [11, 2]
            @test picker.commit_header!(2)
            @test graph[:idxs][] == [9, 11]
            @test picker.state.headers[2] == 9
            header.replace!("1")
            @test !picker.commit_header!(2)
            @test graph[:idxs][] == [9, 11]
            @test controls.notice.clock.active
            @test haskey(picker.header_errors, 2)
            Makie.notify(controls.notice.close.clicks)
            @test !controls.notice.clock.active
            @test haskey(picker.header_errors, 2) # Dismissing a notice does not validate input.

            Makie.notify(picker.reset.clicks)
            @test graph[:idxs][] == [11, 2]
            @test picker.state.headers == picker.state.initial_headers
            @test isempty(picker.header_errors)
            @test isempty(picker.header_drafts)
            picker.open_editor!()
            @test picker.mode[] === :input
            @test all(cb -> cb.size[] == 0, values(picker.checks))
            picker.editor.replace!("(3,8); (8,12)")
            @test graph[:idxs][] == [11, 2]
            @test picker.commit_editor!()
            @test graph[:idxs][] == [3, 8, 12]
            @test picker.state.headers[1:3] == [3, 8, 12]
            @test picker.mode[] === :matrix
            @test all(cb -> cb.size[] > 0, values(picker.checks))

            picker.open_editor!()
            picker.editor.replace!("1:12")
            @test !picker.commit_editor!()
            @test graph[:idxs][] == [3, 8, 12]
            @test picker.mode[] === :input
            Makie.notify(picker.cancel.clicks)
            @test picker.mode[] === :matrix
            @test graph[:idxs][] == [3, 8, 12]
            picker.open_editor!()
            picker.editor.replace!("")
            @test picker.commit_editor!()
            @test isempty(graph[:idxs][])
            Makie.notify(picker.reset.clicks)
            @test graph[:idxs][] == [11, 2]
            @test filter(x -> x isa Axis, fig.content) == axes
            @test samples == original

            picker.open_editor!()
            picker.editor.replace!("4, 7")
            Makie.notify(controls.collapse.clicks)
            Makie.resize!(fig, (controls.minimum_size .+ (180, 100))...)
            Makie.notify(controls.collapse.clicks)
            @test picker.mode[] === :input
            @test picker.editor.box.displayed_string[] == "4, 7"
            @test all(cb -> cb.size[] == 0, values(picker.checks))
            picker.cancel_editor!()
            for size in (controls.minimum_size, controls.minimum_size .+ (180, 100))
                Makie.resize!(fig, size...)
                @test !isempty(Makie.colorbuffer(fig))
                for header in values(picker.headers)
                    bounds = header.box.layoutobservables.computedbbox[]
                    @test all(bounds.origin .>= 0)
                    @test all(bounds.origin .+ bounds.widths .<= Vec2f(size))
                end
            end
            controls.notice.show!(EXT._marginal_capacity_error(3))
            @test !isempty(Makie.colorbuffer(fig))
            @test controls.notice.bounds[].origin[2] >= 0
            # Export must expire stale notices even without a native render loop.
            events(fig.scene).mouseposition[] = (-10.0, -10.0)
            controls.notice.clock.updated = time_ns() - UInt64(4_000_000_000)
            @test !isempty(Makie.colorbuffer(fig))
            @test !controls.notice.clock.active
            mutations = Ref(0)
            watcher = on(controls.notice.close.height) do _
                mutations[] += 1
            end
            try
                for _ in 1:5
                    Makie.notify(events(fig.scene).tick)
                end
                @test mutations[] == 0
            finally
                off(watcher)
            end
            if dark && haskey(ENV, "BAT_UI_EVIDENCE")
                mkpath(ENV["BAT_UI_EVIDENCE"])
                save(joinpath(ENV["BAT_UI_EVIDENCE"], "dark-expanded.png"), fig)
            end

            # Paste admission occurs before any native displayed-string mutation.
            before = picker.editor.box.displayed_string[]
            @test !picker.editor.insert!("9"^5000)
            @test picker.editor.box.displayed_string[] == before
            picker.editor.focus!()
            if EXT._legacy_control_textbox(picker.editor.box)
                @test isnothing(picker.editor.box.cursoranimtask)
            end
            empty!(fig)
            @test isempty(Makie.Observables.listeners(picker.reset.clicks))
            @test isempty(Makie.Observables.listeners(events(fig.scene).unicode_input))
            Makie.notify(picker.reset.clicks)
            @test graph[:idxs][] == [11, 2]
        finally
            empty!(fig)
        end
    end
end

@testset "Notice elapsed time, pauses, and replacement" begin
    clock = EXT._ControlNoticeClock(3.0)
    EXT._start_notice!(clock, UInt64(1_000_000_000))
    EXT._advance_notice!(clock, true, UInt64(2_000_000_000))
    @test clock.remaining == 2.0
    EXT._advance_notice!(clock, false, UInt64(20_000_000_000))
    @test clock.remaining == 2.0
    EXT._advance_notice!(clock, false, UInt64(21_500_000_000))
    @test clock.remaining == 0.5
    EXT._advance_notice!(clock, false, UInt64(22_000_000_000))
    @test !clock.active
    EXT._start_notice!(clock, UInt64(30_000_000_000))
    @test clock.remaining == 3.0
    @test clock.active
end

function blur_range!(fig)
    range_pointer!(fig, (-10,-10))
    events(fig).mousebutton[] = Makie.MouseButtonEvent(Mouse.left,Mouse.press)
    range_release!(fig)
end

@testset "Exact range submission, blur, cancellation, and live edits" begin
    built, graph, samples = fixture(n=2, selection=[1,2], capacity=2)
    fig, controls = built.fig, built.range_controls[]
    changes = Any[]
    watcher = on(value -> push!(changes,value),graph[:window_steps])
    try
        @test_throws ArgumentError bat_makie_plot(samples; display=false, message_duration=0)
        @test_throws ArgumentError bat_makie_plot(samples; display=false, message_duration=Inf)
        @test !any(x -> x isa Button && x.label[] == "Apply range",fig.content)
        @test controls.title.text[] == "Sample range per walker:"
        @test !isempty(Makie.colorbuffer(fig))
        controls.value_control.focus!()
        blur_range!(fig)
        @test isempty(changes)
        @test isnothing(graph[:window_steps][])

        range_text!(fig,(3,17))
        @test isempty(changes)
        range_key!(fig)
        @test changes == [(3,17)]
        blur_range!(fig)
        @test changes == [(3,17)]
        range_text!(fig,(5,15))
        blur_range!(fig)
        @test changes == [(3,17),(5,15)]
        controls.value_control.focus!()
        blur_range!(fig)
        @test length(changes) == 2

        range_text!(fig,(6,14))
        range_key!(fig,Keyboard.kp_enter)
        @test graph[:window_steps][] == (6,14)
        controls.value_control.replace!("7:")
        range_key!(fig)
        @test graph[:window_steps][] == (6,14)
        @test controls.value_entry.bordercolor[] == EXT._status_text_color(fig.scene.backgroundcolor[])
        blur_range!(fig)
        @test graph[:window_steps][] == (6,14)
        controls.value_control.focus!()
        range_key!(fig,Keyboard.escape)
        @test controls.value_entry.displayed_string[] == "6:14"
        @test !controls.value_entry.focused[]
        @test !controls.value_control.replace!(repeat("1",49))
        @test controls.value_entry.displayed_string[] == "6:14"

        # Text blur settles before the next slider gesture takes its baseline.
        range_text!(fig,(10,18))
        range_press!(fig,controls.slider,9/19)
        @test graph[:window_steps][] == (10,18)
        range_move!(fig,controls.slider,11/19)
        @test graph[:window_steps][] == (10,18)
        range_release!(fig)
        @test graph[:window_steps][] == (12,18)
        @test sum(graph[:flat_weights][]) == 7
        range_text!(fig,(1,20);submit=true)
        @test graph[:window_steps][] == (1,20) # Explicit full extent is fixed.

        controls.live_drag[] = true
        range_text!(fig,(4,14))
        @test graph[:window_steps][] == (4,14)
        controls.value_control.replace!("bad")
        @test graph[:window_steps][] == (4,14)
        @test controls.value_entry.bordercolor[] == EXT._status_text_color(fig.scene.backgroundcolor[])
        controls.value_control.replace!("7:12")
        @test graph[:window_steps][] == (7,12)
        @test controls.value_entry.bordercolor[] == EXT._panel_bg_color(fig.scene.backgroundcolor[],0.3)
        before = copy(changes)
        blur_range!(fig)
        @test changes == before
    finally
        off(watcher)
        empty!(fig)
    end
end

end
