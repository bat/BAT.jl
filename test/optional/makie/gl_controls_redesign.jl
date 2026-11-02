# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT, GLMakie, Makie, Test

GLMakie.activate!()
Makie.inline!(false)
const EXT = Base.get_extension(BAT, :BATMakieExt)

function pointer!(fig, position)
    events(fig.scene).mouseposition[] = Tuple(Float64.(position))
end

function click!(fig, block)
    pointer!(fig, (-10, -10))
    box = block.layoutobservables.computedbbox[]
    position = box.origin + box.widths / 2
    pointer!(fig, position + Vec2f(0, 1))
    pointer!(fig, position)
    events(fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.press)
    events(fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.release)
end

function key!(fig, key)
    events(fig.scene).keyboardbutton[] = Makie.KeyEvent(key, Keyboard.press)
    events(fig.scene).keyboardbutton[] = Makie.KeyEvent(key, Keyboard.release)
end

@testset "Native redesigned controls" begin
    samples = DensitySampleVector(v = [[Float64(i + j/10) for i in 1:12] for j in 1:20], logd = zeros(20))
    recipes = (upper = Hist2D, diagonal = Hist1D, lower = Hist2D)
    graph = first(EXT._setup_static_graph(samples, recipes, [1, 2], 3,
        BAT._default_makie_triagonal_config(), BAT._default_makie_diagonal_config()))
    info = EXT.PickerInfo(12, 3, [1, 2], selection -> EXT._apply_vsel_to_graph!(graph, 12, 3, selection))
    built = with_theme(bat_theme()) do
        EXT._build_fig(graph, EXT._init_gridlayout!(graph, 3), info)
    end
    fig, controls = built.fig, built.controls
    picker = controls.picker
    screen = GLMakie.Screen(; visible = false, start_renderloop = false)
    try
        display(screen, fig)
        Makie.notify(events(fig.scene).tick)
        Makie.resize!(fig, 300, 300)
        Makie.notify(events(fig.scene).tick)
        @test Tuple(Makie.viewport(fig.scene)[].widths) == controls.minimum_size
        click!(fig, controls.collapse)
        @test controls.expanded[]
        @test !isempty(Makie.colorbuffer(fig))
        pointer!(fig, (-10, -10))
        pixels = Makie.colorbuffer(fig)
        for button in (picker.reset, picker.switch)
            bounds = button.layoutobservables.computedbbox[]
            x, y = round.(Int, bounds.origin)
            w, h = round.(Int, bounds.widths)
            patch = pixels[size(pixels, 1)-y-h+4:size(pixels, 1)-y-3, x+4:x+w-3]
            @test count(c -> c.r + c.g + c.b < 1.5, patch) >= 8
        end
        header = picker.headers[(1, :row)]
        click!(fig, header.box)
        @test header.box.focused[]
        if EXT._legacy_control_textbox(header.box)
            @test isnothing(header.box.cursoranimtask)
        end
        header.replace!("")
        events(fig.scene).unicode_input[] = '9'
        @test picker.headers[(1, :column)].box.displayed_string[] == "9"
        key!(fig, Keyboard.enter)
        @test graph[:idxs][] == [2, 9]

        header.replace!("2")
        key!(fig, Keyboard.enter)
        @test controls.notice.clock.active
        @test graph[:idxs][] == [2, 9]
        key!(fig, Keyboard.escape)
        @test header.box.displayed_string[] == "9"
        @test !controls.notice.clock.active

        click!(fig, picker.switch)
        @test picker.mode[] === :input
        picker.editor.replace!("(2,9); (9,12)")
        key!(fig, Keyboard.enter)
        @test picker.mode[] === :matrix
        @test graph[:idxs][] == [2, 9, 12]
        click!(fig, picker.switch)
        picker.editor.replace!("1:12")
        click!(fig, picker.switch)
        @test picker.mode[] === :input
        @test graph[:idxs][] == [2, 9, 12]
        click!(fig, picker.cancel)
        @test picker.mode[] === :matrix
        click!(fig, picker.reset)
        @test graph[:idxs][] == [1, 2]
        @test picker.state.headers == picker.state.initial_headers

        range = controls.range[]
        click!(fig, range.value_entry)
        range.value_control.replace!("3:17")
        @test isnothing(graph[:window_steps][])
        click!(fig, range.apply)
        @test graph[:window_steps][] == (3, 17)

        for size in (controls.minimum_size, controls.minimum_size .+ (180, 100))
            Makie.resize!(fig, size...)
            @test !isempty(Makie.colorbuffer(fig))
            menu = only(filter(x -> x isa Menu && x.selection[] === Hist1D, fig.content))
            menu.is_open[] = true
            @test !isempty(Makie.colorbuffer(fig))
            menu.is_open[] = false
        end
        if haskey(ENV, "BAT_UI_EVIDENCE")
            folder = ENV["BAT_UI_EVIDENCE"]
            mkpath(folder)
            pointer!(fig, (-10, -10))
            save(joinpath(folder, "expanded.png"), fig)
            picker.open_editor!()
            picker.editor.replace!("(2,9); (9,12)")
            save(joinpath(folder, "dimension-input.png"), fig)
            picker.cancel_editor!()
            controls.notice.show!(EXT._marginal_capacity_error(3))
            save(joinpath(folder, "notice.png"), fig)
            click!(fig, controls.collapse)
            pointer!(fig, (-10, -10))
            save(joinpath(folder, "folded.png"), fig)
        end
        picker.open_editor!()
        empty!(fig)
        # Makie intentionally retains its priority-typemax keyboard-state tracker.
        @test all(p -> first(p) == typemax(Int), Makie.Observables.listeners(events(fig.scene).keyboardbutton))
        @test isempty(fig.content)
        replacement = Figure(size = (300, 300))
        try
            display(screen, replacement)
            Makie.notify(events(replacement.scene).tick)
            @test Tuple(Makie.viewport(replacement.scene)[].widths) == (300, 300)
            @test GLMakie.window_size(screen.glscreen) == GLMakie.window_size(screen, 300, 300)
        finally
            empty!(replacement)
        end
    finally
        close(screen)
        empty!(fig)
    end
end
