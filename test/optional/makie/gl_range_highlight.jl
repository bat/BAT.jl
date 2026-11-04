# This file is a part of BAT.jl, licensed under the MIT License (MIT).
# Run separately with an external deadline and a working native GL display.
using BAT, GLMakie, Makie, Logging, Test
include("../../plotting/makie_control_testutils.jl")

const EXT = Base.get_extension(BAT, :BATMakieExt)
@test realpath(pkgdir(BAT)) == realpath(joinpath(@__DIR__, "../../.."))
GLMakie.activate!()
Makie.inline!(false)

function button_pixels(screen, button)
    pixels = Makie.colorbuffer(screen)
    bounds = button.layoutobservables.computedbbox[]
    x, y = floor.(Int, bounds.origin .* screen.px_per_unit[])
    w, h = floor.(Int, bounds.widths .* screen.px_per_unit[])
    return pixels[end-y-h+1:end-y, x+1:x+w]
end

@testset "Native pending range highlight" begin
    # Include a named translucent border to check theme preservation, not just
    # the built-in transparent border. Theme values need not be Colorants.
    themes = (bat_theme(), bat_theme_dark(),
        merge(Theme(Button = (strokecolor = (:red, 0.25),)), bat_theme()))
    for theme in themes
        samples = DensitySampleVector(v = [[Float64(i), sin(i)] for i in 1:100], logd = zeros(100))
        recipes = (upper = Hist2D, diagonal = Hist1D, lower = Hist2D)
        graph = first(EXT._setup_static_graph(samples, recipes, [1, 2], 2,
            BAT._default_makie_triagonal_config(), BAT._default_makie_diagonal_config()))
        built = with_theme(theme) do
            EXT._build_fig(graph, EXT._init_gridlayout!(graph, 2))
        end
        fig, controls = built.fig, built.controls
        range = controls.range[]
        screen = GLMakie.Screen(; visible = false, start_renderloop = false)
        try
            # A stopped screen makes logged GL failures synchronous with this
            # test. Rendering only after Apply misses the pending-color update.
            @test_logs min_level=Logging.Warn begin
                display(screen, fig; start_renderloop = false)
                events(fig.scene).mouseposition[] = (-10.0, -10.0)
                initial = button_pixels(screen, range.apply)
                @test Makie.to_color(range.apply.strokecolor[]) == Makie.to_color(controls.collapse.strokecolor[])
                range_text!(fig, (20,80))
                @test isnothing(graph[:window_steps][])
                @test count(button_pixels(screen, range.apply) .!= initial) > 20
                Makie.notify(range.apply.clicks)
                @test graph[:window_steps][] == (20, 80)
                @test button_pixels(screen, range.apply) == initial

                range.value_control.replace!("3:70")
                @test graph[:window_steps][] == (20, 80)
                @test count(button_pixels(screen, range.apply) .!= initial) > 20
                Makie.notify(range.apply.clicks)
                @test graph[:window_steps][] == (3, 70)
                @test button_pixels(screen, range.apply) == initial

                range_text!(fig, (5,60))
                @test count(button_pixels(screen, range.apply) .!= initial) > 20
                range.live_drag[] = true
                range_key!(fig)
                @test graph[:window_steps][] == (5, 60)
                @test button_pixels(screen, range.apply) == initial
                range_text!(fig, (7,50))
                @test graph[:window_steps][] == (7, 50)
                @test button_pixels(screen, range.apply) == initial
                range.live_drag[] = false
                range_text!(fig, (9,40))
                @test graph[:window_steps][] == (7, 50)
                @test count(button_pixels(screen, range.apply) .!= initial) > 20
                Makie.notify(range.apply.clicks)
                @test graph[:window_steps][] == (9, 40)
                @test button_pixels(screen, range.apply) == initial
            end
        finally
            close(screen)
            empty!(fig)
        end
    end
end
