# This file is a part of BAT.jl, licensed under the MIT License (MIT).
# Run separately with an external deadline and a working native GL display.
using BAT, GLMakie, Makie, Logging, Test
include("../../plotting/makie_control_testutils.jl")
const EXT = Base.get_extension(BAT, :BATMakieExt)
@test realpath(pkgdir(BAT)) == realpath(joinpath(@__DIR__, "../../.."))
GLMakie.activate!(visible=false)
Makie.inline!(false)

function check_range_layout(c)
    title = c.title.layoutobservables.computedbbox[]
    field = c.value_entry.layoutobservables.computedbbox[]
    slider = c.slider.layoutobservables.computedbbox[]
    toggle = c.live_drag_toggle.layoutobservables.computedbbox[]
    label = c.live_drag_button.layoutobservables.computedbbox[]
    left = min(title.origin[1],field.origin[1])
    right = max(title.origin[1]+title.widths[1],field.origin[1]+field.widths[1])
    @test (left+right)/2 ≈ slider.origin[1]+slider.widths[1]/2 atol=1
    top = max(title.origin[2]+title.widths[2],field.origin[2]+field.widths[2])
    live_bottom = min(toggle.origin[2],label.origin[2])
    live_top = max(toggle.origin[2]+toggle.widths[2],label.origin[2]+label.widths[2])
    @test (live_bottom+live_top)/2 ≈ (top+slider.origin[2])/2 atol=1
    @test right <= label.origin[1]
end

@testset "Native range rendering and layout" begin
    themes = (bat_theme(),bat_theme_dark(),
        merge(Theme(Button=(strokecolor=(:red,0.25),)),bat_theme()))
    for (name,theme) in zip(("light","dark","translucent"),themes)
        samples = DensitySampleVector(v=[[Float64(i),sin(i)] for i in 1:100],logd=zeros(100))
        recipes = (upper=Hist2D,diagonal=Hist1D,lower=Hist2D)
        graph = first(EXT._setup_static_graph(samples,recipes,[1,2],2,
            BAT._default_makie_triagonal_config(),BAT._default_makie_diagonal_config()))
        built = with_theme(theme) do
            EXT._build_fig(graph,EXT._init_gridlayout!(graph,2))
        end
        fig, controls = built.fig,built.controls
        c = controls.range[]
        screen = GLMakie.Screen(;visible=false,start_renderloop=false)
        try
            # Rendering synchronously makes GL update failures visible to Test.
            @test_logs min_level=Logging.Warn begin
                display(screen,fig;start_renderloop=false)
                @test !any(x -> x isa Button && x.label[] == "Apply range",fig.content)
                for size in (controls.minimum_size,controls.minimum_size .+ (300,200))
                    resize!(fig,size...)
                    for expanded in (false,true)
                        controls.expanded[] = expanded
                        @test !isempty(Makie.colorbuffer(screen))
                        check_range_layout(c)
                        if haskey(ENV,"BAT_UI_EVIDENCE") && size == controls.minimum_size
                            folder = ENV["BAT_UI_EVIDENCE"]
                            mkpath(folder)
                            state = expanded ? "expanded" : "folded"
                            save(joinpath(folder,"range-$name-$state.png"),fig)
                        end
                    end
                end
                range_text!(fig,(20,80))
                @test isnothing(graph[:window_steps][])
                @test !isempty(Makie.colorbuffer(screen))
                range_key!(fig)
                @test graph[:window_steps][] == (20,80)
                range_press!(fig,c.slider,19/99)
                range_move!(fig,c.slider,29/99)
                @test !isempty(Makie.colorbuffer(screen))
                @test graph[:window_steps][] == (20,80)
                range_release!(fig)
                @test graph[:window_steps][] == (30,80)
                @test !isempty(Makie.colorbuffer(screen))
                c.live_drag[] = true
                range_text!(fig,(7,50))
                @test graph[:window_steps][] == (7,50)
                c.value_control.replace!("bad")
                @test !isempty(Makie.colorbuffer(screen))
                @test graph[:window_steps][] == (7,50)
                c.value_control.replace!("8:45")
                @test graph[:window_steps][] == (8,45)
                @test !isempty(Makie.colorbuffer(screen))
                c.live_drag[] = false
                range_text!(fig,(9,40))
                @test graph[:window_steps][] == (8,45)
                range_pointer!(fig,(-10,-10))
                events(fig).mousebutton[] = Makie.MouseButtonEvent(Mouse.left,Mouse.press)
                range_release!(fig)
                @test graph[:window_steps][] == (9,40)
                @test !isempty(Makie.colorbuffer(screen))
            end
        finally
            close(screen)
            empty!(fig)
        end
    end
end
