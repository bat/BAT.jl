# This file is a part of BAT.jl, licensed under the MIT License (MIT).
# Run with an external deadline and a working native OpenGL display.
using BAT, Makie, GLMakie, Test
@test realpath(pkgdir(BAT)) == realpath(joinpath(@__DIR__, "../../.."))
const EXT = Base.get_extension(BAT, :BATMakieExt)
GLMakie.activate!()
Makie.inline!(false)
const DATA = [-2. -1. 0. 1. 2.; 1. -2. 2. 0. -1.]
const CONFIG = BAT._default_makie_triagonal_config()

function primitives(scale, recipe)
    data = scale .* DATA
    weights = ones(size(data, 2))
    return EXT.compute_plotting_primitives(view(data, :, :), view(weights, :),
        recipe, EXT.LiveRecipe(), EXT.LiveCell(), CONFIG)
end

@testset "Unbounded KDE native GL arithmetic admission" begin
    for (scale, recipe) in ((1.0, KDE2D()), (1e8, KDE2D()),
            (1e-8, KDE2D()), (1e20, QuantileKDE2D())), transposed in (false, true)
        p = primitives(scale, recipe)
        spec = only(EXT.compose_plotspecs(p, recipe, CONFIG; transposed))
        fig = Figure(size=(400, 320))
        screen = nothing
        try
            ax = Axis(fig[1, 1])
            plot = heatmap!(ax, spec.args...; spec.kwargs...)
            @test all(>(0), diff(plot[1][])) && all(>(0), diff(plot[2][]))
            @test plot[1][] == (transposed ? p.yedges : p.xedges)
            @test plot[2][] == (transposed ? p.xedges : p.yedges)
            expected = recipe isa KDE2D ? p.density : p.color_grid
            transposed && (expected = permutedims(expected))
            @test isequal(plot[3][], recipe isa KDE2D ? Float32.(expected) : expected)
            screen = GLMakie.Screen(; visible=false, start_renderloop=false)
            display(screen, fig; start_renderloop=false)
            pixels = Makie.colorbuffer(screen)
            @test !isempty(pixels)
            # Black/gray axes alone cannot satisfy this rendering check.
            @test count(c -> max(c.r,c.g,c.b)-min(c.r,c.g,c.b) > 0.1, pixels) > 100
            if recipe isa QuantileKDE2D
                colors = filter(c -> c.alpha == 1, unique(vec(p.color_grid)))
                @test !isempty(colors)
                @test any(color -> count(c -> maximum(abs.((c.r-color.r,c.g-color.g,c.b-color.b))) < 0.02,
                    pixels) > 100, colors)
            else
                lo, hi = plot.scaled_colorrange[]
                @test isfinite(inv(hi-lo)) && inv(hi-lo) >= 2floatmin(Float32)
            end
        finally
            isnothing(screen) || close(screen)
            empty!(fig)
        end
    end
    unsafe = primitives(1e20, KDE2D())
    @test all(d -> isnan(d) || 0 < Float32(d) < Inf, unsafe.density)
    @test_throws ArgumentError EXT.compose_plotspecs(unsafe, KDE2D(), CONFIG)
    @test_throws ArgumentError EXT.compose_plotspecs(unsafe, KDE2D(), CONFIG; transposed=true)
    for scale in (1e-100, 1e100), recipe in (KDE2D(), QuantileKDE2D())
        @test_throws ArgumentError primitives(scale, recipe)
    end
end
