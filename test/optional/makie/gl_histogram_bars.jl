# This file is a part of BAT.jl, licensed under the MIT License (MIT).
# Run separately with an external deadline and an actual native GL display.
using BAT, Makie, GLMakie, Test
@test realpath(pkgdir(BAT)) == realpath(joinpath(@__DIR__, "../../.."))
const EXT = Base.get_extension(BAT, :BATMakieExt)
GLMakie.activate!();
Makie.inline!(false);
@testset "H37 native GL edge-derived bar geometry" begin
    for point in (0.0, 2.0, 1e8)
        source = DensitySampleVector([[point]], zeros(1))
        f = bat_makie_plot(
            source,
            (upper = Hist2D, diagonal = Hist1D, lower = Hist2D),
            [1],
            1;
            display = false,
        )
        screen = nothing
        try
            ax = only(filter(x -> x isa Axis, contents(f[1, 1][1, 1])))
            p = only(filter(x -> x isa EXT.BATCheckedHistBars, ax.scene.plots))
            b = only(p.plots)
            edges = p.bin_edges[]
            r = only(b.plots[1][1][])
            @test r.origin[1] == first(edges)
            @test r.origin[1] + r.widths[1] == last(edges)
            @test r.widths[1] * only(p.bin_heights[]) ≈ 1.0
            screen = GLMakie.Screen(; visible = false, start_renderloop = false)
            display(screen, f; start_renderloop = false)
            buffer = Makie.colorbuffer(screen)
            @test !isempty(buffer)
            # Actual bar color occupies a substantial horizontal span in the image.
            color = b.color[]
            pixels = findall(
                c -> max(abs(c.r - color.r), abs(c.g - color.g), abs(c.b - color.b)) < 0.03,
                buffer,
            )
            @test length(pixels) > 0.5 * prod(ax.scene.viewport[].widths)
        finally
            isnothing(screen) || close(screen)
            empty!(f)
        end
        @test isempty(f.content)
    end
end
