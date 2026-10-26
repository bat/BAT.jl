# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestMakieLayerOrder
using BAT, Makie, Test

axis(fig) = only(filter(x -> x isa Axis, fig.content))
snapshot(fig) = [(typeof(p), deepcopy(p.converted[])) for p in axis(fig).scene.plots]

@testset "Window updates preserve visible statistical overlays" begin
    values = [-5.123, -4.731, -4.098, -3.342, -2.759, 0.12345, 1.12345, 2.12345]
    samples = DensitySampleVector([[v] for v in values], zeros(8))
    original = deepcopy(samples)
    recipes = (upper = Hist2D, diagonal = QuantileHist1D, lower = Hist2D)
    fig = bat_makie_plot(samples, recipes, [1], 1; display = false)
    backend = Makie.current_backend()
    native = nameof(backend) === :GLMakie
    screen = nothing
    try
        held_axis = axis(fig)
        held_axis.title = "retained axis"
        upper_stats, stats = filter(x -> x isa Toggle, fig.content)[1:2]
        stats.active[] = true
        textbox = only(filter(x -> x isa Textbox, fig.content))
        apply = only(filter(x -> x isa Button && x.label[] == "Apply", fig.content))
        if native
            screen = backend.Screen(; visible = false, start_renderloop = false)
            display(screen, fig; start_renderloop = false)
        end
        render() = copy(native ? Makie.colorbuffer(screen) : Makie.colorbuffer(fig))

        # Nested polygon counts change enough for Makie to replace only the fill
        # while reusing the overlay. Exercise both shrinking and growing windows.
        for rows in (6:8, 1:8, 6:8)
            textbox.stored_string[] = "$(first(rows)):$(last(rows))"
            Makie.notify(apply.clicks)
            @test axis(fig) === held_axis
            @test held_axis.title[] == "retained axis"
            plots = held_axis.scene.plots
            poly = only(filter(p -> p isa Makie.Poly, plots))
            meanline = first(filter(p -> p isa Makie.VLines, plots))
            @test only(meanline[1][]) ≈ sum(values[rows]) / length(rows)
            @test findfirst(p -> p === poly, plots) < findfirst(p -> p === meanline, plots)
            if rows == 6:8
                @test any(p -> minimum(v[1] for v in p) < only(meanline[1][]) <
                    maximum(v[1] for v in p), poly[1][])
            end
            before = snapshot(fig)
            pixels_before = render()
            @test !isempty(pixels_before)

            # An unrelated overlay change recomposes the grid but keeps this
            # diagonal's cached fill. The diagonal overlay round trip then forces
            # a complete rebuild with the same data and statistics.
            upper_stats.active[] = true
            @test only(filter(p -> p isa Makie.Poly, held_axis.scene.plots)) === poly
            upper_stats.active[] = false
            stats.active[] = false
            stats.active[] = true
            after = snapshot(fig)
            @test length(before) == length(after)
            @test all(old -> any(new -> isequal(old, new), after), before)
            @test pixels_before == render()
        end
        @test samples == original
    finally
        isnothing(screen) || close(screen)
        empty!(fig)
    end
    @test isempty(fig.content)
end
end
