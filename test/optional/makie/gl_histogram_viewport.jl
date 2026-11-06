# This file is a part of BAT.jl, licensed under the MIT License (MIT).
# Run in a separate process with an external deadline and a native GL display.
using BAT, Makie, GLMakie, Test
@test realpath(pkgdir(BAT)) == realpath(joinpath(@__DIR__, "../../.."))
const EXT = Base.get_extension(BAT, :BATMakieExt)
GLMakie.activate!()
Makie.inline!(false)

function heatmap(axis)
    plots = Makie.Heatmap[]
    function visit(p)
        p isa Makie.Heatmap && push!(plots, p)
        foreach(visit, p.plots)
    end
    foreach(visit, axis.scene.plots)
    return only(plots)
end

@testset "Native GL fixed histogram viewport" begin
    for closed in (:left, :right)
        samples = DensitySampleVector([[0.1, -0.9], [0.9, -0.1]], zeros(2))
        pair = merge(BAT._default_makie_triagonal_config(), (nbins=(2, 4), closed, levels=[1.0]))
        diag = merge(BAT._default_makie_diagonal_config(), (nbins=2, closed,))
        graph, _ = EXT._setup_static_graph(samples,
            (upper=Hist2D, diagonal=QuantileHist1D, lower=QuantileHist2D), [1, 2], 2, pair, diag)
        fig = Figure()
        screen = nothing
        try
            EXT._bind_interactive_grid!(fig, EXT._init_gridlayout!(graph, 2))
            screen = GLMakie.Screen(; visible=false, start_renderloop=false)
            display(screen, fig; start_renderloop=false)
            @test !isempty(Makie.colorbuffer(screen))
            for (i, j) in ((1, 2), (2, 1))
                ax = only(filter(x -> x isa Axis, contents(fig[1, 1][i, j])))
                p = heatmap(ax)
                # Test the actually rendered coordinate arrays and final limits,
                # including the lower panel's transposed coordinates.
                @test extrema(p[1][]) == (j == 1 ? (0.0, 1.0) : (-1.0, 0.0))
                @test extrema(p[2][]) == (i == 1 ? (0.0, 1.0) : (-1.0, 0.0))
                for dim in 1:2
                    bounds = ax.finallimits[]
                    @test bounds.origin[dim] <= first(p[dim][])
                    @test bounds.origin[dim] + bounds.widths[dim] >= last(p[dim][])
                end
            end
            cached = graph[:axis_limits][]
            Makie.update!(graph; window_steps=(Int64(20), Int64(30)))
            @test graph[:axis_limits][] === cached
            @test !isempty(Makie.colorbuffer(screen))
            Makie.update!(graph; window_steps=nothing)
            @test graph[:axis_limits][] === cached
            @test !isempty(Makie.colorbuffer(screen))
        finally
            isnothing(screen) || close(screen)
            empty!(fig)
        end
        @test isempty(fig.content)
    end
end
