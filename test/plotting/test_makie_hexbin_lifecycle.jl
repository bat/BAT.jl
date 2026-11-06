# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieHexbinLifecycle
using BAT, Makie, CairoMakie, Test
const EXT = Base.get_extension(BAT, :BATMakieExt)
const CONFIG = merge(BAT._default_makie_triagonal_config(), (nbins=(5, 5), colormap=:viridis))
const XY = [0.0 0.5 1.0; 0.0 0.8 1.0]

function spec(weights)
    raw = EXT.compute_plotting_primitives(view(XY, :, :), view(weights, :), Hexbin2D(),
        EXT.LiveRecipe(), EXT.LiveCell(), CONFIG)
    Makie.SpecApi.Axis(plots=EXT.compose_plotspecs(raw, Hexbin2D(), CONFIG),
        limits=(0.0, 1.0, 0.0, 1.0), xlabel="x", ylabel="y", xlabelvisible=true, ylabelvisible=true)
end
function setup(fig)
    ax = Axis(fig[1, 1])
    obs = Observable(Makie.PlotSpec[])
    Makie.update_plotspecs!(ax.scene, obs)
    return ax, obs
end

@testset "BAT-owned hexbin update rejects before renderer demand" begin
    for weights in (1e100 .* [1, 2, 3], 1e-100 .* [1, 2, 3], 1e10 .+ [0, 512, 1025])
        fig = Figure()
        ax, obs = setup(fig)
        good = spec([1.0, 2.0, 3.0])
        try
            EXT._update_grid_axis!(ax, obs, good, nothing)
            err = try
                EXT._update_grid_axis!(ax, obs, spec(weights), good.kwargs)
                nothing
            catch e
                e
            end
            @test err !== nothing && occursin("Hexbin2D displayed counts", sprint(showerror, err))
        finally
            empty!(fig)
        end
        @test isempty(fig.content)
        @test isempty(Makie.Observables.listeners(obs))
    end
end

@testset "Existing live owner disposes rejected count updates" begin
    # Focused real-owner/view/listener fixture. No sampler-equivalence claim.
    vis = BATVisualizer(BATMakieVisualization())
    EXT._claim_live_visualizer!(vis)
    fig = Figure()
    ax, obs = setup(fig)
    good = spec([1.0, 2.0, 3.0])
    other = Figure()
    lines!(Axis(other[1, 1]), [0, 1], [1, 0])
    try
        @test EXT._with_live_view(vis) do
            EXT._attach_live_view!(vis, fig)
            EXT._update_grid_axis!(ax, obs, good, nothing)
        end
        EXT._start_live_listener!(vis, () -> nothing, () -> nothing)
        @test !EXT._with_live_view(vis) do
            EXT._update_grid_axis!(ax, obs, spec(1e100 .* [1, 2, 3]), good.kwargs)
        end
        BAT.finish_visualizer!(vis; success=true)
        status = BAT.visualizer_status(vis)
        @test status.outcome === :failed
        @test !status.data_complete && !status.presentation_valid
        @test status.quiesced && status.view_disposed && !status.cleanup_pending
        # Disposed history intentionally retains only bounded diagnostics.
        @test status.error.type == "ResolveException" && status.error.phase === :view &&
            status.error.message == "Exception details are not retained"
        @test isempty(fig.content)
        @test isempty(Makie.Observables.listeners(obs))
        @test !isempty(Makie.colorbuffer(other))
    finally
        BAT.request_visualizer_stop!(vis)
        BAT.finish_visualizer!(vis; success=false)
        empty!(fig)
        empty!(other)
    end
end
end # module
