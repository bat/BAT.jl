# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Optional native-window check. Run in an environment with BAT and GLMakie,
# with an external process timeout and a working display/OpenGL context.
using BAT, GLMakie, Makie, Test
const EXT = Base.get_extension(BAT, :BATMakieExt)
isnothing(EXT) && error("BATMakieExt failed to load")
realpath(dirname(dirname(pathof(BAT)))) == realpath(joinpath(@__DIR__, "../../..")) ||
    error("Native smoke test loaded a different BAT checkout")
GLMakie.activate!()
Makie.inline!(false)

@testset "Native GLMakie close is local and nonjoining" begin
    vis = BATVisualizer(BATMakieVisualization())
    screens = GLMakie.Screen[]
    try
        EXT._claim_live_visualizer!(vis)
        vis.content.n_dof[] = 3
        fig = Figure(size=(240, 160))
        @test EXT._with_live_view(vis) do
            EXT._attach_live_view!(vis, fig)
            lines!(Axis(fig[1, 1]), [0, 1], [0, 1])
            screen = GLMakie.Screen(; visible=false)
            push!(screens, screen)
            display(screen, fig)
        end
        other = Figure(size=(240, 160))
        lines!(Axis(other[1, 1]), [0, 1], [1, 0])
        push!(screens, GLMakie.Screen(; visible=false))
        display(screens[2], other)
        EXT._start_live_listener!(vis, () -> nothing, () -> nothing)
        @test events(fig).window_open[]
        @test isopen(screens[2])
        close(screens[1])
        BAT.finish_visualizer!(vis; success=true)
        status = BAT.visualizer_status(vis)
        @test status.reason === :closed
        @test status.outcome === :disabled
        @test !status.data_complete
        @test status.quiesced
        @test !status.cleanup_pending
        @test status.view_disposed
        @test isopen(screens[2])
        @test screens[2].scene === other.scene
        @test !isempty(other.content)
    finally
        BAT.request_visualizer_stop!(vis)
        BAT.finish_visualizer!(vis; success=false)
        BAT.dispose_visualizer!(vis)
        foreach(close, screens)
    end
end
