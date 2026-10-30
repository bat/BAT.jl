# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Run in its own bounded optional-runner child: the narrower methods below
# observe real workload boundaries and invoke their production implementations.
module TestMakiePrecompileWorkload
using BAT, Makie, Test
const EXT = Base.get_extension(BAT, :BATMakieExt)
const LIVE = Ref{Any}(nothing)
const DISPLAYED = Makie.Figure[]
const RETURNED = Makie.Figure[]
const DRIVEN = Makie.Figure[]
const LIVE_AT_RETURN = Any[]
const STATIC_OPTIONS = NamedTuple[]
const CONTENT_AT_RETURN = Int[]
const DARK_MENU_CHANGES = Ref(0)
const CONVERSIONS = Ref(0)
const WINDOW_REQUESTS = Any[]
const WINDOW_CHANGES = Any[]
const WINDOW_HANDLES = IdDict{Any,Any}()

function EXT._apply_sample_window!(graph::EXT.ComputeGraph, window; live_vis=nothing)
    live = !isnothing(live_vis)
    if !haskey(WINDOW_HANDLES, graph)
        @test isnothing(graph[:window_steps][])
        WINDOW_HANDLES[graph] = Makie.on(graph[:window_steps]) do value
            push!(WINDOW_CHANGES, (live, value))
        end
    end
    push!(WINDOW_REQUESTS, (live, window))
    return invoke(EXT._apply_sample_window!, Tuple{Any,Any}, graph, window; live_vis=live_vis)
end

function EXT._attach_live_view!(vis::BAT.BATVisualizer, fig::Makie.Figure)
    LIVE[] = vis
    return invoke(EXT._attach_live_view!, Tuple{Any,Any}, vis, fig)
end
function EXT._maybe_display(fig::Makie.Figure; kwargs...)
    push!(DISPLAYED, fig)
    return invoke(EXT._maybe_display, Tuple{Any}, fig; kwargs...)
end
function EXT._drive_figure_widgets!(fig::Makie.Figure)
    push!(DRIVEN, fig)
    return invoke(EXT._drive_figure_widgets!, Tuple{Any}, fig)
end
function BAT.bat_makie_plot(samples::BAT.DensitySampleVector{<:NamedTuple}; kwargs...)
    fig = invoke(BAT.bat_makie_plot, Tuple{BAT.DensitySampleVector}, samples; kwargs...)
    push!(RETURNED, fig)
    push!(LIVE_AT_RETURN, BAT.visualizer_figure(LIVE[]))
    push!(STATIC_OPTIONS, (; kwargs...))
    push!(CONTENT_AT_RETURN, length(fig.content))
    if get(kwargs, :dark, false)
        for menu in filter(x -> x isa Makie.Menu, fig.content)
            Makie.on(menu.i_selected) do _
                DARK_MENU_CHANGES[] += 1
            end
        end
    end
    return fig
end
function Makie.convert_arguments(kind::Type{<:Makie.Plot{Makie.plot}},
        samples::BAT.DensitySampleVector{<:NamedTuple}; kwargs...)
    CONVERSIONS[] += 1
    return invoke(Makie.convert_arguments,
        Tuple{Type{<:Makie.Plot{Makie.plot}},BAT.DensitySampleVector}, kind, samples; kwargs...)
end

@testset "Real Makie precompile workload owns its returned figures" begin
    @test EXT._makie_precompile_workload() === nothing
    @test length(RETURNED) == 2
    @test length(DISPLAYED) == 1
    @test length(DRIVEN) == 2
    @test all(>(0), CONTENT_AT_RETURN)
    @test all(options -> options.display === false, STATIC_OPTIONS)
    @test !get(STATIC_OPTIONS[1], :dark, false)
    @test STATIC_OPTIONS[2].dark
    @test RETURNED[1] !== RETURNED[2]
    @test all(fig -> fig !== only(DISPLAYED), RETURNED)
    @test DRIVEN[1] === only(DISPLAYED)
    @test DRIVEN[2] === RETURNED[1]
    @test all(fig -> fig === only(DISPLAYED), LIVE_AT_RETURN)
    # Observe actual graph publications, including the deferred live updates.
    @test length(WINDOW_REQUESTS) == 4
    @test WINDOW_CHANGES == WINDOW_REQUESTS
    @test first.(WINDOW_REQUESTS) == [true, true, false, false]
    @test all(i -> WINDOW_REQUESTS[i][2] isa Tuple{Int64,Int64}, (1, 3))
    @test all(i -> isnothing(WINDOW_REQUESTS[i][2]), (2, 4))
    @test all(graph -> isnothing(graph[:window_steps][]), keys(WINDOW_HANDLES))
    foreach(Makie.off, values(WINDOW_HANDLES))
    @test DARK_MENU_CHANGES[] > 0
    @test CONVERSIONS[] == 1
    @test all(fig -> isempty(fig.content), RETURNED)
    @test all(fig -> isempty(fig.scene.plots) && isempty(fig.scene.children), RETURNED)
    status = BAT.visualizer_status(LIVE[])
    @test status.view_disposed
    @test status.quiesced
    @test !status.cleanup_pending
    @test isempty(only(DISPLAYED).content)
    @test EXT._SUPPRESS_DISPLAY[] === false
    @test BAT.visualizer_figure(LIVE[]) === nothing
end
end
