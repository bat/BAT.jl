# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Isolated child only. Inject failure at the final ordinary-conversion boundary,
# after both static returns have been acquired by the actual workload.
module TestMakiePrecompileFailure
using BAT, Makie, Test, Logging
const EXT = Base.get_extension(BAT, :BATMakieExt)
const OBS = Makie.Observables
const LIVE_WIDGET_FAULT = Ref(false)
const WIDGET_FAILURE = ErrorException("intentional live widget failure")
const PRIMARY = ErrorException("intentional later precompile workload failure")
const CLEANUP = ErrorException("intentional static observer cleanup failure")
const LIVE = Ref{Any}(nothing)
const RETURNED = Makie.Figure[]
const LIVE_AT_RETURN = Any[]
const INJECT_CLEANUP = Ref(false)
const CLEANUP_ATTEMPTS = Ref(0)
const CONVERSION_REACHED = Ref(false)
const DIAGNOSTIC_ATTEMPTS = Ref(0)

# This callback belongs only to our first static figure. Makie's real Scene
# cleanup calls off(handle), then dispatches here. Throw only once so emergency
# test cleanup can finish after the production error-isolation assertions.
struct CleanupFault <: Function end
(::CleanupFault)(_) = nothing
function OBS.off(obs::OBS.AbstractObservable, callback::CleanupFault)
    CLEANUP_ATTEMPTS[] += 1
    CLEANUP_ATTEMPTS[] == 1 && throw(CLEANUP)
    return invoke(OBS.off, Tuple{OBS.AbstractObservable,Any}, obs, callback)
end

struct BrokenCleanupLogger <: Logging.AbstractLogger end
Logging.min_enabled_level(::BrokenCleanupLogger) = Logging.Warn
Logging.shouldlog(::BrokenCleanupLogger, level, mod, group, id) = true
Logging.catch_exceptions(::BrokenCleanupLogger) = false
function Logging.handle_message(::BrokenCleanupLogger, level, message, mod, group, id, file, line; kwargs...)
    if get(kwargs, :operation, nothing) == :precompile_static_figure
        DIAGNOSTIC_ATTEMPTS[] += 1
        throw(ErrorException("intentional cleanup diagnostic logger failure"))
    end
    return nothing
end

function EXT._attach_live_view!(vis::BAT.BATVisualizer, fig::Makie.Figure)
    LIVE[] = vis
    return invoke(EXT._attach_live_view!, Tuple{Any,Any}, vis, fig)
end
function EXT._drive_figure_widgets!(fig::Makie.Figure)
    if LIVE_WIDGET_FAULT[]
        slider = only(filter(x -> x isa Makie.IntervalSlider, fig.content))
        EXT._view_on(LIVE[], slider.interval) do _
            throw(WIDGET_FAILURE)
        end
    end
    return invoke(EXT._drive_figure_widgets!, Tuple{Any}, fig)
end
function BAT.bat_makie_plot(samples::BAT.DensitySampleVector{<:NamedTuple}; kwargs...)
    fig = invoke(BAT.bat_makie_plot, Tuple{BAT.DensitySampleVector}, samples; kwargs...)
    push!(RETURNED, fig)
    push!(LIVE_AT_RETURN, BAT.visualizer_figure(LIVE[]))
    if INJECT_CLEANUP[] && length(RETURNED) == 1
        handle = Makie.on(CleanupFault(), Makie.Observable(0))
        push!(fig.scene.deregister_callbacks, handle)
    end
    return fig
end
function Makie.convert_arguments(::Type{<:Makie.Plot{Makie.plot}},
        ::BAT.DensitySampleVector{<:NamedTuple}; kwargs...)
    CONVERSION_REACHED[] = true
    throw(PRIMARY)
end

@testset "Real Makie precompile workload preserves failure and releases ownership" begin
    for cleanup_fault in (false, true)
        @testset "cleanup fault = $cleanup_fault" begin
            empty!(RETURNED)
            empty!(LIVE_AT_RETURN)
            INJECT_CLEANUP[] = cleanup_fault
            CONVERSION_REACHED[] = false
            EXT._SUPPRESS_DISPLAY[] = cleanup_fault
            caught = try
                Logging.with_logger(BrokenCleanupLogger()) do
                    EXT._makie_precompile_workload()
                end
                nothing
            catch err
                err
            end
            @test caught === PRIMARY
            @test CONVERSION_REACHED[]
            @test length(RETURNED) == 2
            @test RETURNED[1] !== RETURNED[2]
            @test LIVE_AT_RETURN[1] === LIVE_AT_RETURN[2]
            @test all(fig -> fig !== LIVE_AT_RETURN[1], RETURNED)
            @test isempty(RETURNED[2].content)
            if cleanup_fault
                @test CLEANUP_ATTEMPTS[] == 1
                @test DIAGNOSTIC_ATTEMPTS[] == 1
            else
                @test isempty(RETURNED[1].content)
                @test CLEANUP_ATTEMPTS[] == 0
                @test DIAGNOSTIC_ATTEMPTS[] == 0
            end
            status = BAT.visualizer_status(LIVE[])
            @test status.view_disposed
            @test status.quiesced
            @test !status.cleanup_pending
            @test isempty(LIVE_AT_RETURN[1].content)
            @test EXT._SUPPRESS_DISPLAY[] === cleanup_fault
            @test BAT.visualizer_figure(LIVE[]) === nothing
            # A cleanup attempt that throws does not promise full destruction.
            # Release anything the deliberately broken observer left behind.
            foreach(empty!, RETURNED)
            EXT._SUPPRESS_DISPLAY[] = false
        end
    end
end

@testset "Guarded live widget failure cannot pass precompile coverage" begin
    empty!(RETURNED)
    CONVERSION_REACHED[] = false
    LIVE_WIDGET_FAULT[] = true
    caught = try
        EXT._makie_precompile_workload()
        nothing
    catch err
        err
    end
    @test caught isa ErrorException
    @test occursin("live Figure became unavailable", sprint(showerror, caught))
    @test isempty(RETURNED)
    @test !CONVERSION_REACHED[]
    status = BAT.visualizer_status(LIVE[])
    @test status.view_disposed
    @test status.quiesced
    @test !status.cleanup_pending
    @test isnothing(BAT.visualizer_figure(LIVE[]))
    @test EXT._SUPPRESS_DISPLAY[] === false
end
end
