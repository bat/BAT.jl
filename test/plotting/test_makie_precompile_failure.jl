# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Isolated child only. Inject failure at the final ordinary-conversion boundary,
# after both static returns have been acquired by the actual workload.
module TestMakiePrecompileFailure
using BAT, Makie, Test, Logging
const EXT = Base.get_extension(BAT, :BATMakieExt)
const OBS = Makie.Observables
const PRIMARY = ErrorException("intentional later precompile workload failure")
const CLEANUP = ErrorException("intentional static observer cleanup failure")
const LIVE = Ref{Any}(nothing)
const RETURNED = Makie.Figure[]
const CAPTURE_AT_RETURN = Any[]
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
function BAT.bat_makie_plot(samples::BAT.DensitySampleVector{<:NamedTuple}; kwargs...)
    fig = invoke(BAT.bat_makie_plot, Tuple{BAT.DensitySampleVector}, samples; kwargs...)
    push!(RETURNED, fig)
    push!(CAPTURE_AT_RETURN, EXT._SUPPRESSED_FIG[])
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
            empty!(CAPTURE_AT_RETURN)
            INJECT_CLEANUP[] = cleanup_fault
            CONVERSION_REACHED[] = false
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
            @test CAPTURE_AT_RETURN[1] === CAPTURE_AT_RETURN[2]
            @test all(fig -> fig !== CAPTURE_AT_RETURN[1], RETURNED)
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
            @test isempty(CAPTURE_AT_RETURN[1].content)
            @test EXT._SUPPRESS_DISPLAY[] === false
            @test EXT._SUPPRESSED_FIG[] === nothing
            # A cleanup attempt that throws does not promise full destruction.
            # Release anything the deliberately broken observer left behind.
            foreach(empty!, RETURNED)
        end
    end
end
end
