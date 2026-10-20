# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Real native screens, in a separate externally bounded process. The narrow
# methods below observe production resource boundaries; they do not substitute
# a renderer, Task constructor, scheduler, RNG, or numerical algorithm.
using BAT, GLMakie, Makie, Test, Random
const EXT = Base.get_extension(BAT, :BATMakieExt)
realpath(dirname(dirname(pathof(BAT)))) == realpath(joinpath(@__DIR__, "../../..")) ||
    error("Native RNG test loaded a different BAT checkout")
const SCREENS = GLMakie.Screen[]
const SCREEN_TASKS = Any[]
const STARTUPS = Task[]
const STARTUP_THREADS = Int[]
const FIGURES = Figure[]
const STARTUP_FAULT = Ref(:none)
const PRIMARY = ErrorException("intentional renderer startup failure")

function EXT._with_gl_screen(f::Function, fig::Figure, backend::Module; temporary=false, on_failure=nothing)
    wrapped = function(screen)
        push!(SCREENS, screen)
        try
            f(screen)
        finally
            push!(SCREEN_TASKS, screen.rendertask)
        end
    end
    return invoke(EXT._with_gl_screen, Tuple{Any,Any,Any}, wrapped, fig, backend;
        temporary=temporary, on_failure=on_failure)
end

function EXT._enroll_gl_renderloop!(screen::GLMakie.Screen, backend::Module)
    push!(STARTUPS, current_task())
    push!(STARTUP_THREADS, Threads.threadid())
    STARTUP_FAULT[] === :before && throw(PRIMARY)
    invoke(EXT._enroll_gl_renderloop!, Tuple{Any,Any}, screen, backend)
    STARTUP_FAULT[] === :after && throw(PRIMARY)
    return nothing
end

function figure()
    fig = Figure(size=(260, 180))
    push!(FIGURES, fig) # Retain fixtures; automatic Scene finalizers are separate.
    lines!(Axis(fig[1, 1]), [0.0, 1.0], [0.0, 1.0])
    return fig
end

function children_from(state)
    saved = copy(Random.default_rng())
    try
        copy!(Random.default_rng(), state)
        children = [@task rand(UInt64, 8) for _ in 1:3]
        foreach(schedule, children)
        return fetch.(children)
    finally
        copy!(Random.default_rng(), saved)
    end
end

function same_streams(before, after)
    @test before == after
    @test rand(copy(before), UInt64, 16) == rand(copy(after), UInt64, 16)
    @test children_from(before) == children_from(after)
end

function unchanged(f)
    before = copy(Random.default_rng())
    result = f()
    after = copy(Random.default_rng())
    same_streams(before, after)
    return result
end

function callback_work()
    a = rand(UInt64)
    task = @task rand(UInt64, 8)
    schedule(task)
    child = fetch(task)
    return (a=a, child=child, b=rand(UInt64))
end

mutable struct InlineCapture <: AbstractDisplay
    owner::Task
    calls::Vector{Any}
    fail::Bool
end
Base.displayable(::InlineCapture, ::MIME"image/png") = true
function Base.display(c::InlineCapture, ::Figure)
    @test current_task() === c.owner
    push!(c.calls, callback_work())
    c.fail && throw(PRIMARY)
    return nothing
end

function close_figure(fig)
    EXT._close_owned_live_screens!(fig)
    empty!(fig)
end

GLMakie.activate!(visible=false)
Makie.inline!(false)
# Explicitly establish the ordinary-wait runtime condition before checkpoints.
# This does not create a renderer or restore any production RNG operation.
@test timedwait(() -> false, 0.02; pollint=0.001) === :timed_out

try
    @testset "Native warmup, display, reuse and ownership" begin
        @test isempty(EXT._WARMED_BACKENDS)
        other = figure()
        # Also keep an unrelated singleton screen alive: warmup/display must
        # not take it over or stop its render loop.
        display(other)
        other_screen = Makie.getscreen(other.scene)
        other_task = other_screen.rendertask
        count = length(STARTUPS)
        unchanged(EXT.warmup_makie_shaders)
        @test length(STARTUPS) == count
        warmup_screen = last(SCREENS)
        @test last(SCREEN_TASKS) === nothing
        @test !isopen(warmup_screen)
        @test isopen(other_screen) && other_screen.scene === other.scene
        @test other_screen.rendertask === other_task && !istaskdone(other_task)
        screen_count = length(SCREENS)
        unchanged(EXT.warmup_makie_shaders)
        @test length(SCREENS) == screen_count

        fig = figure()
        unchanged(() -> EXT._maybe_display(fig))
        screen = Makie.getscreen(fig.scene)
        renderer = screen.rendertask
        @test isopen(screen) && GLMakie.renderloop_running(screen)
        @test screen !== other_screen
        @test !isempty(Makie.colorbuffer(screen))
        @test length(STARTUPS) == count + 1
        @test istaskdone(last(STARTUPS)) && !istaskfailed(last(STARTUPS))
        @test last(STARTUP_THREADS) == 1
        unchanged(() -> EXT._maybe_display(fig))
        @test screen.rendertask === renderer
        @test length(STARTUPS) == count + 1
        close(screen)
        @test istaskdone(renderer) && !istaskfailed(renderer)
        unchanged(() -> EXT._maybe_display(fig))
        reopened = Makie.getscreen(fig.scene)
        @test isopen(reopened) && GLMakie.renderloop_running(reopened)
        @test isopen(other_screen) && other_screen.rendertask === other_task
        close_figure(fig)
        close_figure(other)
    end

    @testset "Synchronous callback effects survive success and failure" begin
        for fail in (false, true)
            fig = figure()
            owner = current_task()
            calls = Any[]
            on(events(fig).window_open) do open
                if open
                    @test current_task() === owner
                    push!(calls, callback_work())
                    fail && throw(PRIMARY)
                end
                return Consume(false)
            end
            Random.seed!(0x483331)
            before = copy(Random.default_rng())
            expected_call = callback_work()
            expected = copy(Random.default_rng())
            copy!(Random.default_rng(), before)
            error = try
                EXT._maybe_display(fig)
                nothing
            catch err
                err
            end
            after = copy(Random.default_rng())
            @test error === (fail ? PRIMARY : nothing)
            @test calls == [expected_call]
            same_streams(expected, after)
            screen = last(SCREENS)
            if fail
                @test !isopen(screen)
                @test isnothing(screen.rendertask)
            else
                @test GLMakie.renderloop_running(screen)
            end
            close_figure(fig)
        end
    end

    @testset "Inline display keeps caller callbacks and settings" begin
        for fail in (false, true)
            fig = figure()
            capture = InlineCapture(current_task(), [], fail)
            Makie.inline!(true)
            pushdisplay(capture)
            try
                Random.seed!(0x483332)
                before = copy(Random.default_rng())
                expected_call = callback_work()
                expected = copy(Random.default_rng())
                copy!(Random.default_rng(), before)
                error = try
                    EXT._maybe_display(fig)
                    nothing
                catch err
                    err
                end
                after = copy(Random.default_rng())
                @test error === (fail ? PRIMARY : nothing)
                @test capture.calls == [expected_call]
                same_streams(expected, after)
                @test Makie.ALWAYS_INLINE_PLOTS[] === true
                @test Makie.current_backend() === GLMakie
                fail && @test !isopen(last(SCREENS))
            finally
                popdisplay(capture)
                Makie.inline!(false)
                close_figure(fig)
            end
        end
    end

    @testset "Startup failure joins before closing its screen" begin
        for fault in (:before, :after)
            fig = figure()
            STARTUP_FAULT[] = fault
            before = copy(Random.default_rng())
            error = try
                EXT._maybe_display(fig)
                nothing
            catch err
                err
            finally
                STARTUP_FAULT[] = :none
            end
            after = copy(Random.default_rng())
            @test error === PRIMARY
            same_streams(before, after)
            @test istaskdone(last(STARTUPS))
            @test !isopen(last(SCREENS))
            task = last(SCREEN_TASKS)
            @test isnothing(task) || istaskdone(task)
            close_figure(fig)
        end
    end

    @testset "Public static rollback releases BAT's construction theme lock" begin
        samples = DensitySampleVector(v=[[sin(i)] for i in 1:40],
            logd=zeros(40), weight=ones(40))
        entered = Ref(false)
        renderloop = screen -> begin
            with_theme() do
                entered[] = true
            end
            GLMakie.renderloop(screen)
        end
        GLMakie.activate!(renderloop=renderloop, visible=false)
        STARTUP_FAULT[] = :after
        error = try
            bat_makie_plot(samples, (upper=Hist2D, diagonal=Hist1D, lower=Hist2D), [1], 1)
            nothing
        catch err
            err
        finally
            STARTUP_FAULT[] = :none
            GLMakie.activate!(renderloop=GLMakie.renderloop, visible=false)
        end
        @test error === PRIMARY
        @test entered[]
        @test istaskdone(last(STARTUPS))
        @test !isopen(last(SCREENS))
        @test istaskdone(last(SCREEN_TASKS))
    end

    @testset "Custom renderloop runs in its own main-thread task" begin
        fig = figure()
        owner = current_task()
        seen = Ref{Any}(nothing)
        renderloop = screen -> begin
            seen[] = (task=current_task(), thread=Threads.threadid(), work=callback_work())
            GLMakie.renderloop(screen)
        end
        with_theme(GLMakie=(renderloop=renderloop, visible=false)) do
            unchanged(() -> EXT._maybe_display(fig))
        end
        @test seen[] !== nothing
        @test seen[].task !== owner
        @test seen[].thread == 1
        @test seen[].task === Makie.getscreen(fig.scene).rendertask
        close_figure(fig)
        @test istaskdone(seen[].task)
    end

    @testset "Immediate renderer failure retains its cause" begin
        fig = figure()
        renderer = Ref{Union{Nothing,Task}}(nothing)
        renderloop = screen -> begin
            renderer[] = current_task()
            throw(PRIMARY)
        end
        before = copy(Random.default_rng())
        error = try
            with_theme(GLMakie=(renderloop=renderloop, visible=false)) do
                EXT._maybe_display(fig)
            end
            nothing
        catch err
            err
        end
        after = copy(Random.default_rng())
        @test error isa TaskFailedException
        @test error.task === renderer[]
        @test renderer[].result === PRIMARY
        @test istaskdone(renderer[]) && istaskfailed(renderer[])
        @test !isopen(last(SCREENS))
        same_streams(before, after)
        close_figure(fig)
    end

    @testset "Closing during scene attachment does not start a loop" begin
        fig = figure()
        starts = length(STARTUPS)
        on(events(fig).window_open) do open
            open && close(last(SCREENS))
            return Consume(false)
        end
        unchanged(() -> EXT._maybe_display(fig))
        @test !isopen(last(SCREENS))
        @test length(STARTUPS) == starts
        @test isnothing(last(SCREENS).rendertask)
        close_figure(fig)
    end

    @testset "Off-thread GL display rejects before acquiring a screen" begin
        fig = figure()
        count = length(SCREENS)
        error = fetch(Threads.@spawn try
            EXT._maybe_display(fig)
            nothing
        catch err
            err
        end)
        @test error isa ArgumentError
        @test length(SCREENS) == count
        @test isempty(fig.scene.current_screens)
        close_figure(fig)
    end
finally
    STARTUP_FAULT[] = :none
    foreach(close, filter(isopen, unique(SCREENS)))
    foreach(close_figure, FIGURES)
end
