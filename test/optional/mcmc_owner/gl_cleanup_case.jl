# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Native rollback test: GLMakie can swallow an interrupted stop wait and clear
# its handle. BAT must still join that exact task before closing its resources.
using BAT, GLMakie, Makie, Random, Test
const EXT = Base.get_extension(BAT, :BATMakieExt)
@test realpath(pathof(BAT)) == realpath(joinpath(@__DIR__, "../../../src/BAT.jl"))
const CASE, GATES = ARGS
CASE == "gl_cleanup" || error("Unknown case: $CASE")
Base.exit_on_sigint(false)
const PRIMARY = ErrorException("intentional failure after GL enrollment")
const RELEASE = Channel{Nothing}(1)
const STARTUP = Ref{Union{Nothing,Task}}(nothing)
const RENDERER = Ref{Union{Nothing,Task}}(nothing)
const SCREEN = Ref{Any}(nothing)
const OWNER_RETURNED = Threads.Atomic{Bool}(false)
const LATE_WRITES = Threads.Atomic{Int}(0)
const OWNER = current_task()

function event(text)
    println("BAT_OWNER ", text)
    flush(stdout)
end

function EXT._enroll_gl_renderloop!(screen::GLMakie.Screen, backend::Module)
    STARTUP[] = current_task()
    SCREEN[] = screen
    invoke(EXT._enroll_gl_renderloop!, Tuple{Any,Any}, screen, backend)
    throw(PRIMARY)
end

# Narrow test-only screen-wait barrier. The real upstream stop routine calls
# this wait, catches the first owner interruption, and clears its task field.
# The renderer itself, Task wait, scheduler and BAT's retained-handle join are
# unchanged. Polling keeps this owner at a native interruptible wait; a bare
# wait(task) can deliver SIGINT to the last suspended renderer instead.
function Base.wait(screen::GLMakie.Screen{GLMakie.GLFW.Window})
    if current_task() === OWNER && screen.stop_renderloop[] &&
            !isnothing(RENDERER[]) && !istaskdone(RENDERER[])
        event("GL_CLEANUP_WAIT")
        while !isready(RELEASE)
            sleep(0.005)
        end
    end
    invoke(wait, Tuple{GLMakie.Screen}, screen)
end

@testset "Native interrupted GL cleanup joins the retained renderer handle" begin
    @test Threads.threadid() == 1
    @test timedwait(() -> false, 0.02; pollint=0.001) === :timed_out
    GLMakie.activate!(visible=false)
    Makie.inline!(false)
    fig = Figure(size=(240, 160))
    lines!(Axis(fig[1, 1]), [0, 1], [0, 1])
    renderloop = screen -> begin
        RENDERER[] = current_task()
        GLMakie.render_frame(screen)
        while !screen.stop_renderloop[]
            sleep(0.001)
        end
        take!(RELEASE)
        @test !OWNER_RETURNED[]
        # The first signal made GLMakie drop its handle; BAT's independent
        # retained handle must keep cleanup waiting for these final writes.
        @test screen.rendertask === nothing
        @test isopen(screen)
        @test !isempty(Makie.colorbuffer(screen))
        Threads.atomic_add!(LATE_WRITES, 1)
    end
    controller = Threads.@spawn begin
        @test Threads.threadid() != 1
        deadline = time() + 120
        try
            while !isfile(joinpath(GATES, "release_gl_cleanup"))
                time() < deadline || error("External driver did not release GL cleanup")
                sleep(0.005)
            end
            @test !OWNER_RETURNED[]
            @test RENDERER[] isa Task && !istaskdone(RENDERER[])
            @test LATE_WRITES[] == 0
            event("GL_CLEANUP_HELD")
        finally
            put!(RELEASE, nothing)
        end
    end
    before = copy(Random.default_rng())
    GLMakie.activate!(renderloop=renderloop, visible=false)
    error = try
        EXT._maybe_display(fig)
        nothing
    catch err
        err
    finally
        OWNER_RETURNED[] = true
    end
    after = copy(Random.default_rng())
    @test error === PRIMARY
    @test after == before
    @test STARTUP[] isa Task && istaskdone(STARTUP[]) && !istaskfailed(STARTUP[])
    @test RENDERER[] isa Task && istaskdone(RENDERER[]) && !istaskfailed(RENDERER[])
    @test LATE_WRITES[] == 1
    @test !isopen(SCREEN[])
    wait(controller)
    @test !istaskfailed(controller)
    @test LATE_WRITES[] == 1
    empty!(fig)
    event("RESULT " * CASE)
end
