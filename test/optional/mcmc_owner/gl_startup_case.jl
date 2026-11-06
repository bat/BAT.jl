# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Opt-in native GL test, owned and externally bounded by runtests.py.
# Hold the actual startup worker before GL enrollment. Native SIGINT can reach
# this worker or its waiting owner; verify that both paths are joined/reported.
using BAT, GLMakie, Makie, Random, Test, Distributions, Logging
const EXT = Base.get_extension(BAT, :BATMakieExt)
@test realpath(pathof(BAT)) == realpath(joinpath(@__DIR__, "../../../src/BAT.jl"))
const CASE, GATES = ARGS
CASE == "gl_startup" || error("Unknown case: $CASE")
Base.exit_on_sigint(false)

const RELEASE = Channel{Nothing}(1)
const STARTUP = Ref{Union{Nothing,Task}}(nothing)
const OWNER_RNG = Ref{Any}(nothing)
const SCREENS = GLMakie.Screen[]
const RENDERER = Ref{Union{Nothing,Task}}(nothing)
const WORKER_INTERRUPTED = Ref(false)
const OWNER_RETURNED = Threads.Atomic{Bool}(false)
const RENDER_CALLBACK_ENTERED = Ref(false)

function event(text)
    println("BAT_OWNER ", text)
    flush(stdout)
end

function EXT._with_gl_screen(f::Function, fig::Figure, backend::Module; temporary=false, on_failure=nothing)
    wrapped = screen -> begin
        push!(SCREENS, screen)
        f(screen)
    end
    invoke(EXT._with_gl_screen, Tuple{Any,Any,Any}, wrapped, fig, backend;
        temporary=temporary, on_failure=on_failure)
end

function EXT._run_gl_startup(screen::GLMakie.Screen, backend::Module)
    OWNER_RNG[] = copy(Random.default_rng())
    invoke(EXT._run_gl_startup, Tuple{Any,Any}, screen, backend)
end

function EXT._enroll_gl_renderloop!(screen::GLMakie.Screen, backend::Module)
    STARTUP[] = current_task()
    @test Threads.threadid() == 1
    event("GL_STARTUP_WAIT")
    # Hold the already owned worker BEFORE its protected backend enrollment.
    # Never enlarge the production SIGINT-deferred scope with a test barrier.
    try
        take!(RELEASE)
    catch err
        WORKER_INTERRUPTED[] = err isa InterruptException
        rethrow()
    end
    invoke(EXT._enroll_gl_renderloop!, Tuple{Any,Any}, screen, backend)
    RENDERER[] = screen.rendertask
    return nothing
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

@testset "Native interrupted GL startup remains joined and owned" begin
    @test Threads.threadid() == 1
    @test Threads.threadpool() === :interactive
    @test timedwait(() -> false, 0.02; pollint=0.001) === :timed_out
    GLMakie.activate!(visible=false)
    Makie.inline!(false)
    base = BATMakieVisualization(adaptive_batching=false)
    config = BATMakieVisualization((upper=Hist2D, diagonal=Hist1D, lower=Hist2D),
        [1], 1, 20_000, 40_000, false, base.batch_growth_rate, 0.01, base.dark,
        merge(base.triagonal_config, (nbins=(8, 8),)),
        merge(base.diagonal_config, (nbins=8,)); observer_timeout=30.0)
    vis = BATVisualizer(config)
    context = BATContext(rng=BAT.Philox4x((31, 31)), visualizer=vis)
    algorithm = TransformedMCMC(nchains=2, nsteps=16, strict=false,
        pretransform=DoNotTransform(), adaptive_transform=BAT.NoAdaptiveTransform(),
        transform_tuning=NoMCMCTransformTuning(), convergence=AssumeConvergence(),
        init=MCMCRetryInit(nsteps_init=12, max_init_tries=2, strict=false),
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=12, max_ncycles=1, nsteps_final=2))
    renderloop = screen -> begin
        RENDER_CALLBACK_ENTERED[] = EXT._with_live_view(vis) do
            @test vis.content.lifecycle.initializing
            with_theme() do
                @test Threads.threadid() == 1
            end
        end
        GLMakie.renderloop(screen)
    end
    GLMakie.activate!(renderloop=renderloop, visible=false)
    # The release is necessary when the owner receives the signal: its real
    # finally block must wait for this still-held worker. If the worker receives
    # it instead, the worker has already terminated and no renderer was started.
    controller = Threads.@spawn begin
        @test Threads.threadid() != 1
        deadline = time() + 120
        try
            while !isfile(joinpath(GATES, "release_gl_startup"))
                time() < deadline || error("External driver did not release GL startup")
                sleep(0.005)
            end
            if !istaskdone(STARTUP[])
                @test !OWNER_RETURNED[]
                event("GL_STARTUP_HELD")
            end
        finally
            put!(RELEASE, nothing)
        end
    end
    error = nothing
    try
        with_logger(NullLogger()) do
            BAT.evalmeasure(MvNormal([0.0], ones(1, 1)), algorithm, context)
        end
    catch err
        error = err
    finally
        OWNER_RETURNED[] = true
    end
    after = copy(Random.default_rng())
    @test error isa InterruptException
    @test STARTUP[] isa Task && istaskdone(STARTUP[])
    @test !istaskfailed(STARTUP[])
    if WORKER_INTERRUPTED[]
        event("GL_STARTUP_RECIPIENT worker")
        @test isnothing(RENDERER[])
    else
        event("GL_STARTUP_RECIPIENT owner")
        @test RENDER_CALLBACK_ENTERED[]
        @test RENDERER[] isa Task && istaskdone(RENDERER[]) && !istaskfailed(RENDERER[])
    end
    @test isnothing(last(SCREENS).rendertask)
    @test !isopen(last(SCREENS))
    @test after == OWNER_RNG[]
    @test children_from(after) == children_from(OWNER_RNG[])
    status = BAT.visualizer_status(vis)
    @test status.quiesced && status.history_released && !status.cleanup_pending
    @test status.outcome === :failed
    @test isnothing(vis.content.graph[])
    @test isnothing(vis.content.lifecycle.task)
    @test isnothing(vis.content.lifecycle.disposal_task)
    wait(controller)
    @test !istaskfailed(controller)
    @test all(s -> !isopen(s), SCREENS)
    event("RESULT " * CASE)
end
