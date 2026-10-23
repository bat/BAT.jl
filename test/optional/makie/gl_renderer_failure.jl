# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Native MR06 regression gate. Run in its own externally bounded process with
# GLMakie and a working display/OpenGL context. Failures below originate in
# actual native render ticks, including after sampling has retired its listener.
using BAT, GLMakie, Makie, Distributions, Random, Test, Logging
const EXT = Base.get_extension(BAT, :BATMakieExt)
isnothing(EXT) && error("BATMakieExt failed to load")
realpath(dirname(dirname(pathof(BAT)))) == realpath(joinpath(@__DIR__, "../../..")) ||
    error("Native renderer failure test loaded a different BAT checkout")

const PRIMARY = ErrorException("intentional native render-tick failure")
const ENROLL_FAULT = Ref(false)
const RELEASERS = IdDict{Any,Task}()
const CLEANUP_CALLS = NamedTuple{(:thread, :self_join),Tuple{Int,Bool}}[]

# These narrow methods observe real production entry points, then invoke their
# generic implementation. Retain disposal task handles until assertions finish:
# a public status read can retire them. The startup seam injects only a failure
# before native invocation entry; no renderer, scheduler or RNG is substituted.
function EXT._coordinate_live_disposal!(vis::BATVisualizer{BATMakieVisualization})
    RELEASERS[vis.content.lifecycle] = current_task()
    return invoke(EXT._coordinate_live_disposal!, Tuple{Any}, vis)
end

function EXT._close_gl_screen!(screen::GLMakie.Screen, backend::Module)
    push!(CLEANUP_CALLS, (thread=Threads.threadid(), self_join=current_task() === screen.rendertask))
    return invoke(EXT._close_gl_screen!, Tuple{Any,Any}, screen, backend)
end

function EXT._enroll_gl_renderloop!(screen::GLMakie.Screen, backend::Module)
    ENROLL_FAULT[] && throw(PRIMARY)
    return invoke(EXT._enroll_gl_renderloop!, Tuple{Any,Any}, screen, backend)
end

function algorithm(callback=BAT.nop_func)
    return TransformedMCMC(nchains=2, nsteps=12, strict=false, store_burnin=true,
        init=MCMCChainPoolInit(nsteps_init=20), convergence=AssumeConvergence(),
        adaptive_transform=BAT.NoAdaptiveTransform(), transform_tuning=NoMCMCTransformTuning(),
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=8, max_ncycles=1, nsteps_final=2),
        callback=callback)
end

# Independent snapshots include numerical state and each chain/walker RNG;
# equality of returned samples alone would miss changes to future work.
function snapshot(chain)
    return (current=deepcopy(chain.current), proposed=deepcopy(chain.proposed),
        output=deepcopy(chain.output), accepted=copy(chain.accepted), order=copy(chain.walker_order),
        info=chain.info, stepno=chain.stepno, nattempts=copy(chain.nattempts),
        nsamples=copy(chain.nsamples), rng=copy(BAT.get_rng(chain.context)),
        walker_rngs=[copy(BAT.get_rng(c)) for c in chain.walker_genctxs])
end

function future_children(state)
    saved = copy(Random.default_rng())
    try
        copy!(Random.default_rng(), state)
        tasks = [@task rand(UInt64, 4) for _ in 1:2]
        foreach(schedule, tasks)
        return fetch.(tasks)
    finally
        copy!(Random.default_rng(), saved)
    end
end

function done(task; failed=false)
    @test task isa Task
    task isa Task || error("Missing owned task")
    finished = timedwait(() -> istaskdone(task), 30.0; pollint=0.005) === :ok
    @test finished
    finished || error("Owned task did not terminate")
    @test istaskfailed(task) === failed
    return nothing
end

make_vis() = BATVisualizer(BATMakieVisualization(adaptive_batching=false, observer_timeout=30.0))
live_context(vis) = BATContext(rng=BAT.Philox4x((912, 71)), visualizer=vis)
const TARGET = MvNormal(zeros(2), ones(2))
sample_live(vis; logger=NullLogger()) =
    with_logger(() -> BAT.evalmeasure(TARGET, algorithm(), live_context(vis)), logger)

function fail_tick(screen)
    task = screen.rendertask
    @test task isa Task && !istaskdone(task)
    handle = Makie.on(screen.render_tick; priority=100) do _
        throw(PRIMARY)
    end
    return task, handle
end

function await_release(vis; failed=true)
    ctl = vis.content.lifecycle
    # No status/getter/dispose call drives this wait. The already owned
    # coordinator must commit history release autonomously before public reads.
    release = get(RELEASERS, ctl, ctl.disposal_task)
    done(release)
    @test ctl.release_committed && isnothing(vis.content.graph[])
    @test isnothing(ctl.renderer_owner[])
    status = BAT.visualizer_status(vis)
    @test status.quiesced && status.history_released && status.view_disposed
    @test !status.cleanup_pending
    @test !failed || !status.presentation_valid
    @test isnothing(status.cleanup_error)
    @test isnothing(BAT.visualizer_figure(vis))
    pop!(RELEASERS, ctl, nothing)
    return status
end

# Finally cleanup must also work when an earlier assertion throws. Its waits
# are bounded and use owned task handles, without adding assertions that vary
# with an incidental scheduling order or Julia's internal test accounting.
function dispose_fixture(vis)
    BAT.dispose_visualizer!(vis)
    ctl = vis.content.lifecycle
    task = get(RELEASERS, ctl, ctl.disposal_task)
    if task isa Task
        timedwait(() -> istaskdone(task), 30.0; pollint=0.005) === :ok ||
            error("Fixture disposal timed out")
    end
    pop!(RELEASERS, ctl, nothing)
    return nothing
end

function close_figure(fig)
    EXT._close_owned_live_screens!(fig)
    empty!(fig)
    return nothing
end

function small_figure()
    fig = Figure(size=(160, 120))
    lines!(Axis(fig[1, 1]), [0.0, 1.0], [0.0, 1.0])
    return fig
end

function stopped_screen(loop=GLMakie.renderloop)
    fig = small_figure()
    screen = GLMakie.Screen(; start_renderloop=false, visible=false, renderloop=loop)
    try
        display(screen, fig)
        return fig, screen
    catch
        close(screen)
        empty!(fig)
        rethrow()
    end
end

mutable struct PNGDisplay <: AbstractDisplay
    calls::Int
    bytes::Int
    valid_png::Bool
end
Base.displayable(::PNGDisplay, ::MIME"image/png") = true
function Base.display(d::PNGDisplay, fig::Figure)
    d.calls += 1
    io = IOBuffer()
    show(io, MIME"image/png"(), fig)
    bytes = take!(io)
    d.bytes += length(bytes)
    d.valid_png &= length(bytes) >= 8 && bytes[1:8] == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
    return nothing
end

mutable struct YieldLogger <: AbstractLogger
    vis::Any
    hit::Bool
    release::Bool
    snapshot::Any
end
Logging.min_enabled_level(::YieldLogger) = Logging.Debug
Logging.shouldlog(::YieldLogger, args...) = true
Logging.catch_exceptions(::YieldLogger) = false
function Logging.handle_message(logger::YieldLogger, level, message, args...; kwargs...)
    if message isa String && startswith(message, "Completed visualization failed")
        ctl = logger.vis.content.lifecycle
        logger.snapshot = (valid=ctl.view.presentation_valid[], requested=ctl.disposal_requested,
            task=ctl.disposal_task, figure=BAT.visualizer_figure(logger.vis))
        logger.hit = true
        timedwait(() -> logger.release, 20.0; pollint=0.005) === :ok ||
            error("Logger release timed out")
    end
    return nothing
end

# Keep the returned callback alive while allowing its isolated owner to die.
function weak_callback_fixture()
    vis = make_vis()
    callback = EXT._live_renderer_failure_callback(vis)
    return callback, WeakRef(vis.content.lifecycle.renderer_owner), WeakRef(vis.content.lifecycle)
end

# Isolate all old-session locals from the GC assertion in the calling frame.
function replaced_session()
    vis = make_vis()
    replacement = small_figure()
    success = false
    try
        sample_live(vis)
        fig = BAT.visualizer_figure(vis)
        screen = Makie.getscreen(fig.scene, GLMakie)
        weak_graph = WeakRef(vis.content.graph[])
        display(screen, replacement)
        # Inherited GLMakie/Makie screen-reuse behavior retains an old Scene
        # through window-open subscriptions. Record it as a limitation, not an
        # automatic-GC contract or a reason to mutate upstream subscriptions.
        retained_scene = any(screen.window_open.listeners) do pair
            callback = pair[2]
            any(name -> isdefined(callback, name) && getfield(callback, name) === fig.scene,
                fieldnames(typeof(callback)))
        end
        BAT.dispose_visualizer!(vis)
        status = await_release(vis; failed=false)
        @test status.outcome === :complete && status.data_complete && isnothing(status.error)
        success = true
        return screen, replacement, weak_graph, screen.rendertask, retained_scene
    finally
        dispose_fixture(vis)
        success || close_figure(replacement)
    end
end

GLMakie.activate!(visible=false, renderloop=GLMakie.renderloop)
Makie.inline!(false)
# Ordinary wait initialization is outside all sampling RNG checkpoints.
@test timedwait(() -> false, 0.02; pollint=0.001) === :timed_out

try
    EXT.warmup_makie_shaders()
    @testset "Native late failure during sampling preserves fixed-work results" begin
        vis = make_vis()
        other = small_figure()
        handle = Ref{Any}(nothing)
        renderer = Ref{Any}(nothing)
        listener = Ref{Any}(nothing)
        plain_callbacks, live_callbacks = Any[], Any[]
        plain_context = BATContext(rng=BAT.Philox4x((912, 71)))
        context = live_context(vis)
        plain_cb = (tag, states) -> push!(plain_callbacks,
            (tag=tag, states=snapshot.(getproperty.(states, :chain_state))))
        live_cb = function(tag, states)
            push!(live_callbacks, (tag=tag, states=snapshot.(getproperty.(states, :chain_state))))
            if isnothing(renderer[])
                fig = vis.content.lifecycle.view.figure
                screen = Makie.getscreen(fig.scene, GLMakie)
                @test screen.config.renderloop === GLMakie.renderloop
                listener[] = vis.content.lifecycle.task
                renderer[], handle[] = fail_tick(screen)
                done(renderer[]; failed=true)
                @test renderer[].result === PRIMARY
                Makie.off(handle[])
                handle[] = nothing
                @test !vis.content.lifecycle.accepting[]
                @test !vis.content.lifecycle.view.presentation_valid[]
            end
        end
        try
            display(other)
            other_screen = Makie.getscreen(other.scene, GLMakie)
            other_task = other_screen.rendertask
            Random.seed!(117)
            plain = with_logger(() -> BAT.evalmeasure(TARGET, algorithm(plain_cb), plain_context), NullLogger())
            default_plain = copy(Random.default_rng())
            Random.seed!(117)
            live = with_logger(() -> BAT.evalmeasure(TARGET, algorithm(live_cb), context), NullLogger())
            default_live = copy(Random.default_rng())
            @test BAT.samplesof(live) == BAT.samplesof(plain)
            @test snapshot.(BAT.samplegenof(live).chain_states) == snapshot.(BAT.samplegenof(plain).chain_states)
            @test !isempty(plain_callbacks) && plain_callbacks == live_callbacks
            @test BAT.get_rng(context) == BAT.get_rng(plain_context)
            @test default_live == default_plain
            @test rand(copy(default_live), UInt64, 16) == rand(copy(default_plain), UInt64, 16)
            @test future_children(default_live) == future_children(default_plain)
            done(listener[])
            status = await_release(vis)
            @test status.outcome === :failed && !status.data_complete
            @test status.error == (type="ErrorException", message=PRIMARY.msg, phase=:renderloop)
            @test isopen(other_screen) && other_screen.scene === other.scene
            @test other_screen.rendertask === other_task && !istaskdone(other_task)
        finally
            isnothing(handle[]) || Makie.off(handle[])
            dispose_fixture(vis)
            close_figure(other)
        end
    end

    @testset "Completed figures fail autonomously with or without a held view" begin
        for blocked in (false, true)
            vis = make_vis()
            holder = nothing
            release_hold = Ref(false)
            handle = nothing
            try
                sample_live(vis)
                fig = BAT.visualizer_figure(vis)
                @test fig isa Figure
                ctl = vis.content.lifecycle
                view = ctl.view
                screen = Makie.getscreen(fig.scene, GLMakie)
                @test screen.config.renderloop === GLMakie.renderloop
                @test isnothing(ctl.task) && isnothing(ctl.timer)
                if blocked
                    # Hold real view admission over a yield. Failure reporting
                    # must not acquire it; its release supplies cleanup readiness.
                    holder = @async EXT._with_live_view(vis) do
                        timedwait(() -> release_hold[], 20.0; pollint=0.005) === :ok ||
                            error("View holder release timed out")
                    end
                    @test timedwait(() -> view.depth == 1, 5.0; pollint=0.005) === :ok
                end
                before = copy(Random.default_rng()) # caller-created holder is intentional
                anchor = ctl.renderer_owner
                GC.gc(true)
                GC.gc(true)
                @test ctl.renderer_owner === anchor && anchor[] === vis
                task, handle = fail_tick(screen)
                done(task; failed=true)
                @test task.result === PRIMARY
                Makie.off(handle)
                handle = nothing
                @test !view.presentation_valid[]
                @test isnothing(BAT.visualizer_figure(vis))
                @test ctl.outcome === :complete && ctl.data_complete
                if blocked
                    @test !view.disposed[]
                    @test ctl.disposal_task isa Task
                    @test timedwait(() -> view.cleanup_blocked[], 5.0; pollint=0.005) === :ok
                    release_hold[] = true
                    done(holder)
                end
                status = await_release(vis)
                @test status.outcome === :complete && status.data_complete
                @test status.error == (type="ErrorException", message=PRIMARY.msg, phase=:renderloop)
                @test before == copy(Random.default_rng())
                @test screen.config.renderloop === GLMakie.renderloop
                @test !isopen(screen)
            finally
                release_hold[] = true
                if holder isa Task
                    timedwait(() -> istaskdone(holder), 30.0; pollint=0.005) === :ok ||
                        error("View holder cleanup timed out")
                end
                isnothing(handle) || Makie.off(handle)
                dispose_fixture(vis)
            end
        end
    end

    @testset "Normal native pause and close remain benign" begin
        vis = make_vis()
        try
            sample_live(vis)
            fig = BAT.visualizer_figure(vis)
            screen = Makie.getscreen(fig.scene, GLMakie)
            task = screen.rendertask
            GLMakie.pause_renderloop!(screen)
            sleep(0.05)
            @test !istaskdone(task)
            @test BAT.visualizer_figure(vis) === fig
            @test isnothing(BAT.visualizer_status(vis).error)
            GLMakie.start_renderloop!(screen)
            @test screen.rendertask === task
            close(screen)
            done(task)
            status = await_release(vis; failed=false)
            @test status.data_complete && status.outcome === :complete && isnothing(status.error)
            @test screen.config.renderloop === GLMakie.renderloop
        finally
            dispose_fixture(vis)
        end
    end

    @testset "Installation rollback and custom-loop admission" begin
        for mode in (:preentry, :closed, :return, :stopping, :throw)
            failures = Any[]
            loop = mode === :return ? (_ -> nothing) : mode === :throw ? (_ -> throw(PRIMARY)) :
                mode === :stopping ? (screen -> begin
                    screen.stop_renderloop[] = true
                    yield()
                    nothing
                end) : GLMakie.renderloop
            fig, screen = stopped_screen(loop)
            config = screen.config
            try
                mode === :closed && close(screen)
                ENROLL_FAULT[] = mode === :preentry
                err = try
                    EXT._start_observed_gl_renderloop!(fig, screen, GLMakie, (err, bt) -> push!(failures, err))
                    nothing
                catch err
                    err
                end
                @test screen.config === config && config.renderloop === loop
                if mode === :closed
                    @test isnothing(err) && isempty(failures)
                elseif mode === :preentry
                    @test err === PRIMARY && isempty(failures)
                elseif mode === :return
                    @test err isa TaskFailedException && length(failures) == 1
                    @test only(failures) isa ErrorException && occursin("returned", only(failures).msg)
                elseif mode === :throw
                    @test err isa TaskFailedException
                    @test err.task === screen.rendertask && err.task.result === PRIMARY
                    @test failures == [PRIMARY]
                else
                    @test !isnothing(err) && isempty(failures)
                end
            finally
                ENROLL_FAULT[] = false
                close_figure(fig)
            end
        end
        fig, screen = stopped_screen()
        original = screen.config.renderloop
        try
            EXT._start_gl_renderloop!(screen, GLMakie)
            task = screen.rendertask
            @test_throws ArgumentError EXT._start_observed_gl_renderloop!(fig, screen, GLMakie, (_...) -> nothing)
            @test screen.rendertask === task && !istaskdone(task) && screen.config.renderloop === original
        finally
            close_figure(fig)
        end
    end

    @testset "Configuration replacement survives enrollment and failure" begin
        replacement = screen -> GLMakie.renderloop(screen)
        loop = screen -> begin
            screen.config.renderloop = replacement
            GLMakie.renderloop(screen)
        end
        fig, screen = stopped_screen(loop)
        failures = Any[]
        handle = nothing
        try
            config = screen.config
            EXT._start_observed_gl_renderloop!(fig, screen, GLMakie, (err, bt) -> push!(failures, err))
            @test screen.config === config && config.renderloop === replacement
            task, handle = fail_tick(screen)
            done(task; failed=true)
            @test task.result === PRIMARY && failures == [PRIMARY]
            @test config.renderloop === replacement
        finally
            isnothing(handle) || Makie.off(handle)
            close_figure(fig)
        end
    end

    @testset "A stopping predecessor still owns its native screen" begin
        release = Ref(false)
        calls = Ref(0)
        loop = screen -> begin
            calls[] += 1
            timedwait(() -> release[], 20.0; pollint=0.005) === :ok ||
                error("Predecessor release timed out")
            return nothing
        end
        fig, screen = stopped_screen(loop)
        task = nothing
        try
            EXT._start_gl_renderloop!(screen, GLMakie)
            task = screen.rendertask
            config = screen.config
            screen.stop_renderloop[] = true
            @test task isa Task && !istaskdone(task) && !GLMakie.renderloop_running(screen)
            @test_throws ArgumentError EXT._start_observed_gl_renderloop!(fig, screen, GLMakie, (_...) -> nothing)
            @test calls[] == 1 && screen.rendertask === task && !istaskdone(task)
            @test screen.config === config && config.renderloop === loop
            # The public live display path must reject before scene attachment
            # as well, preserving the still-unfinished predecessor's ownership.
            @test_throws ArgumentError EXT._maybe_display(fig; on_render_failure=(_...) -> nothing)
            @test calls[] == 1 && screen.rendertask === task && !istaskdone(task)
            @test screen.scene === fig.scene && isopen(screen)
            @test screen.config === config && config.renderloop === loop
            release[] = true
            done(task)
        finally
            release[] = true
            if task isa Task
                timedwait(() -> istaskdone(task), 30.0; pollint=0.005) === :ok ||
                    error("Predecessor cleanup timed out")
            end
            close_figure(fig)
        end
    end

    @testset "Actual inline PNG and rejection before implicit restart" begin
        capture = PNGDisplay(0, 0, true)
        pushdisplay(capture)
        Makie.inline!(true)
        vis = make_vis()
        handle = nothing
        try
            sample_live(vis)
            @test capture.calls == 1 && capture.bytes > 100 && capture.valid_png
            fig = BAT.visualizer_figure(vis)
            @test fig isa Figure
            screen = Makie.getscreen(fig.scene, GLMakie)
            task, handle = fail_tick(screen)
            done(task; failed=true)
            Makie.off(handle)
            handle = nothing
            status = await_release(vis)
            @test status.data_complete && status.outcome === :complete && status.error.phase === :renderloop
            for stopping in (false, true)
                calls = Ref(0)
                loop = function(screen)
                    calls[] += 1
                    if calls[] == 1
                        stopping && (screen.stop_renderloop[] = true; yield())
                        return nothing
                    end
                    yield()
                    throw(PRIMARY)
                end
                GLMakie.activate!(visible=false, renderloop=loop)
                Makie.inline!(true)
                fig = small_figure()
                try
                    before = capture.calls
                    err = try
                        EXT._maybe_display(fig; on_render_failure=(_...) -> nothing)
                        nothing
                    catch err
                        err
                    end
                    @test !isnothing(err)
                    @test calls[] == 1 && capture.calls == before
                    @test all(s -> s.config.renderloop !== loop || !isopen(s), GLMakie.ALL_SCREENS)
                finally
                    close_figure(fig)
                end
            end
        finally
            isnothing(handle) || Makie.off(handle)
            dispose_fixture(vis)
            GLMakie.activate!(visible=false, renderloop=GLMakie.renderloop)
            Makie.inline!(false)
            popdisplay(capture)
        end
    end

    @testset "Invalidation and cleanup ownership precede yielding BAT diagnostics" begin
        vis = make_vis()
        logger = YieldLogger(vis, false, false, nothing)
        handle = nothing
        try
            sample_live(vis; logger=logger)
            fig = BAT.visualizer_figure(vis)
            screen = Makie.getscreen(fig.scene, GLMakie)
            task, handle = fail_tick(screen)
            @test timedwait(() -> logger.hit, 20.0; pollint=0.005) === :ok
            @test !logger.snapshot.valid && logger.snapshot.requested
            @test logger.snapshot.task isa Task && isnothing(logger.snapshot.figure)
            @test !istaskdone(task) # Held inside BAT's logger; cleanup cannot self-join.
            logger.release = true
            done(task; failed=true)
            @test task.result === PRIMARY
            Makie.off(handle)
            handle = nothing
            status = await_release(vis)
            @test status.error.message == PRIMARY.msg && status.data_complete
        finally
            logger.release = true
            isnothing(handle) || Makie.off(handle)
            dispose_fixture(vis)
        end
    end

    @testset "Explicit disposal after screen reuse releases history and preserves replacement" begin
        screen, replacement, weak_graph, task, retained_scene = replaced_session()
        handle = nothing
        try
            @test retained_scene
            for _ in 1:3
                GC.gc(true)
                yield()
            end
            @test weak_graph.value === nothing
            @test screen.scene === replacement.scene && !istaskdone(task)
            actual, handle = fail_tick(screen)
            @test actual === task
            done(task; failed=true)
            @test task.result === PRIMARY
            @test isopen(screen) # An obsolete observer cannot dispose the replacement.
            @test screen.config.renderloop === GLMakie.renderloop
        finally
            isnothing(handle) || Makie.off(handle)
            close_figure(replacement)
        end
    end

    @testset "Callback holds only the weak mutable anchor" begin
        callback, weak_anchor, weak_control = weak_callback_fixture()
        @test fieldnames(typeof(callback)) == (:weak_owner,)
        @test getfield(callback, :weak_owner) isa WeakRef
        for _ in 1:3
            GC.gc(true)
            yield()
        end
        @test weak_anchor.value === nothing && weak_control.value === nothing
        @test isnothing(callback(PRIMARY, Any[]))
    end

    @testset "Native closure uses the GUI owner and never joins itself" begin
        @test !isempty(CLEANUP_CALLS)
        @test all(call -> call.thread == 1, CLEANUP_CALLS)
        @test all(call -> !call.self_join, CLEANUP_CALLS)
        @test isempty(RELEASERS)
    end
finally
    ENROLL_FAULT[] = false
    GLMakie.activate!(visible=false, renderloop=GLMakie.renderloop)
    Makie.inline!(false)
    GLMakie.closeall()
    empty!(RELEASERS)
end
