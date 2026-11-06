# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Run as the process-bounded `release_lifecycle` optional Makie case. These
# barriers exercise the real listener, view guard, cleanup and release owner.
# Failure cleanup comes after outcome assertions and releases only owned tasks.
module TestMakieReleaseLifecycle

using BAT, Test, Makie, Distributions, Logging

const EXT = Base.get_extension(BAT, :BATMakieExt)
const FAILURE = ErrorException("intentional history-release cleanup failure")

function config()
    c = BATMakieVisualization(adaptive_batching=false, observer_timeout=10.0)
    BATMakieVisualization(c.recipes, [1, 2], 2, 1, 2, false,
        c.batch_growth_rate, 0.01, c.dark, c.triagonal_config,
        c.diagonal_config; observer_timeout=10.0)
end

function waitfor(predicate; seconds=20.0)
    ready = timedwait(predicate, seconds; pollint=0.005) === :ok
    @test ready
    ready || error("History-release test barrier timed out")
    return nothing
end

function caught(f)
    try
        f()
        return nothing
    catch err
        return err
    end
end

locked(f, vis) = lock(f, vis.content.buffer_lock)
coordinator(vis) = locked(() -> vis.content.lifecycle.disposal_task, vis)
release!(gate) = (isready(gate) || put!(gate, nothing); nothing)

function join_owned(task; failed=false)
    isnothing(task) && return nothing
    waitfor(() -> istaskdone(task))
    @test istaskfailed(task) == failed
    if failed
        @test caught(() -> wait(task)) isa TaskFailedException
    else
        wait(task)
    end
    return nothing
end

# Crucially, no status query drives cleanup between disposal and this join.
function join_release(vis, task=coordinator(vis))
    @test task isa Task
    task isa Task || error("Release request has no task owner")
    join_owned(task)
    @test isnothing(vis.content.graph[])
    @test isnothing(vis.content.lifecycle.task)
    @test isnothing(vis.content.lifecycle.owner_task)
    status = BAT.visualizer_status(vis)
    @test status.history_released
    @test !status.cleanup_pending
    @test status.quiesced
    return task
end

function cleanup(vis)
    BAT.dispose_visualizer!(vis)
    task = coordinator(vis)
    isnothing(task) || join_owned(task)
    return nothing
end

function sample_fixture()
    vis = BATVisualizer(config())
    target = BAT.BATDistMeasure(MvNormal(zeros(2), ones(2)))
    context = BATContext(rng=BAT.Philox4x((71, 214)), visualizer=vis)
    algorithm = TransformedMCMC(nchains=2, nsteps=8, strict=false,
        convergence=AssumeConvergence(), adaptive_transform=BAT.NoAdaptiveTransform(),
        transform_tuning=NoMCMCTransformTuning())
    state = BAT.MCMCState(algorithm, target, Int32(1), [zeros(2)], context)
    return vis, state, BAT._empty_chain_outputs(state), target
end

# A small real view is enough to exercise admission and cleanup, without making
# every scheduler test rebuild the complete sample grid. No run task is invented.
function view_fixture()
    vis = BATVisualizer(config())
    EXT._claim_live_visualizer!(vis)
    fig = Figure()
    Axis(fig[1, 1])
    view = EXT._attach_live_view!(vis, fig)
    @test EXT._activate_live_visualizer!(vis)
    return vis, view, fig
end

# The fixture owns this type and method. The production subscription collection
# deliberately accepts handles through Observables.off; no production method is
# replaced and no arbitrary running task receives an injected exception.
mutable struct CleanupHandle{F}
    action::F
    calls::Int
end
CleanupHandle(f) = CleanupHandle(f, 0)
function Makie.Observables.off(handle::CleanupHandle)
    handle.calls += 1
    handle.action(handle.calls)
    return nothing
end

mutable struct PausedDisplay <: AbstractDisplay
    entered::Channel{Nothing}
    resume::Channel{Nothing}
    fail::Bool
end
Base.displayable(::PausedDisplay, ::MIME"image/png") = true
function Base.display(display::PausedDisplay, ::Figure)
    put!(display.entered, nothing)
    take!(display.resume)
    display.fail && throw(FAILURE)
    return nothing
end

function fresh_and_orphan_timer()
    @testset "fresh/no-view disposal" begin
        vis = BATVisualizer(config())
        try
            @test !BAT.visualizer_status(vis).history_released
            @test isnothing(vis.content.lifecycle.view)
            BAT.dispose_visualizer!(vis)
            task = coordinator(vis)
            BAT.dispose_visualizer!(vis)
            @test coordinator(vis) === task
            join_release(vis, task)
            @test isnothing(vis.content.lifecycle.view)
            @test !EXT._with_live_view(() -> error("released view admitted"), vis)
            @test isnothing(vis.content.lifecycle.view)
            @test_throws ArgumentError EXT._claim_live_visualizer!(vis)
            BAT.finish_visualizer!(vis; success=false)
            BAT.dispose_visualizer!(vis)
            @test isnothing(vis.content.lifecycle.task)
            @test isnothing(vis.content.graph[])
            @test BAT.visualizer_status(vis).history_released
        finally
            cleanup(vis)
        end
    end

    @testset "orphan timer and transport, no listener" begin
        vis, state, outputs, _ = sample_fixture()
        timer = Timer(60.0; interval=60.0)
        try
            EXT._claim_live_visualizer!(vis)
            vis.content.n_dof[] = 2
            EXT.register_state_for_vis!(vis, state, outputs)
            locked(vis) do
                vis.content.lifecycle.timer = timer
            end
            @test isopen(timer)
            @test !isempty(vis.content.output_buffer)
            @test isnothing(vis.content.lifecycle.task)
            BAT.dispose_visualizer!(vis)
            join_release(vis)
            @test !isopen(timer)
            @test isnothing(vis.content.lifecycle.timer)
            @test all(isempty, (vis.content.output_buffer, vis.content.pending_samples,
                vis.content.output_timing, vis.content.pending_timing))
            @test isnothing(vis.content.lifecycle.view)
        finally
            close(timer)
            cleanup(vis)
        end
    end
end

function blocked_listener()
    vis, state, outputs, _ = sample_fixture()
    entered, resume = Channel{Nothing}(1), Channel{Nothing}(1)
    listener = nothing
    calls = Ref(0)
    try
        EXT._claim_live_visualizer!(vis)
        vis.content.n_dof[] = 2
        EXT.register_state_for_vis!(vis, state, outputs)
        EXT._start_live_listener!(vis, () -> begin
            calls[] += 1
            put!(entered, nothing)
            take!(resume)
        end, () -> nothing)
        listener = vis.content.lifecycle.task
        waitfor(() -> isready(entered))
        take!(entered)
        BAT.dispose_visualizer!(vis)
        task = coordinator(vis)
        for _ in 1:4
            BAT.dispose_visualizer!(vis)
            @test coordinator(vis) === task
            yield()
        end
        @test !istaskdone(listener)
        @test !istaskdone(task)
        @test !isnothing(vis.content.graph[])
        @test !vis.content.lifecycle.release_committed
        @test BAT.visualizer_status(vis).cleanup_pending
        @test !BAT.visualizer_status(vis).history_released
        release!(resume)
        join_owned(listener)
        join_release(vis, task)
        @test calls[] == 1
    finally
        release!(resume)
        join_owned(listener)
        cleanup(vis)
    end
end

function paused_initialization()
    for fail in (false, true)
        @testset "public initializer paused, fail=$fail" begin
            vis, state, outputs, target = sample_fixture()
            display = PausedDisplay(Channel{Nothing}(1), Channel{Nothing}(1), fail)
            owner = nothing
            old_inline = Makie.ALWAYS_INLINE_PLOTS[]
            Makie.inline!(true)
            pushdisplay(display)
            try
                owner = @async with_logger(NullLogger()) do
                    caught() do
                        BAT.init_visualizer!(vis; rng=BAT.get_rng(state.chain_state.context),
                            mcmc_states=[state], outputs=[outputs], target=target,
                            f_pretransform=identity)
                    end
                end
                waitfor(() -> isready(display.entered))
                take!(display.entered)
                @test vis.content.lifecycle.initializing
                @test isnothing(vis.content.lifecycle.task)
                BAT.dispose_visualizer!(vis)
                task = coordinator(vis)
                yield()
                @test !istaskdone(task)
                @test !isnothing(vis.content.graph[])
                @test !vis.content.lifecycle.release_committed
                release!(display.resume)
                join_owned(owner)
                @test fetch(owner) === (fail ? FAILURE : nothing)
                join_release(vis, task)
                @test !vis.content.lifecycle.initializing
                @test isnothing(vis.content.lifecycle.task)
                # Neither a repeated rollback nor a stale startup activation
                # can enroll a fallback/listener after the release handoff.
                BAT.finish_visualizer!(vis; success=false)
                @test !EXT._activate_live_visualizer!(vis)
                @test isnothing(vis.content.lifecycle.task)
            finally
                release!(display.resume)
                join_owned(owner)
                popdisplay(display)
                Makie.inline!(old_inline)
                cleanup(vis)
            end
        end
    end
end

function late_startup_enrollment()
    # Select the small reservation/enrollment boundary directly: public startup
    # above covers the real caller; this owns a new task enrolled while its
    # initialization reservation is still held, after disposal began waiting.
    vis = BATVisualizer(config())
    entered, resume = Channel{Nothing}(1), Channel{Nothing}(1)
    listener = nothing
    try
        EXT._claim_live_visualizer!(vis)
        locked(vis) do
            vis.content.lifecycle.initializing = true
        end
        BAT.dispose_visualizer!(vis)
        task = coordinator(vis)
        yield()
        @test !istaskdone(task)
        listener = @task begin
            put!(entered, nothing)
            take!(resume)
        end
        locked(vis) do
            vis.content.lifecycle.task = listener
        end
        schedule(listener)
        waitfor(() -> isready(entered))
        take!(entered)
        EXT._finish_live_initialization!(vis)
        yield()
        @test !vis.content.lifecycle.initializing
        @test !istaskdone(listener)
        @test !istaskdone(task)
        @test !isnothing(vis.content.graph[])
        @test !vis.content.lifecycle.release_committed
        release!(resume)
        join_owned(listener)
        join_release(vis, task)
    finally
        release!(resume)
        EXT._finish_live_initialization!(vis)
        join_owned(listener)
        cleanup(vis)
    end
end

function active_and_reentrant_callbacks()
    @testset "admitted callback supplies final readiness event" begin
        vis, view, fig = view_fixture()
        entered, resume = Channel{Nothing}(1), Channel{Nothing}(1)
        callback = nothing
        source = Observable(0)
        EXT._view_on(vis, source) do _
            put!(entered, nothing)
            take!(resume)
        end
        try
            callback = @async source[] = 1
            waitfor(() -> isready(entered))
            take!(entered)
            BAT.dispose_visualizer!(vis)
            task = coordinator(vis)
            yield()
            @test view.depth == 1
            @test !istaskdone(task)
            @test !view.disposed[]
            @test !isempty(fig.content)
            @test !isnothing(vis.content.graph[])
            release!(resume)
            join_owned(callback)
            join_release(vis, task)
            @test view.depth == 0
            @test isempty(fig.content)
            @test isempty(view.subscriptions)
            @test isempty(Makie.Observables.listeners(source))
        finally
            release!(resume)
            join_owned(callback)
            cleanup(vis)
        end
    end

    @testset "reentrant nested callback waits for outermost exit" begin
        vis, view, fig = view_fixture()
        source = Observable(0)
        owner = Ref{Union{Nothing,Task}}(nothing)
        EXT._view_on(vis, source) do _
            EXT._with_live_view(vis) do
                @test view.depth == 2
                BAT.dispose_visualizer!(vis)
                owner[] = coordinator(vis)
                yield()
                @test !istaskdone(owner[])
                @test !isnothing(vis.content.graph[])
                @test !view.disposed[]
            end
            @test view.depth == 1
            yield()
            @test !istaskdone(owner[])
            @test !isempty(fig.content)
        end
        try
            source[] = 1
            join_release(vis, owner[])
            @test view.depth == 0
            @test isempty(fig.content)
            @test isempty(Makie.Observables.listeners(source))
        finally
            cleanup(vis)
        end
    end
end

function preadmission_guard_exit()
    vis, view, fig = view_fixture()
    listener_entered, listener_resume = Channel{Nothing}(1), Channel{Nothing}(1)
    callback_started = Channel{Nothing}(1)
    body_called = Ref(false)
    callback = nothing
    listener = @task begin
        put!(listener_entered, nothing)
        take!(listener_resume)
    end
    held_view, held_control = false, false
    advance() = foreach(_ -> yield(), 1:256)
    try
        locked(vis) do
            vis.content.lifecycle.task = listener
        end
        schedule(listener)
        waitfor(() -> isready(listener_entered))
        take!(listener_entered)

        # First let the callback pass its initial control reads and queue on
        # the view guard. No production method or running task is overridden.
        lock(view.lock)
        held_view = true
        callback = @async begin
            put!(callback_started, nothing)
            EXT._with_live_view(() -> (body_called[] = true), vis)
        end
        waitfor(() -> isready(callback_started))
        take!(callback_started)
        advance()
        @test !istaskdone(callback)

        BAT.dispose_visualizer!(vis)
        task = coordinator(vis)
        waitfor(() -> locked(() -> !vis.content.lifecycle.disposal_retry, vis))
        @test !istaskdone(task)

        # First queue the coordinator on control reacquisition after its
        # listener join. Then hand view.lock to the callback: it owns that
        # guard but queues behind the coordinator for its admission check.
        lock(vis.content.buffer_lock)
        held_control = true
        release!(listener_resume)
        waitfor(() -> istaskdone(listener))
        advance()
        unlock(view.lock)
        held_view = false
        advance()
        @test islocked(view.lock)
        @test view.depth == 0
        @test !body_called[]
        @test !istaskdone(callback)
        @test !isnothing(vis.content.graph[])
        unlock(vis.content.buffer_lock)
        held_control = false

        join_owned(callback)
        join_owned(listener)
        @test fetch(callback) === false
        @test !body_called[]
        # The rejected guard-holder's exit is the sole remaining readiness
        # event. No second dispose/status call may rescue this assertion.
        join_release(vis, task)
        @test isempty(fig.content)
        @test !islocked(view.lock)
    finally
        held_control && unlock(vis.content.buffer_lock)
        held_view && unlock(view.lock)
        release!(listener_resume)
        join_owned(listener)
        join_owned(callback)
        cleanup(vis)
    end
end

function cleanup_retry()
    @testset "retry arriving during failed off is not lost" begin
        vis, view, _ = view_fixture()
        entered, resume = Channel{Nothing}(1), Channel{Nothing}(1)
        failing = Ref(true)
        handle = CleanupHandle() do call
            if call == 1 && failing[]
                put!(entered, nothing)
                take!(resume)
                throw(FAILURE)
            end
        end
        push!(view.subscriptions, handle)
        try
            BAT.dispose_visualizer!(vis)
            task = coordinator(vis)
            waitfor(() -> isready(entered))
            take!(entered)
            @test view.cleanup_running[]
            @test !isnothing(vis.content.graph[])
            BAT.dispose_visualizer!(vis)
            @test coordinator(vis) === task
            @test locked(() -> vis.content.lifecycle.disposal_retry, vis)
            release!(resume)
            join_release(vis, task)
            @test handle.calls == 2
            @test isempty(view.subscriptions)
            @test !isnothing(BAT.visualizer_status(vis).cleanup_error)
        finally
            failing[] = false
            release!(resume)
            cleanup(vis)
        end
    end

    @testset "rejected and self-reentrant callbacks do not spin" begin
        vis, view, _ = view_fixture()
        failing = Ref(true)
        admitted = Ref(0)
        handle = CleanupHandle() do _
            if failing[]
                @test !EXT._with_live_view(() -> (admitted[] += 1), vis)
                BAT.dispose_visualizer!(vis) # synchronous coordinator reentry
                yield() # let a defective retry loop remain observable/bounded
                throw(FAILURE)
            end
        end
        push!(view.subscriptions, handle)
        try
            BAT.dispose_visualizer!(vis)
            task = coordinator(vis)
            waitfor(() -> handle.calls >= 1 && !view.cleanup_running[])
            @test handle.calls == 1
            @test handle in view.subscriptions
            @test !isnothing(vis.content.graph[])
            @test !istaskdone(task)
            @test !BAT.visualizer_status(vis).history_released
            for _ in 1:10
                @test !EXT._with_live_view(() -> (admitted[] += 1), vis)
                EXT._request_view_disposal!(vis)
                EXT._schedule_live_view_cleanup!(vis; retry=false)
                yield()
            end
            @test handle.calls == 1
            @test admitted[] == 0
            @test coordinator(vis) === task
            @test !locked(() -> vis.content.lifecycle.disposal_retry, vis)
            failing[] = false
            BAT.dispose_visualizer!(vis)
            join_release(vis, task)
            @test handle.calls == 2
        finally
            failing[] = false
            cleanup(vis)
        end
    end
end

function failed_task_boundaries()
    @testset "terminal failed listener still releases" begin
        vis = BATVisualizer(config())
        listener = @task throw(FAILURE)
        try
            EXT._claim_live_visualizer!(vis)
            locked(vis) do
                vis.content.lifecycle.task = listener
            end
            schedule(listener)
            join_owned(listener; failed=true)
            BAT.dispose_visualizer!(vis)
            join_release(vis)
            @test !isnothing(BAT.visualizer_status(vis).cleanup_error)
        finally
            cleanup(vis)
        end
    end

    @testset "post-commit task failure is pending until explicit retry" begin
        # An owned NEW task models the tiny post-commit boundary without throwing
        # into a running coordinator. Status and retry are production operations.
        vis = BATVisualizer(config())
        task = @task begin
            locked(vis) do
                vis.content.graph[] = nothing
                vis.content.lifecycle.release_committed = true
            end
            throw(FAILURE)
        end
        try
            EXT._claim_live_visualizer!(vis)
            BAT.request_visualizer_stop!(vis; reason=:disposed)
            locked(vis) do
                ctl = vis.content.lifecycle
                ctl.disposal_requested = true
                ctl.disposal_retry = false
                ctl.release_complete = false
                ctl.disposal_task = task
            end
            schedule(task)
            join_owned(task; failed=true)
            @test isnothing(vis.content.graph[])
            for _ in 1:2
                status = BAT.visualizer_status(vis)
                @test !status.history_released
                @test status.cleanup_pending
            end
            BAT.dispose_visualizer!(vis)
            retry = coordinator(vis)
            @test retry !== task
            join_release(vis, retry)
        finally
            cleanup(vis)
        end
    end
end

function wrong_thread()
    vis, _, _ = view_fixture()
    try
        owner = vis.content.lifecycle.owner_thread
        results = Channel{Any}(Threads.nthreads(:default))
        Threads.@threads :static for _ in 1:Threads.nthreads(:default)
            if Threads.threadid() != owner
                put!(results, caught(() -> BAT.dispose_visualizer!(vis)))
            end
        end
        close(results)
        answers = collect(results)
        if isempty(answers)
            @test_skip "wrong-thread disposal requires another default thread"
        else
            @test all(err -> err isa ArgumentError, answers)
            @test !vis.content.lifecycle.disposal_requested
            @test isnothing(coordinator(vis))
            @test !isnothing(vis.content.graph[])
        end
        BAT.dispose_visualizer!(vis)
        join_release(vis)
    finally
        cleanup(vis)
    end
end

@testset "Live history release lifecycle" begin
    fresh_and_orphan_timer()
    @testset "blocked production listener" begin
        blocked_listener()
    end
    paused_initialization()
    @testset "startup handoff precedes listener snapshot" begin
        late_startup_enrollment()
    end
    active_and_reentrant_callbacks()
    @testset "rejected admission releases a blocking view guard" begin
        preadmission_guard_exit()
    end
    cleanup_retry()
    failed_task_boundaries()
    @testset "owner-thread disposal" begin
        wrong_thread()
    end
end

end # module
