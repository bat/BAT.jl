# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Process-bounded Cairo case for the renderer-to-lifecycle bridge. Calling the
# bridge directly establishes control/ownership behavior, not native renderer
# detection; the separate GL cases exercise actual render-loop exceptions.
module TestMakieRendererFailure
using BAT, Makie, Test, Distributions, Logging, Random

const EXT = Base.get_extension(BAT, :BATMakieExt)
const FAILURE = ErrorException("intentional renderer failure bridge fixture")
const FIRST_FAILURE = ErrorException("intentional earlier observation failure")
const LOGGER_FAILURE = ErrorException("intentional renderer diagnostic logger failure")

struct NoRetentionDisplay <: AbstractDisplay end
Base.displayable(::NoRetentionDisplay, ::MIME"image/png") = true
Base.display(::NoRetentionDisplay, ::Makie.Figure) = nothing

function with_display(f)
    display = NoRetentionDisplay()
    previous = Makie.ALWAYS_INLINE_PLOTS[]
    Makie.inline!(true)
    pushdisplay(display)
    try
        return with_logger(f, NullLogger())
    finally
        popdisplay(display)
        Makie.inline!(previous)
    end
end

function configured()
    c = BATMakieVisualization(adaptive_batching=false)
    BATMakieVisualization((diagonal=Hist1D, upper=Hist2D, lower=Scatter2D),
        [1, 2], 2, 20_000, 40_000, false, c.batch_growth_rate, 0.01, c.dark,
        merge(c.triagonal_config, (nbins=(8, 8),)),
        merge(c.diagonal_config, (nbins=8,)); observer_timeout=60.0)
end

locked(f, vis) = lock(f, vis.content.buffer_lock)
coordinator(vis) = locked(() -> vis.content.lifecycle.disposal_task, vis)
release!(gate) = (isready(gate) || put!(gate, nothing); nothing)

function waitfor(predicate)
    ready = timedwait(predicate, 30.0; pollint=0.005) === :ok
    @test ready
    ready || error("Renderer failure bridge fixture did not reach its barrier")
    return nothing
end

function join_owned(task)
    isnothing(task) && return nothing
    waitfor(() -> istaskdone(task))
    @test !istaskfailed(task)
    wait(task)
    return nothing
end

# No finish/dispose/status call may drive release before these assertions.
function join_release(vis, task=coordinator(vis))
    @test task isa Task
    task isa Task || error("Renderer failure did not enroll cleanup")
    join_owned(task)
    @test vis.content.graph[] === nothing
    @test vis.content.lifecycle.renderer_owner[] === nothing
    @test vis.content.lifecycle.task === nothing
    @test vis.content.lifecycle.timer === nothing
    @test vis.content.lifecycle.owner_task === nothing
    @test isempty(vis.content.lifecycle.view.subscriptions)
    status = BAT.visualizer_status(vis)
    @test status.history_released && status.quiesced && !status.cleanup_pending
    @test status.view_disposed && !status.presentation_valid
    @test isnothing(status.cleanup_error)
    return status
end

function cleanup(vis)
    BAT.dispose_visualizer!(vis)
    join_owned(coordinator(vis))
    return nothing
end

samples_signature(samples) = (v=deepcopy(collect(samples.v)), logd=copy(samples.logd),
    weight=copy(samples.weight), info=deepcopy(collect(samples.info)),
    aux=deepcopy(collect(samples.aux)))

Base.@noinline history_reference(vis) = WeakRef(vis.content.graph[].outputs)

function complete_run!(vis)
    context = BATContext(rng=BAT.Philox4x((144, 72)), visualizer=vis)
    target = BAT.BATDistMeasure(MvNormal(zeros(2), ones(2)))
    algorithm = TransformedMCMC(nchains=2, nsteps=24, strict=false,
        convergence=AssumeConvergence(), adaptive_transform=BAT.NoAdaptiveTransform(),
        transform_tuning=NoMCMCTransformTuning())
    state = BAT.MCMCState(algorithm, target, Int32(1), [zeros(2)], context)
    outputs = BAT._empty_chain_outputs(state)
    BAT.init_visualizer!(vis; rng=BAT.get_rng(context), mcmc_states=[state],
        outputs=[outputs], target=target, f_pretransform=identity)
    BAT.mcmc_iterate!!(outputs, state; max_nsteps=24, update_visualizer=true)
    BAT.finish_visualizer!(vis; success=true)
    status = BAT.visualizer_status(vis)
    @test status.outcome === :complete && status.data_complete
    @test status.quiesced && status.presentation_valid && !status.cleanup_pending
    figure = BAT.visualizer_figure(vis)
    @test figure isa Makie.Figure
    @test !isempty(figure.content)
    @test sum(length, outputs) > 0
    return (; context, outputs, figure, publication=status.last_publication)
end

# The oracle borrows and restores the caller RNG, including task-splitting
# state. Real children distinguish future-child parity from ordinary draws.
function child_streams(state)
    saved = copy(Random.default_rng())
    tasks = Task[]
    try
        copy!(Random.default_rng(), state)
        for _ in 1:2
            push!(tasks, Threads.@spawn rand(UInt64, 8))
        end
        return fetch.(tasks)
    finally
        for task in tasks
            istaskdone(task) || wait(task)
        end
        copy!(Random.default_rng(), saved)
    end
end

struct BlockingLogger{F} <: AbstractLogger
    inspect::F
    entered::Channel{Any}
    resume::Channel{Nothing}
end
Logging.min_enabled_level(::BlockingLogger) = Logging.Error
Logging.shouldlog(::BlockingLogger, args...) = true
Logging.catch_exceptions(::BlockingLogger) = false
function Logging.handle_message(logger::BlockingLogger, args...; kwargs...)
    # Capture before the logger's first possible yield. The test receives only
    # scalar/control evidence, never a retained graph or exception backtrace.
    put!(logger.entered, logger.inspect())
    take!(logger.resume)
    throw(LOGGER_FAILURE)
end

function completed_failure()
    vis = BATVisualizer(configured())
    entered, resume = Channel{Any}(1), Channel{Nothing}(1)
    reporter = nothing
    try
        fixture = complete_run!(vis)
        before_samples = samples_signature.(fixture.outputs)
        before_context_rng = copy(BAT.get_rng(fixture.context))
        callback = EXT._live_renderer_failure_callback(vis)
        weak_anchor = WeakRef(vis.content.lifecycle.renderer_owner)
        weak_graph = history_reference(vis)
        GC.gc(true)
        GC.gc(true)
        @test weak_anchor.value === vis.content.lifecycle.renderer_owner
        @test weak_anchor.value[] === vis
        @test BAT.visualizer_figure(vis) === fixture.figure

        logger = BlockingLogger(entered, resume) do
            unavailable = isnothing(BAT.visualizer_figure(vis))
            locked(vis) do
                ctl = vis.content.lifecycle
                (; unavailable, valid=ctl.view.presentation_valid[],
                    outcome=ctl.outcome, complete=ctl.data_complete,
                    accepting=ctl.accepting[], requested=ctl.disposal_requested,
                    enrolled=ctl.disposal_task isa Task, error=ctl.error)
            end
        end
        reporter = @async begin
            before = copy(Random.default_rng())
            result = with_logger(() -> callback(FAILURE, Any[]), logger)
            (; before, after=copy(Random.default_rng()), result)
        end
        waitfor(() -> isready(entered))
        observed = take!(entered)
        @test observed.unavailable && !observed.valid && !observed.accepting
        @test observed.requested && observed.enrolled
        @test observed.outcome === :complete && observed.complete
        @test observed.error.phase === :renderloop
        @test observed.error.message == FAILURE.msg
        @test !istaskdone(reporter)

        # Cleanup must finish even while BAT's arbitrary diagnostic logger is
        # suspended. No status query, finish or explicit disposal starts it.
        status = join_release(vis)
        @test status.outcome === :complete && status.data_complete
        @test status.reason === :none
        @test status.last_publication == fixture.publication
        @test isempty(fixture.figure.content)
        @test BAT.visualizer_figure(vis) === nothing
        @test isequal(samples_signature.(fixture.outputs), before_samples)
        @test BAT.get_rng(fixture.context) == before_context_rng
        GC.gc(true)
        GC.gc(true)
        @test isnothing(weak_graph.value)

        release!(resume)
        join_owned(reporter)
        streams = fetch(reporter)
        @test streams.result === nothing # The logger's throw is contained.
        @test streams.before == streams.after
        @test rand(copy(streams.before), UInt64, 16) == rand(copy(streams.after), UInt64, 16)
        @test child_streams(streams.before) == child_streams(streams.after)

        # A retained callback is inert after the cleanup owner clears its anchor.
        @test callback(FIRST_FAILURE, Any[]) === nothing
        @test BAT.visualizer_status(vis) == status
        @test coordinator(vis) === nothing
    finally
        release!(resume)
        try
            join_owned(reporter)
        finally
            cleanup(vis)
        end
    end
    return nothing
end

function held_view_failure(first_stop)
    vis = BATVisualizer(configured())
    try
        # Existing minimal real-view fixture API: no fake listener, native
        # screen or replacement production dispatch is needed for this boundary.
        EXT._claim_live_visualizer!(vis)
        figure = Figure()
        Axis(figure[1, 1])
        view = EXT._attach_live_view!(vis, figure)
        @test EXT._activate_live_visualizer!(vis)
        callback = EXT._live_renderer_failure_callback(vis)
        task = nothing
        admitted = EXT._with_live_view(vis) do
            if !isnothing(first_stop)
                exception = first_stop === :aborted ? FIRST_FAILURE : nothing
                BAT.request_visualizer_stop!(vis; reason=first_stop, exception)
            end
            callback(FAILURE, Any[])
            task = coordinator(vis)
            @test task isa Task
            @test !view.presentation_valid[] && view.cleanup_pending[]
            @test !vis.content.lifecycle.accepting[]
            waitfor(() -> view.cleanup_blocked[])
            @test !istaskdone(task)
            @test vis.content.graph[] !== nothing
            @test vis.content.lifecycle.renderer_owner[] === vis
        end
        @test !admitted
        # The admitted view's exit is the only event that can resume cleanup.
        status = join_release(vis, task)
        @test !status.data_complete
        @test status.outcome === (isnothing(first_stop) ? :failed :
            first_stop === :aborted ? :aborted : :disabled)
        @test status.reason === (isnothing(first_stop) ? :failed : first_stop)
        @test status.error.message == (first_stop === :aborted ? FIRST_FAILURE.msg : FAILURE.msg)
        @test status.error.phase === (first_stop === :aborted ? :aborted : :renderloop)
        @test isempty(figure.content)
    finally
        cleanup(vis)
    end
    return nothing
end

# A fresh visualizer has no tasks or screens to clean up. Return only the
# callback and weak references to mutable allocations, so it cannot be kept
# alive accidentally by an immutable-owner box in this fixture's stack frame.
Base.@noinline function abandoned_owner()
    vis = BATVisualizer(configured())
    callback = EXT._live_renderer_failure_callback(vis)
    return callback, WeakRef(vis.content.lifecycle.renderer_owner),
        WeakRef(vis.content.lifecycle), WeakRef(vis.content.graph[].outputs)
end

@testset "Renderer failure bridge control and lifetime" begin
    with_display() do
        @testset "Completed data survives failure before yielding/throwing diagnostics" begin
            completed_failure()
        end
        @testset "Held view cleanup preserves first stop $first_stop" for first_stop in (nothing, :closed, :aborted)
            held_view_failure(first_stop)
        end
        @testset "Retained callback cannot own an abandoned session" begin
            callback, anchor, control, graph = abandoned_owner()
            GC.gc(true)
            GC.gc(true)
            @test isnothing(anchor.value)
            @test isnothing(control.value)
            @test isnothing(graph.value)
            @test callback(FAILURE, Any[]) === nothing
        end
    end
end
end # module
