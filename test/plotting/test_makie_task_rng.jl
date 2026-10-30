# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Run in the externally bounded CairoMakie optional subprocess. H30 covers
# BAT-owned enrollment; renderer-owned GL task creation remains H31. Fixture
# callbacks own their methods; no production dispatch is replaced here.
module TestMakieTaskRNG
using BAT, Makie, Test, Distributions, LinearAlgebra, Logging, Random
const EXT = Base.get_extension(BAT, :BATMakieExt)
const FAILURE = ErrorException("intentional observer task RNG fixture failure")

mutable struct Capture{F} <: AbstractDisplay
    action::F
    figures::Vector{Makie.Figure}
end
Base.displayable(::Capture, ::MIME"image/png") = true
function Base.display(display::Capture, figure::Makie.Figure)
    push!(display.figures, figure)
    display.action(figure)
    return nothing
end
function with_display(f; action=(_ -> nothing))
    display = Capture(action, Makie.Figure[])
    previous = Makie.ALWAYS_INLINE_PLOTS[]
    Makie.inline!(true)
    pushdisplay(display)
    try
        return with_logger(() -> f(display), NullLogger())
    finally
        popdisplay(display)
        Makie.inline!(previous)
    end
end

function configured()
    c = BATMakieVisualization(adaptive_batching=false)
    recipes = (diagonal=Hist1D, upper=Hist2D, lower=Scatter2D)
    return BATMakieVisualization(recipes, [1, 2], 2, 20_000, 40_000, false,
        c.batch_growth_rate, 0.01, c.dark,
        merge(c.triagonal_config, (nbins=(8, 8),)),
        merge(c.diagonal_config, (nbins=8,)); observer_timeout=30.0)
end
function algorithm(; callback=BAT.nop_func)
    TransformedMCMC(nchains=2, nsteps=24, strict=false, store_burnin=true,
        init=MCMCChainPoolInit(nsteps_init=30), convergence=AssumeConvergence(),
        adaptive_transform=BAT.NoAdaptiveTransform(),
        transform_tuning=NoMCMCTransformTuning(), callback=callback,
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=12, max_ncycles=1, nsteps_final=3))
end
function fixture()
    vis = BATVisualizer(configured())
    context = BATContext(rng=BAT.Philox4x((144, 72)), visualizer=vis)
    target = BAT.BATDistMeasure(MvNormal(zeros(2), ones(2)))
    state = BAT.MCMCState(algorithm(), target, Int32(1), [zeros(2)], context)
    return vis, state, BAT._empty_chain_outputs(state), target
end
function initialize(vis, state, outputs, target)
    BAT.init_visualizer!(vis; rng=BAT.get_rng(state.chain_state.context),
        mcmc_states=[state], outputs=[outputs], target=target, f_pretransform=identity)
    return vis.content.lifecycle.view.figure
end
function completed_fixture()
    vis, state, outputs, target = fixture()
    figure = initialize(vis, state, outputs, target)
    BAT.mcmc_iterate!!(outputs, state; max_nsteps=12, update_visualizer=true)
    BAT.finish_visualizer!(vis; success=true)
    @test BAT.visualizer_status(vis).data_complete
    @test BAT.visualizer_figure(vis) === figure
    return vis, figure
end
locked(f, vis) = lock(f, vis.content.buffer_lock)
function waitfor(predicate)
    ready = timedwait(predicate, 30.0; pollint=0.005) === :ok
    @test ready
    ready || error("Observer task RNG fixture did not reach its barrier")
end
function caught(f)
    try
        f()
        return nothing
    catch err
        return err
    end
end
function join_release(vis, task=locked(() -> vis.content.lifecycle.disposal_task, vis))
    @test task isa Task
    task isa Task || error("Disposal did not enroll its owned coordinator")
    waitfor(() -> istaskdone(task))
    wait(task)
    @test fetch(task) === nothing
    @test vis.content.graph[] === nothing
    @test vis.content.lifecycle.task === nothing
    @test vis.content.lifecycle.timer === nothing
    status = BAT.visualizer_status(vis)
    @test status.history_released && status.quiesced && !status.cleanup_pending
    return task
end
function cleanup(vis)
    # All normal outcome/resource assertions precede this fallback.
    BAT.request_visualizer_stop!(vis; reason=:disabled)
    BAT.finish_visualizer!(vis; success=false)
    BAT.dispose_visualizer!(vis)
    waitfor(() -> !BAT.visualizer_status(vis).cleanup_pending)
end

# Observe real child streams without advancing the state being inspected. Only
# the test oracle borrows the current task RNG; production callbacks do not.
function child_streams(state)
    saved = copy(Random.default_rng())
    tasks = Task[]
    try
        copy!(Random.default_rng(), state)
        for _ in 1:3
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
function assert_rng_equal(before, after)
    @test before == after # Xoshiro equality includes the task-splitting word.
    @test rand(copy(before), UInt64, 16) == rand(copy(after), UInt64, 16)
    @test child_streams(before) == child_streams(after)
end
function unchanged(f)
    before = copy(Random.default_rng())
    result = f()
    after = copy(Random.default_rng())
    assert_rng_equal(before, after)
    return result
end
function caller_work()
    before = rand(UInt64, 4)
    child = fetch(Threads.@spawn rand(UInt64, 8))
    after = rand(UInt64, 4)
    return (; before, child, after)
end
function expected_caller_work(state)
    saved = copy(Random.default_rng())
    try
        copy!(Random.default_rng(), state)
        value = caller_work()
        return (value=value, state=copy(Random.default_rng()))
    finally
        copy!(Random.default_rng(), saved)
    end
end
function check_user_effects(f)
    before = copy(Random.default_rng())
    expected = expected_caller_work(before)
    @test expected.state != before
    actual = f()
    after = copy(Random.default_rng())
    @test actual == expected.value
    assert_rng_equal(expected.state, after)
    return nothing
end

sample_signature(samples) = (v=deepcopy(collect(samples.v)), logd=copy(samples.logd),
    weight=copy(samples.weight), info=deepcopy(collect(samples.info)), aux=deepcopy(collect(samples.aux)))
function chain_signature(chain)
    return (stepno=chain.stepno, accepted=copy(chain.accepted),
        current=map(sample_signature, chain.current), proposed=map(sample_signature, chain.proposed),
        output=sample_signature(chain.output), nattempts=copy(chain.nattempts),
        nsamples=copy(chain.nsamples), rng=copy(BAT.get_rng(chain.context)),
        walkers=[copy(BAT.get_rng(ctx)) for ctx in chain.walker_genctxs])
end

function completed_disposal()
    vis = BATVisualizer(configured())
    context = BATContext(rng=BAT.Philox4x((91, 217)), visualizer=vis)
    try
        with_display() do display
            result = BAT.evalmeasure(MvNormal(zeros(2), ones(2)), algorithm(), context)
            @test !isempty(BAT.samplesof(result))
            @test length(display.figures) == 1
            figure = only(display.figures)
            @test BAT.visualizer_figure(vis) === figure
            @test BAT.visualizer_status(vis).data_complete
            private_before = copy(vis.content.lifecycle.task_rng)
            unchanged() do
                BAT.dispose_visualizer!(vis)
                join_release(vis)
            end
            private_after = copy(vis.content.lifecycle.task_rng)
            @test private_after != private_before
            @test isempty(figure.content)
            @test BAT.visualizer_status(vis).data_complete
            unchanged() do
                for _ in 1:3
                    BAT.dispose_visualizer!(vis)
                    @test BAT.visualizer_status(vis).history_released
                end
            end
            @test vis.content.lifecycle.task_rng == private_after
            @test vis.content.lifecycle.disposal_task === nothing
        end
    finally
        cleanup(vis)
    end
end

function active_result(live)
    vis = live ? BATVisualizer(configured()) : BATVisualizer()
    context = BATContext(rng=BAT.Philox4x((0x71, 0x95)), visualizer=vis)
    events = Any[]
    disposed = Ref(false)
    owner = current_task()
    callback = function(event, states)
        @test current_task() === owner
        # This belongs to the numerical caller in both runs. A broad RNG rewind
        # over the callback or disposal must not erase its draws or fork.
        work = caller_work()
        push!(events, (event=event, work=work,
            chains=chain_signature.(getfield.(states, :chain_state))))
        if live && !disposed[]
            @test vis.content.lifecycle.phase === :running
            unchanged() do
                BAT.dispose_visualizer!(vis)
                join_release(vis)
            end
            disposed[] = true
        end
        return nothing
    end
    try
        return with_display() do display
            Random.seed!(0x30a3)
            result = BAT.evalmeasure(MvNormal(zeros(2), ones(2)), algorithm(; callback), context)
            after = copy(Random.default_rng())
            streams = child_streams(after)
            @test length(events) == 1
            @test length(display.figures) == Int(live)
            if live
                @test disposed[]
                status = BAT.visualizer_status(vis)
                @test status.reason === :disposed && status.outcome === :disabled
                @test status.history_released && !status.cleanup_pending
                @test vis.content.graph[] === nothing
            end
            return (samples=sample_signature(BAT.samplesof(result)),
                chains=chain_signature.(BAT.samplegenof(result).chain_states),
                context=copy(BAT.get_rng(context)), events=events, after=after, streams=streams)
        end
    finally
        live && cleanup(vis)
    end
end

function startup_failure()
    vis, state, outputs, target = fixture()
    work = Ref{Any}(nothing)
    try
        with_display(; action=figure -> begin
            work[] = caller_work()
            throw(FAILURE)
        end) do display
            check_user_effects() do
                err = caught(() -> initialize(vis, state, outputs, target))
                @test err === FAILURE
                @test length(display.figures) == 1
                # Startup rollback itself joins before returning; status only
                # checks the resulting state rather than driving cleanup.
                status = BAT.visualizer_status(vis)
                @test status.outcome === :failed && status.history_released
                @test status.quiesced && !status.cleanup_pending
                @test status.error.message == FAILURE.msg
                @test vis.content.lifecycle.timer === nothing
                @test vis.content.lifecycle.task === nothing
                return work[]
            end
        end
    finally
        cleanup(vis)
    end
end

# Known shape and LADJ keep transform discovery from evaluating this callable.
# The actual serial row loop must reach row one successfully, then preserve
# caller-owned work and the original exception from row two.
mutable struct StartupFailingInverse <: Function
    calls::Int
    work::Any
    tasks::Vector{Task}
    successful::Vector{Vector{Float64}}
end
struct StartupPretransform <: Function
    backward::StartupFailingInverse
end
(::StartupPretransform)(x::AbstractVector{<:Real}) = 0.5 .* x
BAT.inverse(f::StartupPretransform) = f.backward
BAT.resultshape(::StartupFailingInverse, shape::BAT.ArrayShape) = shape
function (f::StartupFailingInverse)(x::AbstractVector{<:Real})::Vector{Float64}
    f.calls += 1
    push!(f.tasks, current_task())
    if f.calls == 2
        f.work = caller_work()
        throw(FAILURE)
    end
    value = 2 .* Float64.(x)
    push!(f.successful, value)
    return value
end
BAT.with_logabsdet_jacobian(f::StartupFailingInverse, x::AbstractVector{<:Real}) =
    (f(x), length(x) * log(2.0))

function startup_transform_failure()
    vis, state, outputs, target = fixture()
    # Populate legitimate compressed walker output through the actual sampler,
    # before the observation acquisition and the caller-RNG checkpoint.
    state = BAT.mcmc_iterate!!(outputs, state; max_nsteps=32, update_visualizer=false)
    rows = only(outputs)
    @test length(rows) >= 2
    length(rows) >= 2 || error("Startup transform fixture needs two sampler rows")
    original = sample_signature(rows)
    explicit_rng = copy(BAT.get_rng(state.chain_state.context))
    backward = StartupFailingInverse(0, nothing, Task[], Vector{Float64}[])
    transform = StartupPretransform(backward)
    owner = current_task()
    try
        with_display() do display
            check_user_effects() do
                err = caught() do
                    BAT.init_visualizer!(vis; rng=BAT.get_rng(state.chain_state.context),
                        mcmc_states=[state], outputs=[outputs], target=target,
                        f_pretransform=transform)
                end
                @test err === FAILURE
                @test backward.calls == 2
                @test backward.tasks == [owner, owner]
                @test backward.successful == [2 .* Float64.(rows.v[1])]
                @test isempty(display.figures) # Fails before figure construction.
                @test isequal(sample_signature(rows), original)
                @test BAT.get_rng(state.chain_state.context) == explicit_rng
                ctl = vis.content.lifecycle
                # Check autonomous rollback before querying status or invoking
                # fallback cleanup. No listener was ever admitted.
                @test !ctl.initializing
                @test ctl.task === nothing && ctl.timer === nothing
                @test ctl.disposal_task === nothing || istaskdone(ctl.disposal_task)
                @test vis.content.graph[] === nothing
                @test ctl.last_publication == 0
                status = BAT.visualizer_status(vis)
                @test status.outcome === :failed && status.history_released
                @test status.quiesced && !status.cleanup_pending && !status.data_complete
                @test status.error.type == "ErrorException"
                @test status.error.message == FAILURE.msg
                @test status.error.phase === :startup
                # A yielding interval after rollback cannot permit an orphaned
                # transform or observer to mutate the released session.
                sleep(0.01)
                @test backward.calls == 2 && backward.tasks == [owner, owner]
                @test vis.content.graph[] === nothing && ctl.last_publication == 0
                @test isempty(vis.content.output_buffer) && isempty(vis.content.pending_samples)
                @test isempty(vis.content.init_segments) && isempty(vis.content.invalid_init)
                return backward.work
            end
        end
    finally
        cleanup(vis)
    end
end

function interaction_failure()
    with_display() do _
        vis, figure = completed_fixture()
        observer = nothing
        work = Ref{Any}(nothing)
        try
            graph = EXT._live_graph(vis)
            @test EXT._live_graph_update!(vis; window_steps=(Int64(2), Int64(8)))
            observer = Makie.on(graph[:window_steps]) do window
                if isnothing(window)
                    work[] = caller_work()
                    throw(FAILURE)
                end
            end
            all_button = only(filter(x -> x isa Makie.Button && x.label[] == "All", figure.content))
            check_user_effects() do
                all_button.clicks[] += 1 # Real guarded BAT All entry point.
                join_release(vis)
                return work[]
            end
            status = BAT.visualizer_status(vis)
            @test status.outcome === :complete && status.data_complete
            @test !status.presentation_valid && !isnothing(status.error)
            @test status.history_released
        finally
            isnothing(observer) || Makie.off(observer)
            cleanup(vis)
        end
    end
end

function window_close()
    with_display() do _
        vis, state, outputs, target = fixture()
        observer = nothing
        work = Ref{Any}(nothing)
        try
            figure = initialize(vis, state, outputs, target)
            events(figure).window_open[] = true
            # This fixture-owned event observer runs before BAT's real close
            # listener. It is legitimate initiating-task work, not enrollment.
            observer = Makie.on(events(figure).window_open; priority=10) do opened
                if !opened
                    work[] = caller_work()
                end
                return Makie.Consume(false)
            end
            check_user_effects() do
                events(figure).window_open[] = false
                join_release(vis)
                return work[]
            end
            status = BAT.visualizer_status(vis)
            @test status.reason === :closed && status.outcome === :disabled
            @test status.history_released && !status.data_complete
        finally
            isnothing(observer) || Makie.off(observer)
            cleanup(vis)
        end
    end
end

# A fixture-owned subscription protocol, accepted by the production view's
# existing Any handles. Its cleanup can block/fail without redefining BAT.
mutable struct CleanupHandle{F}
    action::F
    calls::Int
end
CleanupHandle(action) = CleanupHandle(action, 0)
function Makie.Observables.off(handle::CleanupHandle)
    handle.calls += 1
    handle.action(handle.calls)
    return nothing
end
function view_fixture()
    vis = BATVisualizer(configured())
    EXT._claim_live_visualizer!(vis)
    figure = Makie.Figure()
    Makie.Axis(figure[1, 1])
    view = EXT._attach_live_view!(vis, figure)
    @test EXT._activate_live_visualizer!(vis)
    return vis, view
end

function busy_cleanup_retry()
    vis, view = view_fixture()
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
        private_before = copy(vis.content.lifecycle.task_rng)
        unchanged(() -> EXT._schedule_live_view_cleanup!(vis))
        @test vis.content.lifecycle.task_rng == private_before
        @test vis.content.lifecycle.disposal_task === nothing
        unchanged(() -> BAT.dispose_visualizer!(vis))
        task = vis.content.lifecycle.disposal_task
        waitfor(() -> isready(entered))
        take!(entered)
        @test view.cleanup_running[] && !istaskdone(task)
        private_after = copy(vis.content.lifecycle.task_rng)
        @test private_after != private_before
        unchanged() do
            EXT._schedule_live_view_cleanup!(vis)
            BAT.dispose_visualizer!(vis) # Real retry arriving during cleanup.
        end
        @test vis.content.lifecycle.disposal_task === task
        @test vis.content.lifecycle.task_rng == private_after
        put!(resume, nothing)
        unchanged(() -> join_release(vis, task))
        @test handle.calls == 2
        @test vis.content.lifecycle.task_rng == private_after
        @test BAT.visualizer_status(vis).cleanup_error.message == FAILURE.msg
    finally
        failing[] = false
        isready(resume) || put!(resume, nothing)
        cleanup(vis)
    end
end

function failed_coordinator_boundary()
    # Explicit terminal-owned-task fixture, not an injected exception in a
    # running production coordinator or a reproduced allocator/scheduler fault.
    vis = BATVisualizer(configured())
    EXT._claim_live_visualizer!(vis)
    task = @task throw(FAILURE)
    try
        BAT.request_visualizer_stop!(vis; reason=:disposed)
        locked(vis) do
            ctl = vis.content.lifecycle
            ctl.disposal_requested = true
            ctl.disposal_retry = false
            ctl.disposal_task = task
        end
        schedule(task)
        err = caught(() -> wait(task))
        @test err isa TaskFailedException && err.task === task
        @test istaskdone(task) && istaskfailed(task)
        private_before = copy(vis.content.lifecycle.task_rng)
        unchanged() do
            status = BAT.visualizer_status(vis)
            @test status.cleanup_pending && !status.history_released
            BAT.dispose_visualizer!(vis)
            replacement = vis.content.lifecycle.disposal_task
            @test replacement !== task
            join_release(vis, replacement)
        end
        @test vis.content.lifecycle.task_rng != private_before
        @test BAT.visualizer_status(vis).cleanup_error.type == "TaskFailedException"
    finally
        cleanup(vis)
    end
end

function enroll(f, vis)
    locked(vis) do
        Base.disable_sigint() do
            EXT._with_live_task_rng(f, vis.content.lifecycle)
        end
    end
end
function helper_failure_boundaries()
    # Supplemental actual-helper checks. These deliberately throw directly at
    # construction boundaries; they are neither public fault injection nor
    # native SIGINT coverage. Every created child is explicitly joined.
    for exception in (FAILURE, InterruptException()), phase in (:before, :constructed, :scheduled)
        @testset "$(typeof(exception)) at $phase" begin
            vis = BATVisualizer(configured())
            child = Ref{Union{Nothing,Task}}(nothing)
            consumed = Ref(copy(vis.content.lifecycle.task_rng))
            before = copy(Random.default_rng())
            private_before = copy(vis.content.lifecycle.task_rng)
            err = caught() do
                enroll(vis) do
                    if phase !== :before
                        child[] = @task rand(UInt64, 8)
                        phase === :scheduled && schedule(child[])
                    end
                    consumed[] = copy(Random.default_rng())
                    throw(exception)
                end
            end
            assert_rng_equal(before, copy(Random.default_rng()))
            @test err === exception
            @test vis.content.lifecycle.task_rng == consumed[]
            @test (vis.content.lifecycle.task_rng == private_before) == (phase === :before)
            if !isnothing(child[])
                phase === :constructed && schedule(child[])
                first_stream = fetch(child[])
                next_child = unchanged() do
                    enroll(vis) do
                        task = @task rand(UInt64, 8)
                        schedule(task)
                        task
                    end
                end
                @test fetch(next_child) != first_stream
                @test vis.content.lifecycle.task_rng != consumed[]
            end
        end
    end
end

@testset "BAT observer task RNG ownership" begin
    @test !isnothing(EXT)
    @testset "Completed and repeated disposal" begin
        completed_disposal()
    end
    @testset "Active disposal retains legitimate public callback randomness" begin
        plain = active_result(false)
        repeated = active_result(false)
        observed = active_result(true)
        @test isequal(repeated, plain)
        @test isequal(observed, plain)
    end
    @testset "Startup display failure retains callback randomness" begin
        startup_failure()
    end
    @testset "Actual serial startup transform failure retains caller effects" begin
        startup_transform_failure()
    end
    @testset "Real All failure retains callback randomness" begin
        interaction_failure()
    end
    @testset "Real window close retains caller event randomness" begin
        window_close()
    end
    @testset "Busy cleanup and failed subscription retry reuse enrollment" begin
        busy_cleanup_retry()
    end
    @testset "Terminal failed coordinator fixture permits replacement" begin
        failed_coordinator_boundary()
    end
    @testset "Supplemental actual-helper failure boundaries" begin
        helper_failure_boundaries()
    end
end
end
