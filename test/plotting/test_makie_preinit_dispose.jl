# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Process-bounded optional `preinit_dispose` case. Numerical likelihood barriers
# exercise the actual public sampler; no production hook is replaced. Status
# queries never drive the release-before-resume assertion.
module TestMakiePreinitDispose

using BAT, Test, Makie, Distributions, Random, Logging

const EXT = Base.get_extension(BAT, :BATMakieExt)
const FAILURE = ErrorException("intentional density failure after pre-view disposal")
const INTERRUPTION = InterruptException()

mutable struct Capture{F} <: AbstractDisplay
    figures::Vector{Figure}
    action::F
end
Base.displayable(::Capture, ::MIME"image/png") = true
function Base.display(capture::Capture, figure::Figure)
    push!(capture.figures, figure)
    capture.action()
    return nothing
end

function with_capture(f; action=() -> nothing)
    capture = Capture(Figure[], action)
    previous = Makie.ALWAYS_INLINE_PLOTS[]
    Makie.inline!(true)
    pushdisplay(capture)
    try
        return with_logger(() -> f(capture), NullLogger())
    finally
        popdisplay(capture)
        Makie.inline!(previous)
    end
end

function waitfor(predicate; seconds=30.0)
    ready = timedwait(predicate, seconds; pollint=0.005) === :ok
    @test ready
    ready || error("Pre-view disposal fixture timed out")
    return nothing
end

function join_owned(task)
    @test task isa Task
    task isa Task || error("Pre-view disposal fixture lost an owned task")
    waitfor(() -> istaskdone(task))
    @test !istaskfailed(task)
    wait(task)
    return nothing
end

caught(f) = try
    f()
    nothing
catch err
    err
end

locked(f, vis) = lock(f, vis.content.buffer_lock)
coordinator(vis) = locked(() -> vis.content.lifecycle.disposal_task, vis)

function join_release(vis, task=coordinator(vis))
    join_owned(task)
    @test fetch(task) === nothing
    @test isnothing(vis.content.graph[])
    @test isnothing(vis.content.lifecycle.owner_task)
    @test isnothing(vis.content.lifecycle.task)
    @test isnothing(vis.content.lifecycle.timer)
    @test !vis.content.lifecycle.accepting[]
    return nothing
end

function cleanup(vis)
    # Only this fixture's observer is touched, after the outcome assertions.
    BAT.request_visualizer_stop!(vis; reason=:disabled)
    BAT.finish_visualizer!(vis; success=false)
    BAT.dispose_visualizer!(vis)
    task = coordinator(vis)
    isnothing(task) || join_owned(task)
    return nothing
end

function configuration()
    c = BATMakieVisualization(adaptive_batching=false)
    BATMakieVisualization((upper=Hist2D, diagonal=Hist1D, lower=Hist2D),
        [1, 2], 2, 20_000, 40_000, false, c.batch_growth_rate, 0.01, c.dark,
        merge(c.triagonal_config, (nbins=(8, 8),)),
        merge(c.diagonal_config, (nbins=8,)); observer_timeout=30.0)
end

function algorithm(; init=:pool, store_burnin=true, callback=BAT.nop_func, pretransform=NormalBased())
    initialization = init === :pool ? MCMCChainPoolInit(nsteps_init=20) :
        MCMCRetryInit(nsteps_init=20, max_init_tries=4)
    TransformedMCMC(nchains=2, nsteps=16, strict=false, store_burnin=store_burnin,
        init=initialization, callback=callback, pretransform=pretransform, convergence=AssumeConvergence(),
        adaptive_transform=BAT.NoAdaptiveTransform(),
        transform_tuning=NoMCMCTransformTuning(),
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=12, max_ncycles=1, nsteps_final=3))
end

function chain_snapshot(chain)
    (current=deepcopy(chain.current), proposed=deepcopy(chain.proposed),
        output=deepcopy(chain.output), accepted=copy(chain.accepted),
        order=copy(chain.walker_order), info=chain.info, stepno=chain.stepno,
        nattempts=copy(chain.nattempts), nsamples=copy(chain.nsamples),
        rng=copy(BAT.get_rng(chain.context)),
        walker_rngs=[copy(BAT.get_rng(context)) for context in chain.walker_genctxs])
end

mutable struct DensityGate
    lock::ReentrantLock
    entered::Channel{Any}
    resume::Base.Event
    finished::Threads.Atomic{Bool}
    selected::Bool
    mode::Symbol
    first_pass_count::Int
    fail::Bool
    workers::Vector{Task}
    observed_cycles::Set{Int32}
    released_before_resume::Bool
end
DensityGate(mode, first_pass_count, fail) = DensityGate(ReentrantLock(), Channel{Any}(1),
    Base.Event(), Threads.Atomic{Bool}(false), false, mode, first_pass_count, fail,
    Task[], Set{Int32}(), false)

# The prior/density is ordinary finite Gaussian work. The only scheduling seam
# is an owned event inside the actual likelihood. Task identity separates the
# numerical owner from genuine mcmc_iterate!! initialization workers.
function gated_likelihood(vis, gate, x)
    if !isnothing(vis)
        state = locked(vis) do
            ctl = vis.content.lifecycle
            permit = ctl.startup_owner
            owner = isnothing(permit) ? nothing : permit.value
            cycles = Int32[s.cycle for segments in values(vis.content.init_segments) for s in segments]
            (owner=owner, phase=ctl.phase, initializing=ctl.initializing,
                count=length(vis.content.init_segments), cycles=cycles,
                capture=BAT.capture_visualizer_init(vis))
        end
        selected = lock(gate.lock) do
            union!(gate.observed_cycles, state.cycles)
            if state.owner isa Task && current_task() !== state.owner
                current_task() in gate.workers || push!(gate.workers, current_task())
            end
            eligible = state.phase === :starting && !state.initializing &&
                state.owner isa Task &&
                (gate.mode === :owner ? current_task() === state.owner :
                    current_task() !== state.owner &&
                    (gate.mode === :worker || state.count >= gate.first_pass_count))
            if !gate.selected && eligible
                gate.selected = true
                true
            else
                false
            end
        end
        if selected
            put!(gate.entered, merge(state, (task=current_task(),)))
            wait(gate.resume)
            gate.fail && throw(FAILURE)
        end
    end
    return -sum(abs2, x) / 2
end

# Keep this measure-owned failure path alongside the generic likelihood
# regression tests in test_abstract_density.jl. Both must retain the actual
# cause through EvalException, including after visualizer disposal.
struct GatedMeasure{M,V} <: BAT.BATMeasure
    base::M
    vis::V
    gate::DensityGate
end
BAT.varshape(m::GatedMeasure) = BAT.varshape(m.base)
BAT.getdof(m::GatedMeasure) = BAT.getdof(m.base)
BAT.unshaped(m::GatedMeasure) = m
BAT.get_initsrc_from_target(m::GatedMeasure) = BAT.get_initsrc_from_target(m.base)
BAT.logdensityof(m::GatedMeasure, x) =
    gated_likelihood(m.vis, m.gate, x) + BAT.logdensityof(m.base, x)

function contains_failure(err)
    err === FAILURE && return true
    err isa BAT.EvalException && return contains_failure(err.ret)
    err isa CompositeException && return any(contains_failure, err.exceptions)
    err isa TaskFailedException && return any(
        frame -> contains_failure(frame.exception), Base.current_exceptions(err.task))
    return false
end

function assert_disabled(vis)
    status = BAT.visualizer_status(vis)
    @test status.outcome === :disabled
    @test status.reason === :disposed
    @test status.quiesced && status.history_released && !status.cleanup_pending
    @test !status.data_complete
    @test status.last_publication == 0
    @test isnothing(vis.content.lifecycle.view)
    @test isnothing(vis.content.lifecycle.startup_owner)
    @test isempty(vis.content.init_segments)
    @test isempty(vis.content.invalid_init)
    @test !BAT.capture_visualizer_init(vis)
    @test isempty(vis.content.chain_ids)
    return nothing
end

function public_run(; observed, init=:pool, store_burnin=true, mode=:owner,
    release_first=true, fail=false)
    vis = observed ? BATVisualizer(configuration()) : nothing
    context = observed ? BATContext(rng=BAT.Philox4x((144, 72)), visualizer=vis) :
        BATContext(rng=BAT.Philox4x((144, 72)))
    events = Any[]
    callback = (tag, states) -> (push!(events,
        (tag=tag, states=chain_snapshot.(getproperty.(states, :chain_state)))); nothing)
    alg = algorithm(; init=init, store_burnin=store_burnin, callback=callback,
        pretransform=fail ? DoNotTransform() : NormalBased())
    first_count = init === :pool ? minimum(alg.init.init_tries_per_chain) * alg.nchains : alg.nchains
    gate = DensityGate(mode, first_count, fail)
    likelihood = x -> gated_likelihood(vis, gate, x)
    target = fail ? GatedMeasure(BAT.batmeasure(MvNormal(zeros(2), ones(2))), vis, gate) :
        PosteriorMeasure(BAT.logfuncdensity(likelihood), MvNormal(zeros(2), ones(2)))
    release_task = Ref{Union{Nothing,Task}}(nothing)
    controller = observed ? @async(begin
        try
            waitfor(() -> isready(gate.entered) || gate.finished[])
            if isready(gate.entered)
                entry = take!(gate.entered)
                @test entry.phase === :starting
                @test !entry.initializing
                @test entry.capture == store_burnin
                @test entry.owner === vis.content.lifecycle.owner_task
                @test Threads.threadid() == vis.content.lifecycle.owner_thread
                if mode !== :owner
                    @test entry.task !== entry.owner
                    @test !istaskdone(entry.task)
                end
                if mode === :second_pass
                    @test entry.count >= first_count
                    @test !isempty(entry.cycles) && all(==(Int32(1)), entry.cycles)
                end
                # This controller, rather than the numerical owner, enrolls
                # cleanup. Its complete RNG must be preserved as well.
                controller_rng = copy(Random.default_rng())
                BAT.dispose_visualizer!(vis)
                @test Random.default_rng() == controller_rng
                release_task[] = coordinator(vis)
                private_after = copy(vis.content.lifecycle.task_rng)
                BAT.dispose_visualizer!(vis)
                @test Random.default_rng() == controller_rng
                @test vis.content.lifecycle.task_rng == private_after
                if release_first
                    join_release(vis, release_task[])
                    # These assertions precede the density's event release and
                    # cannot be rescued by public sampler finalization.
                    @test !istaskdone(entry.task)
                    @test BAT.capture_visualizer_init(vis) == store_burnin
                    @test locked(() -> length(vis.content.init_segments), vis) >= entry.count
                    @test vis.content.lifecycle.startup_owner.value === entry.owner
                    gate.released_before_resume = true
                end
            end
        finally
            notify(gate.resume)
        end
        nothing
    end) : nothing

    result = nothing
    exception = nothing
    default_next = UInt64[]
    default_state = nothing
    default_children = Vector{Vector{UInt64}}(undef, 3)
    try
        with_capture() do capture
            # Fixture controller construction is outside the default-RNG
            # comparison. Production task creation remains inside it.
            Random.seed!(8921)
            try
                result = BAT.evalmeasure(target, alg, context)
            catch err
                exception = err
            finally
                gate.finished[] = true
            end
            default_state = copy(Random.default_rng())
            default_next = rand(copy(default_state), UInt64, 16)
            @sync for i in eachindex(default_children)
                Threads.@spawn default_children[i] = rand(UInt64, 16)
            end
            @test isempty(capture.figures)
        end
        if observed
            join_owned(controller)
            @test gate.selected
            @test gate.released_before_resume == release_first
            isnothing(release_task[]) || join_release(vis, release_task[])
            @test all(istaskdone, gate.workers)
            mode !== :owner && @test !isempty(gate.workers)
            assert_disabled(vis)
            @test_throws ArgumentError BAT.init_visualizer!(vis;
                rng=UncopyableRNG(), mcmc_states=BAT.MCMCState[], outputs=Any[],
                f_pretransform=identity)
            if fail
                @test contains_failure(exception)
                @test isnothing(result)
                @test !isnothing(BAT.visualizer_status(vis).error)
            else
                @test isnothing(exception)
                @test isnothing(BAT.visualizer_status(vis).error)
            end
        end
        if !fail
            @test isnothing(exception)
            isnothing(exception) || throw(exception)
            return (samples=BAT.samplesof(result),
                chains=chain_snapshot.(BAT.samplegenof(result).chain_states),
                events=events, context_rng=copy(BAT.get_rng(context)), default_next=default_next,
                default_state=default_state, default_children=default_children,
                cycles=copy(gate.observed_cycles))
        end
        return nothing
    finally
        gate.finished[] = true
        notify(gate.resume)
        if observed
            try
                join_owned(controller)
            finally
                cleanup(vis)
            end
        end
    end
end

function fixture()
    vis = BATVisualizer(configuration())
    context = BATContext(rng=BAT.Philox4x((71, 214)), visualizer=vis)
    target = BAT.BATDistMeasure(MvNormal(zeros(2), ones(2)))
    state = BAT.MCMCState(algorithm(), target, Int32(1), [zeros(2)], context)
    return vis, state, BAT._empty_chain_outputs(state), target
end

function initialize(vis, state, outputs, target; rng=BAT.get_rng(state.chain_state.context))
    BAT.init_visualizer!(vis; rng=rng, mcmc_states=[state], outputs=[outputs],
        target=target, f_pretransform=identity)
end

struct UncopyableRNG <: AbstractRNG end
Base.copy(::UncopyableRNG) = error("Disabled startup must not copy its RNG")
struct FailingPreparation end
Base.getproperty(::FailingPreparation, ::Symbol) = throw(FAILURE)
struct InterruptingPreparation end
Base.getproperty(::InterruptingPreparation, ::Symbol) = throw(INTERRUPTION)
struct InterruptingRNG <: AbstractRNG end
Base.copy(::InterruptingRNG) = throw(INTERRUPTION)

function manual_cases()
    @testset "Direct fresh startup and rejected reentrant startup" begin
        vis, state, outputs, target = fixture()
        reentered = Ref(false)
        try
            BAT.finish_visualizer!(vis)
            @test vis.content.lifecycle.phase === :fresh
            action = function()
                reentered[] = true
                @test vis.content.lifecycle.initializing
                @test isnothing(vis.content.lifecycle.startup_owner)
                @test_throws ArgumentError initialize(vis, state, outputs, target)
                @test vis.content.lifecycle.initializing
                nothing
            end
            with_capture(; action=action) do capture
                initialize(vis, state, outputs, target)
                @test length(capture.figures) == 1
            end
            @test reentered[]
            @test !vis.content.lifecycle.initializing
            @test_throws ArgumentError initialize(vis, state, outputs, target)
            @test_throws ArgumentError BAT.prepare_visualizer_init!(vis, algorithm())
            @test vis.content.lifecycle.phase === :running
        finally
            cleanup(vis)
        end
    end

    @testset "Explicit preparation supports later same-owner view startup" begin
        vis, state, outputs, target = fixture()
        try
            BAT.prepare_visualizer_init!(vis, algorithm())
            yield()
            @test vis.content.lifecycle.startup_owner.value === current_task()
            with_capture() do capture
                initialize(vis, state, outputs, target)
                @test length(capture.figures) == 1
            end
            @test isnothing(vis.content.lifecycle.startup_owner)
            @test vis.content.lifecycle.phase === :running
            @test_throws ArgumentError initialize(vis, state, outputs, target)
        finally
            cleanup(vis)
        end
    end

    @testset "Released same-owner continuation is single use" begin
        vis, state, outputs, target = fixture()
        try
            BAT.prepare_visualizer_init!(vis, algorithm())
            owner = current_task()
            @test vis.content.lifecycle.startup_owner.value === owner
            BAT.dispose_visualizer!(vis)
            join_release(vis)
            # A foreign task must not consume the legitimate weak permit or
            # run the startup owner's finally/cleanup path.
            foreign = @async caught(() -> initialize(vis, state, outputs, target))
            join_owned(foreign)
            @test fetch(foreign) isa ArgumentError
            @test vis.content.lifecycle.startup_owner.value === owner
            @test !vis.content.lifecycle.initializing
            @test BAT.capture_visualizer_init(vis)
            @test BAT.visualizer_status(vis).history_released
            with_capture() do capture
                @test initialize(vis, state, outputs, target; rng=UncopyableRNG()) === nothing
                @test isempty(capture.figures)
            end
            @test isnothing(vis.content.lifecycle.startup_owner)
            @test_throws ArgumentError initialize(vis, state, outputs, target)
            @test_throws ArgumentError BAT.prepare_visualizer_init!(vis, algorithm())
            # Initialization metadata remains the numerical owner's concern;
            # a no-view init alone must not clear it opportunistically.
            @test BAT.capture_visualizer_init(vis)
            BAT.clear_visualizer_init!(vis)
            assert_disabled(vis)
        finally
            cleanup(vis)
        end
    end

    @testset "Explicit benign stop permits one no-view continuation" begin
        for reason in (:closed, :disabled)
            vis, state, outputs, target = fixture()
            try
                BAT.prepare_visualizer_init!(vis, algorithm())
                BAT.request_visualizer_stop!(vis; reason=reason)
                @test vis.content.lifecycle.outcome === :disabled
                @test vis.content.lifecycle.reason === reason
                with_capture() do capture
                    @test initialize(vis, state, outputs, target; rng=UncopyableRNG()) === nothing
                    @test isempty(capture.figures)
                end
                @test isnothing(vis.content.lifecycle.startup_owner)
                @test !vis.content.lifecycle.initializing
                @test_throws ArgumentError initialize(vis, state, outputs, target)
            finally
                cleanup(vis)
            end
        end
    end

    @testset "Fresh disposal is not an acquired startup" begin
        vis, state, outputs, target = fixture()
        try
            BAT.dispose_visualizer!(vis)
            join_release(vis)
            @test_throws ArgumentError initialize(vis, state, outputs, target)
            @test_throws ArgumentError BAT.prepare_visualizer_init!(vis, algorithm())
        finally
            cleanup(vis)
        end
    end

    @testset "Wrong-thread attempts cannot spend the owner's permit" begin
        vis, state, outputs, target = fixture()
        try
            BAT.prepare_visualizer_init!(vis, algorithm())
            BAT.dispose_visualizer!(vis)
            join_release(vis)
            owner_thread = vis.content.lifecycle.owner_thread
            errors = Channel{Any}(Threads.nthreads(:default))
            Threads.@threads :static for _ in 1:Threads.nthreads(:default)
                if Threads.threadid() != owner_thread
                    put!(errors, caught(() -> initialize(vis, state, outputs, target)))
                end
            end
            close(errors)
            answers = collect(errors)
            if isempty(answers)
                @test_skip "wrong-thread startup requires another default thread"
            else
                @test all(err -> err isa ArgumentError, answers)
            end
            @test vis.content.lifecycle.startup_owner.value === current_task()
            @test initialize(vis, state, outputs, target; rng=UncopyableRNG()) === nothing
        finally
            cleanup(vis)
        end
    end

    @testset "Non-benign stops and terminal hooks revoke pending startup" begin
        for operation in (:aborted, :failed, :cleanup_failed, :exception, :clear, :finish_ok, :finish_failed)
            @testset "$operation" begin
                vis, state, outputs, target = fixture()
                try
                    BAT.prepare_visualizer_init!(vis, algorithm())
                    BAT.dispose_visualizer!(vis)
                    join_release(vis)
                    if operation === :exception
                        BAT.request_visualizer_stop!(vis; reason=:disabled, exception=FAILURE)
                    elseif operation === :clear
                        BAT.clear_visualizer_init!(vis)
                    elseif operation in (:finish_ok, :finish_failed)
                        BAT.finish_visualizer!(vis; success=operation === :finish_ok)
                    else
                        BAT.request_visualizer_stop!(vis; reason=operation)
                    end
                    # The first outcome remains disabled, but permission must
                    # not be inferred from that sticky outcome alone.
                    @test vis.content.lifecycle.outcome === :disabled
                    @test isnothing(vis.content.lifecycle.startup_owner)
                    @test_throws ArgumentError initialize(vis, state, outputs, target)
                finally
                    cleanup(vis)
                end
            end
        end
    end

    # These deterministic throws exercise installed cleanup scopes; they do
    # not emulate native delivery at an arbitrary instruction in a SIGINT mask.
    @testset "Preparation and admitted startup preserve interruption" begin
        for stage in (:prepare, :startup)
            vis, state, outputs, target = fixture()
            try
                # Prepared startup isolates admitted=true, claimed=false.
                stage === :startup && BAT.prepare_visualizer_init!(vis, algorithm())
                err, trace = with_logger(NullLogger()) do
                    try
                        stage === :prepare ?
                            BAT.prepare_visualizer_init!(vis, InterruptingPreparation()) :
                            initialize(vis, state, outputs, target; rng=InterruptingRNG())
                        (nothing, nothing)
                    catch exception
                        (exception, catch_backtrace())
                    end
                end
                @test err === INTERRUPTION
                origin = stage === :prepare ? :getproperty : :copy
                @test any(frame -> frame.func === origin &&
                    endswith(String(frame.file), "test_makie_preinit_dispose.jl"), stacktrace(trace))
                status = BAT.visualizer_status(vis)
                @test status.outcome === :failed
                @test status.error.phase === stage
                @test !vis.content.lifecycle.initializing
                @test isnothing(vis.content.lifecycle.startup_owner)
                task = coordinator(vis)
                isnothing(task) || join_owned(task)
                @test BAT.visualizer_status(vis).history_released
                @test isnothing(vis.content.graph[])
                @test isempty(vis.content.init_segments)
                @test !BAT.capture_visualizer_init(vis)
                @test_throws ArgumentError initialize(vis, state, outputs, target)
            finally
                cleanup(vis)
            end
        end
    end

    @testset "Failed preparation consumes the claimed session" begin
        vis, state, outputs, target = fixture()
        try
            with_logger(NullLogger()) do
                @test caught(() -> BAT.prepare_visualizer_init!(vis, FailingPreparation())) === FAILURE
            end
            @test isnothing(vis.content.lifecycle.startup_owner)
            @test isempty(vis.content.init_segments)
            @test !BAT.capture_visualizer_init(vis)
            @test_throws ArgumentError initialize(vis, state, outputs, target)
            @test_throws ArgumentError BAT.prepare_visualizer_init!(vis, algorithm())
        finally
            cleanup(vis)
        end
    end
    return nothing
end

# A completed task's result is an unambiguous strong retention control. Allocate
# it inside the task and leave this helper before collecting: only the explicit
# alias below should own that result after observer disposal.
Base.@noinline function abandoned_preparation()
    vis = BATVisualizer(configuration())
    context = BATContext(rng=BAT.Philox4x((4, 19)), visualizer=vis)
    entered = Channel{WeakRef}(1)
    resume = Base.Event()
    owner = @async begin
        payload = zeros(UInt8, 128_000)
        BAT.prepare_visualizer_init!(vis, algorithm())
        put!(entered, WeakRef(payload))
        wait(resume)
        payload
    end
    completed = false
    try
        waitfor(() -> isready(entered))
        root = take!(entered)
        BAT.dispose_visualizer!(vis)
        join_release(vis)
        @test !istaskdone(owner)
        @test vis.content.lifecycle.startup_owner.value === owner
        notify(resume)
        join_owned(owner)
        # Intentionally no init/finish: this abandoned internal acquisition
        # must not retain a completed task solely to preserve its identity.
        completed = true
        return (context=context, root=root, alias=Ref{Any}(owner))
    finally
        notify(resume)
        try
            join_owned(owner)
        finally
            completed || cleanup(vis)
        end
    end
end

Base.@noinline drop_task_alias!(fixture) = (fixture.alias[] = nothing; nothing)
Base.@noinline payload_alive(fixture) = !isnothing(fixture.root.value)
collect_twice() = (GC.gc(true); GC.gc(true); nothing)

@testset "Pre-view disposal preserves the same numerical acquisition" begin
    # Establish the owner's blocking-wait scheduling condition before seeded
    # campaigns or controller construction. A runtime may lazily create its
    # per-thread scheduler Task on the first empty-queue wait. This ordinary
    # callback-free Base wait is outside RNG measurements and does not warm a
    # renderer, rewind user work, or assume a particular Julia version.
    @test timedwait(() -> false, 0.02; pollint=0.001) === :timed_out
    for case in ((init=:pool, store_burnin=true, mode=:owner, release_first=true),
        (init=:retry, store_burnin=true, mode=:worker, release_first=true),
        (init=:pool, store_burnin=false, mode=:owner, release_first=false),
        (init=:pool, store_burnin=true, mode=:second_pass, release_first=true))
        @testset "$case" begin
            expected = public_run(; observed=false, case...)
            repeated = public_run(; observed=false, case...)
            observed = public_run(; observed=true, case...)
            for (name, result) in (("plain/plain harness control", repeated),
                                   ("observed/plain public sampling", observed))
                @testset "$name" begin
                    @test result.samples == expected.samples
                    @test result.chains == expected.chains
                    @test result.events == expected.events
                    @test result.context_rng == expected.context_rng
                    @test result.default_next == expected.default_next
                    @test result.default_state == expected.default_state
                    @test result.default_children == expected.default_children
                end
            end
            if case.mode === :second_pass
                @test Int32(1) in observed.cycles
            end
        end
    end
end

@testset "Numerical failure after disposal retains its actual cause" begin
    public_run(; observed=true, init=:pool, mode=:owner, fail=true)
    public_run(; observed=true, init=:retry, mode=:worker, fail=true)
end

@testset "Pre-view continuation ownership" begin
    manual_cases()
end

@testset "A pending weak permit does not retain an abandoned owner result" begin
    fixture = abandoned_preparation()
    try
        GC.@preserve fixture begin
            collect_twice()
            @test payload_alive(fixture) # explicit task alias negative control
        end
        drop_task_alias!(fixture)
        GC.@preserve fixture begin
            collect_twice()
            @test !payload_alive(fixture)
            @test isnothing(fixture.context.visualizer.content.lifecycle.owner_task)
            @test isnothing(fixture.context.visualizer.content.graph[])
        end
    finally
        cleanup(fixture.context.visualizer)
    end
end

end # module
