# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Run release/diagnostics separately through test/optional/makie/runtests.jl.
# Every weak reference targets a mutable allocation. Construction, joining and
# deliberate external aliases have helper scopes so test locals are not owners.
module TestMakieHistoryRelease
using BAT, Makie, Test, Distributions, LinearAlgebra, Logging, Random
const EXT = Base.get_extension(BAT, :BATMakieExt)

struct NoRetentionDisplay <: AbstractDisplay end
Base.displayable(::NoRetentionDisplay, ::MIME"image/png") = true
Base.display(::NoRetentionDisplay, ::Makie.Figure) = nothing

function with_display(f)
    previous = Makie.ALWAYS_INLINE_PLOTS[]
    display = NoRetentionDisplay()
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
    BATMakieVisualization(c.recipes, [1, 2], 2, 20_000, 40_000, false,
        c.batch_growth_rate, 0.01, c.dark,
        merge(c.triagonal_config, (nbins=(24, 24),)),
        merge(c.diagonal_config, (nbins=24,)); observer_timeout=60.0)
end

function algorithm(; nsteps=48, callback=BAT.nop_func)
    TransformedMCMC(nchains=2, nsteps=nsteps, strict=false, store_burnin=true,
        init=MCMCChainPoolInit(nsteps_init=30), convergence=AssumeConvergence(), callback=callback,
        adaptive_transform=BAT.NoAdaptiveTransform(),
        transform_tuning=NoMCMCTransformTuning(),
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=16, max_ncycles=1, nsteps_final=3))
end

function mutable_reference(value)
    @test ismutable(value)
    @test !isempty(value)
    return WeakRef(value)
end

# Materialize several independent cache families in the same live graph. The
# restricted window also materializes selected samples separate from history.
function history_references(vis)
    graph = EXT._live_graph(vis)
    @test graph[:has_step_timing][]
    limit = graph[:max_step][]
    @test limit > 3
    @test EXT._live_graph_update!(vis; diagonal_recipe=Hist1D,
        upper_recipe=KDE2D, lower_recipe=Scatter2D, show_trace_upper=true,
        window_steps=(Int64(2), limit - 1))
    hist = graph[EXT.primitive_symbol(Hist1D(), (1, 1))][]
    kde = graph[EXT.primitive_symbol(KDE2D(), (2, 1))][]
    scatter = graph[EXT.primitive_symbol(Scatter2D(), (2, 1))][]
    trace = graph[EXT.primitive_symbol(Trace2D(), (2, 1))][]
    flat = graph[:flat_samples][]
    window = graph[:window_data][]
    return (
        graph=mutable_reference(graph.outputs),
        samples=mutable_reference(graph[:samples][][1][1].v.data.data),
        timing=mutable_reference(graph[:sample_timing][]),
        full_logd=mutable_reference(window.samples.logd),
        selected_logd=mutable_reference(flat.logd),
        selected_weights=mutable_reference(flat.weight),
        hist=mutable_reference(hist.weights),
        kde=mutable_reference(kde.density),
        scatter=mutable_reference(scatter.x),
        trace=mutable_reference(trace.x),
    )
end

Base.@noinline function completed_run(n; use_public=false)
    vis = BATVisualizer(configured())
    context = BATContext(rng=BAT.Philox4x((144, 72)), visualizer=vis)
    target = BAT.BATDistMeasure(MvNormal(zeros(2), ones(2)))
    if use_public
        result = BAT.bat_sample(target, algorithm(; nsteps=n), context)
        @test !isempty(result.result)
    else
        # Controlled production ingestion, with the real scalar sampler, state
        # extraction, listener, timing, final publication and rendering.
        state = BAT.MCMCState(algorithm(), target, Int32(1), [zeros(2)], context)
        outputs = BAT._empty_chain_outputs(state)
        BAT.init_visualizer!(vis; rng=BAT.get_rng(context), mcmc_states=[state],
            outputs=[outputs], target=target, f_pretransform=identity)
        BAT.mcmc_iterate!!(outputs, state; max_nsteps=n, update_visualizer=true)
        BAT.finish_visualizer!(vis; success=true)
    end
    status = BAT.visualizer_status(vis)
    @test status.outcome === :complete && status.data_complete
    @test status.quiesced && !status.cleanup_pending && !status.history_released
    figure = BAT.visualizer_figure(vis)
    @test figure isa Makie.Figure
    @test !isempty(figure.content)
    rows = sum(length, Iterators.flatten(EXT._live_graph(vis)[:samples][]))
    refs = history_references(vis)
    @test BAT.visualizer_figure(vis) === figure
    return (context=context, roots=refs, figure=Ref{Any}(figure), rows=rows,
        publications=status.last_publication)
end

Base.@noinline function dispose_and_join!(context)
    vis = context.visualizer
    BAT.dispose_visualizer!(vis)
    # Do not query status or call finish: the cleanup coordinator itself must
    # release the graph and terminate. Retire this local task before any GC.
    task = lock(() -> vis.content.lifecycle.disposal_task, vis.content.buffer_lock)
    @test task isa Task
    ready = timedwait(() -> istaskdone(task), 30.0; pollint=0.005) === :ok
    @test ready
    ready || error("History-release coordinator did not terminate")
    wait(task)
    @test fetch(task) === nothing
    @test vis.content.graph[] === nothing
    @test vis.content.lifecycle.task === nothing
    @test vis.content.lifecycle.timer === nothing
    @test vis.content.lifecycle.owner_task === nothing
    @test isempty(vis.content.output_buffer)
    @test isempty(vis.content.pending_samples)
    @test isempty(vis.content.output_timing)
    @test isempty(vis.content.pending_timing)
    return nothing
end

collect_twice() = (GC.gc(true); GC.gc(true); nothing)
alive(refs) = map(w -> w.value !== nothing, refs)

function assert_released(fixture)
    vis = fixture.context.visualizer
    # This is the first status read since disposal; the GC assertion has
    # already established autonomous release while the context stays alive.
    status = BAT.visualizer_status(vis)
    @test status.history_released && !status.cleanup_pending && status.quiesced
    @test status.outcome === :complete && status.data_complete
    @test status.last_publication == fixture.publications
    @test BAT.visualizer_figure(vis) === nothing
    @test isempty(fixture.figure[].content)
    @test isempty(vis.content.lifecycle.view.subscriptions)
    return nothing
end

Base.@noinline function graph_alias(fixture)
    Ref{Any}(EXT._live_graph(fixture.context.visualizer))
end
Base.@noinline function control_alias(fixture)
    # The actual textbox validator captures the graph independently of its
    # event subscriptions. Such caller-owned controls are outside release.
    Ref{Any}(only(filter(x -> x isa Makie.Textbox, fixture.figure[].content)))
end
Base.@noinline drop_alias!(holder) = (holder[] = nothing; nothing)

Base.@noinline function make_repeated_runs()
    [completed_run(n) for n in (256, 2048, 256, 2048)]
end
Base.@noinline function dispose_repeated!(fixtures)
    foreach(fixture -> dispose_and_join!(fixture.context), fixtures)
    return nothing
end

# Exact numerical snapshots for this finite, acyclic random-walk fixture.
# Exclude visualization handles and compare every numerical chain field rather
# than context/closure identity. No production sampler method is replaced.
function parity_structure(x)
    x isa Union{Number,Symbol,AbstractString,Nothing} && return x
    x isa Type && return (:type, string(x))
    x isa AbstractArray && return (:array, size(x), map(parity_structure, collect(vec(x))))
    x isa NamedTuple && return NamedTuple{keys(x)}(map(parity_structure, values(x)))
    x isa Tuple && return map(parity_structure, x)
    values = ntuple(i -> parity_structure(getfield(x, i)), fieldcount(typeof(x)))
    return (x isa Function ? :function : nameof(typeof(x)), values)
end
parity_structure(::Val{value}) where {value} = (:Val, value)
parity_samples(s) = (v=parity_structure(s.v), logd=copy(s.logd),
    weight=copy(s.weight), info=parity_structure(s.info), aux=parity_structure(s.aux))
parity_rng(rng) = (state=parity_structure(rng), next=rand(copy(rng), UInt64, 16))
function parity_chain(chain)
    return (info=parity_structure(chain.info), stepno=chain.stepno,
        nattempts=copy(chain.nattempts), nsamples=copy(chain.nsamples),
        accepted=copy(chain.accepted), walker_order=copy(chain.walker_order),
        current=map(parity_samples, chain.current), proposed=map(parity_samples, chain.proposed),
        output=parity_samples(chain.output), proposal=parity_structure(chain.proposal),
        transform=parity_structure(chain.f_transform), cycle_rng=parity_structure(chain.rngpart_cycle),
        context_rng=parity_rng(BAT.get_rng(chain.context)),
        walker_rngs=[parity_rng(BAT.get_rng(c)) for c in chain.walker_genctxs])
end

Base.@noinline function active_disposal_result(live)
    vis = live ? BATVisualizer(configured()) : BATVisualizer()
    context = BATContext(rng=BAT.Philox4x((0x1948, 0x3271)), visualizer=vis)
    events = NamedTuple[]
    owner = current_task()
    disposed = Ref(false)
    publications = Ref(0)
    callback = function(event, states)
        push!(events, (event=parity_structure(event), owner=current_task() === owner,
            states=[(chain=parity_chain(s.chain_state),
                proposal_tuner=parity_structure(s.proposal_tuner_state),
                transform_tuner=parity_structure(s.trafo_tuner_state),
                temperer=parity_structure(s.temperer_state)) for s in states]))
        if live && !disposed[]
            # This is the real public burn-in callback, after visualization
            # startup; it must not accidentally exercise the separate H19 gap.
            @test vis.content.lifecycle.phase === :running
            @test vis.content.lifecycle.accepting[]
            @test vis.content.lifecycle.task isa Task
            @test !istaskdone(vis.content.lifecycle.task)
            parent_rng = copy(Random.default_rng())
            dispose_and_join!(context) # Joins observer cleanup, never this owner.
            @test rand(copy(Random.default_rng()), UInt64, 16) == rand(parent_rng, UInt64, 16)
            publications[] = vis.content.lifecycle.last_publication
            disposed[] = true
        end
        return nothing
    end
    alg = algorithm(; nsteps=64, callback=callback)
    Random.seed!(0x5194)
    result = BAT.evalmeasure(MvNormal(zeros(2), ones(2)), alg, context)
    default_next = rand(copy(Random.default_rng()), UInt64, 16)
    chains = BAT.samplegenof(result).chain_states
    @test all(chain -> chain.stepno == 64, chains)
    @test length(events) == 1
    @test only(events).event == (:Val, :mcmc_burnin)
    @test only(events).owner
    if live
        @test disposed[]
        @test vis.content.graph[] === nothing
        @test !vis.content.lifecycle.accepting[]
        @test vis.content.lifecycle.last_publication == publications[]
        status = BAT.visualizer_status(vis)
        @test status.outcome === :disabled && status.reason === :disposed
        @test !status.data_complete && status.history_released
        @test status.quiesced && !status.cleanup_pending
        @test BAT.visualizer_figure(vis) === nothing
    end
    return (samples=parity_samples(BAT.samplesof(result)),
        chains=parity_chain.(chains), events=events,
        context_rng=parity_rng(BAT.get_rng(context)), default_next=default_next)
end

function release_case()
    @testset "Explicit active disposal preserves public sampling" begin
        expected = active_disposal_result(false)
        observed = active_disposal_result(true)
        @test isequal(observed.samples, expected.samples)
        @test isequal(observed.chains, expected.chains)
        @test isequal(observed.events, expected.events)
        @test isequal(observed.context_rng, expected.context_rng)
        @test observed.default_next == expected.default_next
    end

    @testset "Public completion retains an interactive borrowed figure" begin
        fixture = completed_run(48; use_public=true)
        GC.@preserve fixture begin
            collect_twice()
            @test all(alive(fixture.roots))
        end
        dispose_and_join!(fixture.context)
        GC.@preserve fixture begin
            collect_twice()
            @test all(!, alive(fixture.roots))
            @test isempty(fixture.figure[].content)
        end
        assert_released(fixture)
        BAT.dispose_visualizer!(fixture.context.visualizer)
        @test BAT.visualizer_status(fixture.context.visualizer).history_released
    end

    @testset "Retained contexts lose increasing and repeated histories" begin
        fixtures = make_repeated_runs()
        @test fixtures[2].rows > fixtures[1].rows
        @test fixtures[4].rows > fixtures[3].rows
        before = [Base.summarysize(f.context.visualizer.content) for f in fixtures]
        GC.@preserve fixtures begin
            collect_twice()
            @test all(f -> all(alive(f.roots)), fixtures)
        end
        dispose_repeated!(fixtures)
        GC.@preserve fixtures begin
            collect_twice()
            @test all(f -> all(!, alive(f.roots)), fixtures)
            foreach(assert_released, fixtures)
        end
        after = [Base.summarysize(f.context.visualizer.content) for f in fixtures]
        # Measurements are evidence, not allocator/RSS or universal size budgets.
        println("Retained visualizer content bytes before=", before, " after=", after)
        @test all(after .< before)
    end

    @testset "External graph and control aliases retain their own history" begin
        for alias_function in (graph_alias, control_alias)
            fixture = completed_run(256)
            alias = alias_function(fixture)
            dispose_and_join!(fixture.context)
            GC.@preserve alias fixture begin
                collect_twice()
                @test all(alive(fixture.roots))
                @test isempty(fixture.figure[].content)
            end
            assert_released(fixture)
            drop_alias!(alias)
            GC.@preserve fixture begin
                collect_twice()
                @test all(!, alive(fixture.roots))
            end
        end
    end
end

const FORMAT_CALLS = Ref(0)
struct PayloadError <: Exception
    payload::Vector{UInt8}
end
Base.showerror(::IO, ::PayloadError) =
    (FORMAT_CALLS[] += 1; error("User formatter must not run during record/status/release"))
struct HostileString <: AbstractString end
Base.ncodeunits(::HostileString) = (FORMAT_CALLS[] += 1; error("Custom string inspected"))
Base.String(::HostileString) = (FORMAT_CALLS[] += 1; error("Custom string converted"))

Base.@noinline function record_payloads!(vis)
    first = zeros(UInt8, 1_000_000)
    cleanup = zeros(UInt8, 1_000_000)
    BAT.request_visualizer_stop!(vis; reason=:failed,
        exception=CapturedException(PayloadError(first), Any[]))
    EXT._record_live_cleanup_error!(vis, PayloadError(cleanup), Any[])
    # Preserve first-error precedence for both diagnostic slots.
    BAT.request_visualizer_stop!(vis; reason=:aborted, exception=ErrorException("later error"))
    EXT._record_live_cleanup_error!(vis, ErrorException("later cleanup"), Any[])
    return (first=WeakRef(first), cleanup=WeakRef(cleanup))
end

Base.@noinline function record_task_failure!(vis)
    bytes = zeros(UInt8, 1_000_000)
    task = @task throw(PayloadError(bytes))
    schedule(task)
    try
        wait(task)
        error("Expected failed task")
    catch err
        @test err isa TaskFailedException
        BAT.request_visualizer_stop!(vis; reason=:failed, exception=CapturedException(err, Any[]))
    end
    return WeakRef(bytes)
end

Base.@noinline function record_resolve_failure!(vis)
    graph = EXT._live_graph(vis)
    Makie.map!(_ -> error("H14 intentional graph failure"), graph, :samples, :h14_fault)
    try
        graph[:h14_fault][]
        error("Expected failed graph computation")
    catch err
        @test err isa Makie.ComputePipeline.ResolveException
        EXT._fail_visualizer!(vis, err, catch_backtrace(), :h14_compute)
    end
    return nothing
end

function diagnostic_cases()
    @testset "Bounded inert summaries never execute user formatting" begin
        summarize(error) = EXT._live_error_summary(error, :test)
        @test summarize(ErrorException("first failure")).message == "first failure"
        @test summarize(ArgumentError("invalid option")).message == "invalid option"
        @test summarize(InterruptException()).type == "InterruptException"
        @test summarize(ErrorException(HostileString())).message isa String
        @test summarize(ArgumentError(HostileString())).message isa String
        parent = repeat("parent payload", 100_000)
        @test summarize(ErrorException(SubString(parent, 1, 10))).message != parent[1:10]
        for message in (repeat("x", 1_000_000), repeat("α🙂", 10_000),
            repeat("x", 4095) * "🙂tail", String(UInt8[0xff, 0x41, 0xc3, 0xa9]),
            String(fill(UInt8(0xff), 100_000)))
            result = summarize(ErrorException(message))
            @test result.message isa String
            @test ncodeunits(result.message) <= 4096 + 32
            @test ncodeunits(result.type) <= 256 + 32
            if isvalid(message)
                @test isvalid(result.message)
            end
            @test result.phase === :test
        end
        nested = CapturedException(CapturedException(PayloadError(UInt8[]), Any[]), Any[])
        @test summarize(nested).type == "CapturedException"
        # Type metadata can itself contain an unusually long name. Define this
        # test-only exception without overriding any production dispatch.
        name = Symbol(repeat("HistoryDiagnostic", 40))
        Core.eval(@__MODULE__, :(struct $name <: Exception end))
        long_type = Base.invokelatest(getfield, @__MODULE__, name)
        @test ncodeunits(summarize(Base.invokelatest(long_type)).type) <= 256 + 32
        @test FORMAT_CALLS[] == 0
    end

    @testset "Diagnostic payloads release at recording, including cleanup" begin
        for already_released in (false, true)
            vis = BATVisualizer(configured())
            context = BATContext(visualizer=vis)
            already_released && dispose_and_join!(context)
            refs = record_payloads!(vis)
            GC.@preserve context begin
                collect_twice()
                @test all(!, alive(refs))
                status = BAT.visualizer_status(vis)
                @test status.error.type == "PayloadError" && status.error.phase === :failed
                @test status.cleanup_error.type == "PayloadError"
            end
            already_released || dispose_and_join!(context)
            @test BAT.visualizer_status(vis).history_released
            @test vis.content.graph[] === nothing
        end
        @test FORMAT_CALLS[] == 0
    end

    @testset "Failed task does not survive through lifecycle diagnostics" begin
        vis = BATVisualizer(configured())
        context = BATContext(visualizer=vis)
        ref = record_task_failure!(vis)
        GC.@preserve context begin
            collect_twice()
            @test ref.value === nothing
            @test BAT.visualizer_status(vis).error.type == "TaskFailedException"
        end
        dispose_and_join!(context)
        @test BAT.visualizer_status(vis).history_released
        @test FORMAT_CALLS[] == 0
    end

    @testset "Real ResolveException cannot retain a disposed graph" begin
        fixture = completed_run(256)
        record_resolve_failure!(fixture.context.visualizer)
        dispose_and_join!(fixture.context)
        GC.@preserve fixture begin
            collect_twice()
            @test all(!, alive(fixture.roots))
        end
        assert_released(fixture)
        status = BAT.visualizer_status(fixture.context.visualizer)
        @test status.error.type == "ResolveException"
        @test status.error.phase === :h14_compute
        @test status.error.message isa String
    end
end

case = get(ENV, "BAT_MAKIE_HISTORY_CASE", "")
case in ("release", "diagnostics") || error("Run via the optional Makie subprocess runner")
@testset "Makie history ownership: $case" begin
    with_display() do
        case == "release" ? release_case() : diagnostic_cases()
    end
end
end
