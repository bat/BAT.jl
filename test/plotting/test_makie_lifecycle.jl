# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Each named case is run in its own externally bounded process by
# test/optional/makie/runtests.jl. Barriers below select actual production
# interleavings; emergency cleanup runs only after the outcome assertions.
module TestMakieLifecycle
using BAT, Test, Makie, Distributions, LinearAlgebra, Random
const EXT = Base.get_extension(BAT, :BATMakieExt)
const FAILURE = ErrorException("intentional lifecycle regression failure")

mutable struct Capture <: AbstractDisplay
    figures::Vector{Makie.Figure}
    fail::Bool
end
Base.displayable(::Capture, ::MIME"image/png") = true
function Base.display(c::Capture, fig::Makie.Figure)
    push!(c.figures, fig)
    c.fail && throw(FAILURE)
    nothing
end
function capture(f; fail=false)
    c = Capture(Makie.Figure[], fail)
    old = Makie.ALWAYS_INLINE_PLOTS[]
    Makie.inline!(true)
    pushdisplay(c)
    try
        return f(c)
    finally
        popdisplay(c)
        Makie.inline!(old)
    end
end
function config(; batch=4, limit=8, timeout=60.0, poll=0.01)
    c = BATMakieVisualization(adaptive_batching=false, observer_timeout=timeout)
    BATMakieVisualization(c.recipes, [1,2], 2, batch, limit, false,
        c.batch_growth_rate, poll, c.dark, c.triagonal_config, c.diagonal_config;
        observer_timeout=timeout)
end
function algorithm(; callback=BAT.nop_func, convergence=AssumeConvergence())
    TransformedMCMC(nchains=2, nsteps=32, strict=false, store_burnin=true,
        init=MCMCChainPoolInit(nsteps_init=20), callback=callback,
        convergence=convergence, adaptive_transform=BAT.NoAdaptiveTransform(),
        transform_tuning=NoMCMCTransformTuning(),
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=12,max_ncycles=1,nsteps_final=3))
end
function fixture(; backend=config(), target=BAT.BATDistMeasure(MvNormal(zeros(2),ones(2))))
    vis = BATVisualizer(backend)
    ctx = BATContext(rng=BAT.Philox4x((144,72)),visualizer=vis)
    state = BAT.MCMCState(algorithm(),target,Int32(1),[zeros(2)],ctx)
    vis, state, BAT._empty_chain_outputs(state), target
end
function initialize(vis,state,outputs,target)
    capture() do c
        BAT.init_visualizer!(vis; rng=BAT.get_rng(state.chain_state.context),
            mcmc_states=[state],outputs=[outputs],target=target,f_pretransform=identity)
        @test length(c.figures) == 1
        return only(c.figures)
    end
end
function cleanup(vis)
    BAT.request_visualizer_stop!(vis;reason=:disabled)
    BAT.finish_visualizer!(vis;success=false)
    BAT.dispose_visualizer!(vis)
    timedwait(() -> !BAT.visualizer_status(vis).cleanup_pending, 5.0; pollint=0.01)
end
function waitfor(predicate; seconds=10.0)
    ok = timedwait(predicate,seconds;pollint=0.005) == :ok
    @test ok
    ok || error("Lifecycle test barrier timed out")
    nothing
end
function caught(f)
    try
        f()
        nothing
    catch err
        err
    end
end
function assert_failed(vis)
    waitfor(() -> BAT.visualizer_status(vis).quiesced &&
        !BAT.visualizer_status(vis).cleanup_pending)
    s = BAT.visualizer_status(vis)
    @test s.outcome == :failed
    @test !s.data_complete
    @test !isnothing(s.error)
    @test s.quiesced
    @test !s.cleanup_pending
end
function assert_complete(vis)
    s = BAT.visualizer_status(vis)
    @test s.outcome == :complete
    @test s.data_complete
    @test s.quiesced
    @test !s.cleanup_pending
    @test isnothing(s.error)
end
# Register a real sample transport without constructing a figure. The production
# listener owns the supplied callables, including wakeup, failure and finalization.
function listener_fixture(tick!, finish!; backend=config())
    vis,state,outputs,target = fixture(;backend)
    EXT._claim_live_visualizer!(vis)
    vis.content.n_dof[]=2
    EXT.register_state_for_vis!(vis,state,outputs)
    EXT._start_live_listener!(vis,() -> tick!(vis,state),() -> finish!(vis,state))
    return vis,state,outputs,target
end
function packet!(vis,state; step=2)
    chain=state.chain_state
    chain.stepno=step
    chain.output.v[1] .= [Float64(step), -Float64(step)]
    chain.output.weight[1]=1
    chain.output.logd[1]=-1
    chain.output.info[1]=BAT.MCMCSampleID(Int32(1),Int32(1),Int32(1),Int64(step-1),Int32(1),true)
    BAT.update_visualizer!(vis;chain_state=chain,nonzero_weights=true,is_sampling_step=true)
end

function healthy()
    vis=BATVisualizer(config())
    target=MvNormal(zeros(2),ones(2))
    ctx=BATContext(rng=BAT.Philox4x((12,34)),visualizer=vis)
    try
        capture() do c
            result=BAT.evalmeasure(target,algorithm(),ctx)
            @test !isempty(BAT.samplesof(result))
            assert_complete(vis)
            @test isnothing(vis.content.lifecycle.shutdown_started)
            @test BAT.visualizer_status(vis).last_publication > 0
            @test !BAT.visualizer_status(vis).view_disposed
            @test EXT._live_graph_update!(vis;show_stats_upper=true)
            @test vis.content.graph[:show_stats_upper][]
            @test !isempty(Makie.colorbuffer(only(c.figures)))
            BAT.finish_visualizer!(vis;success=true)
            assert_complete(vis)
            # A later control failure invalidates presentation, not historical
            # observation completeness or the final successful publication.
            published=BAT.visualizer_status(vis).last_publication
            observer=Makie.on(vis.content.graph[:show_stats_lower]) do _
                throw(FAILURE)
            end
            try
                @test !EXT._live_graph_update!(vis;show_stats_lower=true)
                @test BAT.visualizer_figure(vis)===nothing
                status=BAT.visualizer_status(vis)
                @test status.outcome==:complete && status.data_complete
                @test status.last_publication==published
                @test !status.presentation_valid
                @test !isnothing(status.error)
                waitfor(() -> BAT.visualizer_status(vis).view_disposed &&
                    !BAT.visualizer_status(vis).cleanup_pending)
                @test BAT.visualizer_figure(vis)===nothing
            finally
                Makie.off(observer)
            end
            BAT.dispose_visualizer!(vis)
            waitfor(() -> BAT.visualizer_status(vis).view_disposed)
            BAT.dispose_visualizer!(vis)
            @test BAT.visualizer_status(vis).view_disposed
            @test BAT.visualizer_status(vis).data_complete
        end
    finally
        cleanup(vis)
    end
end

struct FailingConvergence <: BAT.ConvergenceTest end
BAT.bat_convergence_impl(::AbstractVector{<:BAT.DensitySampleVector},::FailingConvergence,::BATContext)=throw(FAILURE)
const SAMPLER_ARMED=Threads.Atomic{Bool}(false)
if get(ENV,"BAT_MAKIE_LIFECYCLE_CASE","")=="sampler"
    # This subprocess alone specializes the real proposal helper. Every step
    # before arming follows the original implementation; a later real sampler
    # step throws directly, preserving its usual nested task exception path.
    function BAT.mcmc_propose_transition(current_z::BAT.ArrayOfSimilarArrays,
        proposal::BAT.RandomWalkProposalState,genctxs::AbstractVector)
        SAMPLER_ARMED[] && throw(FAILURE)
        return invoke(BAT.mcmc_propose_transition,
            Tuple{BAT.ArrayOfSimilarArrays,BAT.MCMCProposalState,AbstractVector},
            current_z,proposal,genctxs)
    end
end
function computational_failure(kind)
    vis=BATVisualizer(config())
    callback=kind==:callback ? ((args...)->(BAT.visualizer_status(vis).phase==:running && throw(FAILURE);nothing)) :
        kind==:sampler ? ((args...)->(SAMPLER_ARMED[]=BAT.visualizer_status(vis).phase==:running;nothing)) : BAT.nop_func
    alg=algorithm(;callback,convergence=kind==:convergence ? FailingConvergence() : AssumeConvergence())
    target=MvNormal(zeros(2),ones(2))
    try
        err=capture() do c
            e=caught(() -> BAT.evalmeasure(target,alg,BATContext(rng=BAT.Philox4x((12,34)),visualizer=vis)))
            @test length(c.figures)==1
            e
        end
        @test !isnothing(err)
        @test occursin(FAILURE.msg,sprint(showerror,err))
        kind!=:sampler && @test err===FAILURE
        s=BAT.visualizer_status(vis)
        @test s.outcome==:aborted
        @test !s.data_complete
        @test s.quiesced
        @test !s.cleanup_pending
    finally
        cleanup(vis)
    end
end

const SNAPSHOT_CALLS=Ref(0)
function disabled_updates(vis::V,chain::C,n::Int) where {V,C}
    for _ in 1:n
        BAT.update_visualizer!(vis;chain_state=chain,nonzero_weights=true,is_sampling_step=true)
    end
    return nothing
end
function producer(; snapshot_failure=false)
    # A failed chain lookup exercises the same guarded producer boundary as
    # observation snapshot/append errors, while real sampler steps continue.
    vis,state,outputs,target=fixture(;backend=config(batch=100,limit=200))
    plain_ctx=BATContext(rng=BAT.Philox4x((144,72)))
    plain=BAT.MCMCState(algorithm(),target,Int32(1),[zeros(2)],plain_ctx)
    plain_out=BAT._empty_chain_outputs(plain)
    try
        initialize(vis,state,outputs,target)
        if snapshot_failure
            # This specialization exists only in this case's subprocess. It
            # injects a snapshot-helper failure into a real sampler chain; it
            # does not replace the generic lifecycle or simulate its state.
            chain_type=typeof(state.chain_state)
            @eval function EXT._pending_walker_outputs(chain::$chain_type, nonzero::Bool)
                SNAPSHOT_CALLS[]+=1
                throw(FAILURE)
            end
        else
            empty!(vis.content.chain_ids)
        end
        plain=BAT.mcmc_iterate!!(plain_out,plain;max_nsteps=24)
        observed=Base.invokelatest(BAT.mcmc_iterate!!,outputs,state;max_nsteps=24,update_visualizer=true)
        BAT.finish_visualizer!(vis;success=true)
        @test outputs==plain_out
        @test BAT.get_rng(observed.chain_state.context)==BAT.get_rng(plain.chain_state.context)
        for (a,b) in zip(observed.chain_state.walker_genctxs,plain.chain_state.walker_genctxs)
            @test BAT.get_rng(a)==BAT.get_rng(b)
        end
        assert_failed(vis)
        # Later valid hooks return before the now-invalid lookup and ingestion.
        before=BAT.visualizer_status(vis)
        @test Base.invokelatest(BAT.update_visualizer!,vis;chain_state=observed.chain_state,
            nonzero_weights=true,is_sampling_step=true)===nothing
        @test BAT.visualizer_status(vis).error==before.error
        @test BAT.visualizer_status(vis).last_publication==before.last_publication
        # Enter the world containing the snapshot specialization once, outside
        # each typed hot loop, including the warmup that excludes compilation.
        Base.invokelatest(disabled_updates,vis,observed.chain_state,1)
        small_alloc=@allocated Base.invokelatest(disabled_updates,vis,observed.chain_state,10_000)
        large_alloc=@allocated Base.invokelatest(disabled_updates,vis,observed.chain_state,100_000)
        small_time=@elapsed Base.invokelatest(disabled_updates,vis,observed.chain_state,10_000)
        large_time=@elapsed Base.invokelatest(disabled_updates,vis,observed.chain_state,100_000)
        @info "Warmed disabled-observer hooks" small_alloc large_alloc small_time large_time
        snapshot_failure && @test SNAPSHOT_CALLS[]==1
    finally
        cleanup(vis)
    end
end

function listener()
    for after_detach in (false,true)
        entered=Channel{Nothing}(1)
        release=Channel{Nothing}(1)
        calls=Ref(0)
        tick! = function(vis,state)
            if after_detach
                batch=EXT._detach_live_batch!(vis,[state];force=false)
                isnothing(batch) && return nothing
            end
            calls[]+=1
            put!(entered,nothing)
            take!(release)
            throw(FAILURE)
        end
        vis,state,_,_=listener_fixture(tick!,(args...)->nothing;backend=config(batch=1,limit=1))
        task=vis.content.lifecycle.task
        producer_task=nothing
        try
            producer_task=@async packet!(vis,state)
            waitfor(() -> isready(entered))
            take!(entered)
            @test BAT.visualizer_status(vis).last_publication==0
            put!(release,nothing)
            waitfor(() -> istaskdone(producer_task))
            fetch(producer_task)
            BAT.finish_visualizer!(vis;success=true)
            assert_failed(vis)
            @test istaskdone(task)
            @test calls[]==1 # no retry after an uncertain/failed batch
            @test BAT.visualizer_status(vis).last_publication==0
        finally
            isready(release) || put!(release,nothing)
            cleanup(vis)
        end
    end
end

function timeout_case()
    entered=Channel{Nothing}(1)
    release=Channel{Nothing}(1)
    # Producer deadlines must remain responsive even with a much longer render
    # polling interval. The blocked listener cannot supply those wakeups.
    vis,state,_,_=listener_fixture((args...)->(put!(entered,nothing);take!(release)),
        (args...)->nothing;backend=config(batch=1,limit=1,timeout=0.15,poll=5.0))
    task=vis.content.lifecycle.task
    try
        waitfor(() -> isready(entered))
        take!(entered)
        producer_task=@async packet!(vis,state)
        waitfor(() -> istaskdone(producer_task);seconds=3.0)
        fetch(producer_task)
        s=BAT.visualizer_status(vis)
        @test s.reason==:backpressure_timeout
        @test !s.data_complete
        @test !s.quiesced
        # Closing the polling timer must not disable the owner's finish deadline.
        timer=vis.content.lifecycle.timer
        isnothing(timer) || close(timer)
        start=time_ns()
        BAT.finish_visualizer!(vis;success=true)
        @test (time_ns()-start)/1e9 < 3.0
        @test !BAT.visualizer_status(vis).quiesced
        @test BAT.visualizer_status(vis).cleanup_pending
        @test BAT.visualizer_status(vis).reason==s.reason
        @test BAT.visualizer_status(vis).error==s.error # shutdown cannot erase the first failure
        # Startup rollback and its enclosing sampler finally may both finish.
        # Repeated cleanup must retain the already expired shutdown allowance.
        shutdown_started=vis.content.lifecycle.shutdown_started
        @test !isnothing(shutdown_started)
        BAT.finish_visualizer!(vis;success=false)
        @test vis.content.lifecycle.shutdown_started===shutdown_started
        @test !BAT.visualizer_status(vis).quiesced
        @test BAT.visualizer_status(vis).error==s.error
        put!(release,nothing)
        waitfor(() -> istaskdone(task))
        BAT.finish_visualizer!(vis;success=true)
        @test BAT.visualizer_status(vis).quiesced
        @test !BAT.visualizer_status(vis).data_complete
        @test isnothing(vis.content.lifecycle.shutdown_started)
    finally
        isready(release) || put!(release,nothing)
        cleanup(vis)
    end
end

struct FailingInverse <: Function end
BAT.inverse(::FailingInverse) = throw(FAILURE)
function transform_failure()
    history = Ref{Any}(nothing)
    tick! = (vis,state) -> EXT._flush_live_buffer!(vis,[state],FailingInverse(),
        history,fill(-1.0,2),fill(1.0,2))
    vis,state,outputs,_=listener_fixture(tick!,(args...)->nothing;backend=config(batch=1,limit=1))
    history[]=(samples=vis.content.graph[:samples][],timing=[EXT._empty_walker_spans(outputs)])
    original=history[]
    try
        producer_task=@async packet!(vis,state)
        waitfor(() -> istaskdone(producer_task))
        fetch(producer_task)
        waitfor(() -> BAT.visualizer_status(vis).outcome==:failed)
        BAT.finish_visualizer!(vis;success=true)
        assert_failed(vis)
        @test history[]===original # detached data was never committed or replayed
        @test BAT.visualizer_status(vis).last_publication==0
        @test occursin(FAILURE.msg,BAT.visualizer_status(vis).error.message)
    finally
        cleanup(vis)
    end
end

function publication()
    vis,state,outputs,target=fixture(;backend=config(batch=100,limit=200))
    observer=nothing
    try
        initialize(vis,state,outputs,target)
        g=vis.content.graph
        history=Ref((samples=g[:samples][],timing=[EXT._empty_walker_spans(outputs)]))
        before=BAT.visualizer_status(vis).last_publication
        observer=Makie.on(g[:samples]) do _
            throw(FAILURE)
        end
        packet!(vis,state)
        EXT._flush_live_buffer!(vis,[state],identity,history,fill(-1.0,2),fill(1.0,2);force=true)
        BAT.finish_visualizer!(vis;success=true)
        assert_failed(vis)
        @test length(history[].samples[1][1])==1 # internal commit precedes failed publication
        @test BAT.visualizer_status(vis).last_publication==before
        @test !BAT.visualizer_status(vis).presentation_valid
    finally
        isnothing(observer) || Makie.off(observer)
        cleanup(vis)
    end
end

function stop_publication()
    history=Ref{Any}(nothing)
    ready=Ref(false)
    tick! = function(vis,state)
        ready[]=true # initial selection has passed its own view admission
        EXT._flush_live_buffer!(vis,[state],identity,history,fill(-1.0,2),fill(1.0,2))
    end
    vis,state,outputs,_=listener_fixture(tick!,(args...)->nothing;backend=config(batch=1,limit=1))
    history[]=(samples=vis.content.graph[:samples][],timing=[EXT._empty_walker_spans(outputs)])
    try
        waitfor(() -> ready[])
        producer_task=nothing
        # Hold the real view guard while the real listener commits a detached
        # batch. Publication must then await admission; stop wins that admission.
        EXT._with_live_view(vis) do
            producer_task=@async packet!(vis,state)
            waitfor(() -> !isempty(history[].samples[1][1]))
            @test BAT.visualizer_status(vis).last_publication==0
            BAT.request_visualizer_stop!(vis;reason=:disabled)
        end
        waitfor(() -> istaskdone(producer_task))
        fetch(producer_task)
        BAT.finish_visualizer!(vis;success=true)
        status=BAT.visualizer_status(vis)
        @test status.outcome==:disabled
        @test !status.data_complete
        @test status.quiesced
        @test status.last_publication==0
        @test isempty(vis.content.graph[:samples][][1][1])
    finally
        cleanup(vis)
    end
end

struct FailingPreparation end
Base.getproperty(::FailingPreparation,::Symbol)=throw(FAILURE)
function startup()
    vis=BATVisualizer(config())
    BAT.finish_visualizer!(vis)
    @test BAT.visualizer_status(vis).phase===:fresh
    @test isnothing(vis.content.lifecycle.shutdown_started)
    # Failure after fresh-session acquisition must roll back even though no
    # listener or figure has been acquired yet, and the consumed session stays spent.
    vis=BATVisualizer(config())
    try
        @test caught(() -> BAT.prepare_visualizer_init!(vis,FailingPreparation()))===FAILURE
        assert_failed(vis)
        @test isempty(vis.content.init_segments)
        @test isempty(vis.content.invalid_init)
        @test !BAT.capture_visualizer_init(vis)
        @test_throws ArgumentError BAT.prepare_visualizer_init!(vis,algorithm())
    finally
        cleanup(vis)
    end
    vis,state,outputs,target=fixture()
    try
        err=capture(;fail=true) do c
            caught(() -> BAT.init_visualizer!(vis;rng=BAT.get_rng(state.chain_state.context),
                mcmc_states=[state],outputs=[outputs],target=target,f_pretransform=identity))
        end
        @test err===FAILURE
        @test BAT.visualizer_status(vis).quiesced
        waitfor(() -> BAT.visualizer_status(vis).view_disposed)
        @test BAT.visualizer_status(vis).view_disposed
        @test !BAT.visualizer_status(vis).data_complete
        grid_source=Makie.ComputePipeline.get_observable!(vis.content.graph[:gridlayout];use_deepcopy=false)
        @test isempty(Makie.Observables.listeners(grid_source))
        @test isempty(vis.content.lifecycle.view.subscriptions)
        @test isempty(vis.content.init_segments)
        @test_throws ArgumentError BAT.prepare_visualizer_init!(vis,algorithm())
    finally
        cleanup(vis)
    end
    # Rejected acquisition must not tear down the session that already owns it.
    vis,state,outputs,target=fixture()
    try
        initialize(vis,state,outputs,target)
        before=BAT.visualizer_status(vis)
        @test_throws ArgumentError BAT.prepare_visualizer_init!(vis,algorithm())
        @test BAT.visualizer_status(vis).phase==before.phase==:running
        @test !BAT.visualizer_status(vis).view_disposed
    finally
        cleanup(vis)
    end
end

function view_case()
    vis,state,outputs,target=fixture()
    observer=nothing
    unrelated=Figure()
    ax=Axis(unrelated[1,1]);lines!(ax,[0,1],[0,1])
    try
        fig=initialize(vis,state,outputs,target)
        # Captured/headless creation does not deliver a genuine close transition.
        @test BAT.visualizer_status(vis).phase==:running
        g=vis.content.graph
        depth=Ref(0);peak=Ref(0)
        observer=Makie.on(g[:show_stats_upper]) do _
            depth[]+=1;peak[]=max(peak[],depth[])
            try
                EXT._live_graph_update!(vis;show_stats_lower=true)
                @test !g[:show_stats_lower][] # intent remains deferred in this update
            finally
                depth[]-=1
            end
        end
        @test EXT._live_graph_update!(vis;show_stats_upper=true)
        @test g[:show_stats_lower][]
        @test peak[]==1
        Makie.off(observer);observer=nothing
        events(fig).window_open[]=true
        events(fig).window_open[]=false
        BAT.finish_visualizer!(vis;success=true)
        @test BAT.visualizer_status(vis).outcome!=:complete
        waitfor(() -> BAT.visualizer_status(vis).view_disposed)
        @test BAT.visualizer_status(vis).view_disposed
        @test !isempty(unrelated.content)
        continued=BAT.mcmc_iterate!!(outputs,state;max_nsteps=12,update_visualizer=true)
        @test !isempty(only(outputs)) # real sampling still progresses after the close
        @test continued.chain_state.stepno>=12
    finally
        isnothing(observer) || Makie.off(observer)
        cleanup(vis)
    end
    # Disposal called from an admitted operation cannot join/free its own view.
    vis,state,outputs,target=fixture()
    try
        fig=initialize(vis,state,outputs,target)
        EXT._with_live_view(vis) do
            BAT.dispose_visualizer!(vis)
            @test !BAT.visualizer_status(vis).view_disposed
            @test BAT.visualizer_status(vis).cleanup_pending
            @test !isempty(fig.content)
        end
        BAT.finish_visualizer!(vis;success=false)
        waitfor(() -> BAT.visualizer_status(vis).view_disposed)
        @test BAT.visualizer_status(vis).view_disposed
        @test BAT.visualizer_status(vis).quiesced
    finally
        cleanup(vis)
    end
end

# A real different-thread call, without depending on @spawn's thread selection.
# The optional runner supplies at least two default worker threads for this case.
function figure_other_threads(f, owner_thread)
    @test Threads.nthreads(:default) >= 2
    answers=Channel{Any}(Threads.nthreads(:default))
    # The owner can be in the interactive pool, making every default worker a
    # different thread. Serialize probes so they do not contend with each other
    # on the getter's control guard and legitimately return `nothing` instead.
    probe_lock=ReentrantLock()
    Threads.@threads :static for _ in 1:Threads.nthreads(:default)
        thread=Threads.threadid()
        if thread != owner_thread
            value=try
                lock(f,probe_lock)
            catch err
                err
            end
            put!(answers,(thread=thread,value=value))
        end
    end
    close(answers)
    results=collect(answers)
    @test !isempty(results)
    @test all(r -> r.thread != owner_thread, results)
    return results
end

# Another task holds the guard until assertions finish. A broken blocking getter
# is detected before releasing the holder; the finally still releases both tasks.
function figure_contended_guard(vis, guard)
    entered=Channel{Nothing}(1)
    release=Channel{Nothing}(1)
    holder=@async lock(guard) do
        put!(entered,nothing)
        take!(release)
    end
    probe=nothing
    try
        waitfor(() -> isready(entered))
        probe=@async BAT.visualizer_figure(vis)
        done=timedwait(() -> istaskdone(probe),5.0;pollint=0.01)==:ok
        @test done
        done && @test fetch(probe)===nothing
        @test !istaskdone(holder)
    finally
        isready(release) || put!(release,nothing)
        wait(holder)
        isnothing(probe) || wait(probe)
    end
end

function figure_case()
    @test BAT.visualizer_figure(BATVisualizer())===nothing
    fresh=BATVisualizer(config())
    @test isnothing(fresh.content.lifecycle.view)
    @test BAT.visualizer_figure(fresh)===nothing
    @test fresh.content.lifecycle.phase===:fresh
    @test isnothing(fresh.content.lifecycle.view)
    @test isnothing(fresh.content.lifecycle.task)
    @test isnothing(fresh.content.lifecycle.timer)
    @test all(r -> r.value===nothing,
        figure_other_threads(() -> BAT.visualizer_figure(fresh),Threads.threadid()))
    stripped=deepcopy(fresh)
    @test isnothing(stripped.content)
    @test BAT.visualizer_figure(stripped)===nothing
    @test isnothing(stripped.content)

    # Real active and disabled sessions must not expose the retained view or
    # change observation merely because availability was queried.
    active,state,outputs,target=fixture()
    try
        active_fig=initialize(active,state,outputs,target)
        @test BAT.visualizer_figure(active)===nothing
        @test active.content.lifecycle.phase===:running
        @test active.content.lifecycle.accepting[]
        @test !isempty(active_fig.content)
        BAT.request_visualizer_stop!(active;reason=:disabled)
        @test BAT.visualizer_figure(active)===nothing
        BAT.finish_visualizer!(active;success=false)
        @test BAT.visualizer_status(active).outcome===:disabled
        @test BAT.visualizer_figure(active)===nothing
        BAT.dispose_visualizer!(active)
        @test BAT.visualizer_figure(active)===nothing
    finally
        cleanup(active)
    end

    vis=BATVisualizer(config())
    observer=nothing
    try
        capture() do c
            context=BATContext(rng=BAT.Philox4x((12,34)),visualizer=vis)
            result=BAT.bat_sample(MvNormal(zeros(2),ones(2)),algorithm(),context)
            @test !isempty(result.result)
            # The first access is not preceded by a test status/finish query.
            fig=BAT.visualizer_figure(vis)
            @test fig===only(c.figures)
            @test fig===BAT.visualizer_figure(vis)
            @test length(c.figures)==1
            assert_complete(vis)
            ctl=vis.content.lifecycle
            view=ctl.view
            graph=vis.content.graph
            before=BAT.visualizer_status(vis)
            @test BAT.visualizer_figure(vis)===fig
            @test BAT.visualizer_status(vis)==before
            @test BAT.visualizer_figure(deepcopy(vis))===nothing
            @test context.visualizer===vis
            @test result.optargs.context.visualizer !== vis
            @test isnothing(result.optargs.context.visualizer.content)
            @test BAT.visualizer_figure(result.optargs.context.visualizer)===nothing
            mktempdir() do dir
                path=joinpath(dir,"completed-live.png")
                Makie.save(path,fig)
                @test isfile(path) && filesize(path)>0
            end

            # Supplementary boundary states: each mutation is restored before
            # continuing. They test availability predicates, not lifecycle flow.
            for (phase,outcome) in ((:starting,:inactive),(:running,:running),
                    (:draining,:running),(:quiesced,:failed),(:quiesced,:aborted),(:quiesced,:disabled))
                saved=(ctl.phase,ctl.outcome,ctl.data_complete)
                try
                    ctl.phase=phase
                    ctl.outcome=outcome
                    ctl.data_complete=false
                    @test BAT.visualizer_figure(vis)===nothing
                finally
                    ctl.phase,ctl.outcome,ctl.data_complete=saved
                end
            end
            for (field,value) in ((:data_complete,false),(:initializing,true))
                saved=getproperty(ctl,field)
                try
                    setproperty!(ctl,field,value)
                    @test BAT.visualizer_figure(vis)===nothing
                finally
                    setproperty!(ctl,field,saved)
                end
            end
            for field in (:task,:disposal_task)
                saved=getproperty(ctl,field)
                unfinished=@task nothing
                try
                    setproperty!(ctl,field,unfinished)
                    @test BAT.visualizer_figure(vis)===nothing
                    @test getproperty(ctl,field)===unfinished
                finally
                    setproperty!(ctl,field,saved)
                end
            end
            timer=Timer(3600.0)
            try
                ctl.timer=timer
                @test BAT.visualizer_figure(vis)===nothing
                @test isopen(timer)
            finally
                close(timer)
                ctl.timer=nothing
            end
            for flag in (view.disposed,view.cleanup_pending,view.cleanup_running)
                saved=flag[]
                try
                    flag[]=true
                    @test BAT.visualizer_figure(vis)===nothing
                finally
                    flag[]=saved
                end
            end
            try
                view.presentation_valid[]=false
                @test BAT.visualizer_figure(vis)===nothing
            finally
                view.presentation_valid[]=true
            end
            try
                view.figure=nothing
                @test BAT.visualizer_figure(vis)===nothing
                @test isnothing(view.figure)
            finally
                view.figure=fig
            end
            try
                ctl.view=nothing
                @test BAT.visualizer_figure(vis)===nothing
                @test isnothing(ctl.view)
            finally
                ctl.view=view
            end
            saved_error,saved_phase=ctl.error,ctl.error_phase
            try
                ctl.error=CapturedException(FAILURE,Any[])
                ctl.error_phase=:historical_test_diagnostic
                @test BAT.visualizer_figure(vis)===fig
                @test BAT.visualizer_status(vis).error.message==FAILURE.msg
            finally
                ctl.error,ctl.error_phase=saved_error,saved_phase
            end

            # Explicit completed-task boundary, with no intervening status query.
            # Rejected access must not retire the handle before owner/idle checks.
            done=@async nothing
            wait(done)
            ctl.task=done
            ctl.phase=:draining
            ctl.owner_task=current_task()
            original_error=ctl.error
            wrong_thread=figure_other_threads(() -> BAT.visualizer_figure(vis),ctl.owner_thread)
            @test all(r -> r.value isa ArgumentError, wrong_thread)
            @test ctl.task===done && ctl.phase===:draining
            @test ctl.outcome===:complete && ctl.data_complete
            @test ctl.error===original_error
            @test view.presentation_valid[] && !view.cleanup_pending[]
            lock(vis.content.buffer_lock) do
                contended=figure_other_threads(() -> BAT.visualizer_figure(vis),ctl.owner_thread)
                @test all(r -> r.value===nothing, contended)
            end
            for guard in (vis.content.buffer_lock,view.lock)
                figure_contended_guard(vis,guard)
                @test ctl.task===done && ctl.phase===:draining
            end
            for (field,value) in ((:depth,1),(:updating,true))
                saved=getproperty(view,field)
                try
                    setproperty!(view,field,value)
                    @test BAT.visualizer_figure(vis)===nothing
                    @test ctl.task===done && ctl.phase===:draining
                finally
                    setproperty!(view,field,saved)
                end
            end
            try
                view.pending[:show_stats_upper]=true
                @test BAT.visualizer_figure(vis)===nothing
                @test ctl.task===done && ctl.phase===:draining
            finally
                empty!(view.pending)
            end
            @test EXT._with_live_view(vis) do
                @test BAT.visualizer_figure(vis)===nothing
                @test ctl.task===done && ctl.phase===:draining
            end

            # A real retained BAT toggle still updates its graph, while getter
            # reentry from the synchronous graph callback is unavailable.
            reentered=Any[]
            observer=Makie.on(graph[:show_stats_upper]) do _
                push!(reentered,BAT.visualizer_figure(vis))
            end
            old_stats=graph[:show_stats_upper][]
            toggle=first(filter(x -> x isa Makie.Toggle,fig.content))
            toggle.active[]=!toggle.active[]
            @test graph[:show_stats_upper][] != old_stats
            @test reentered==[nothing]
            @test ctl.task===done && ctl.phase===:draining
            Makie.off(observer);observer=nothing
            @test BAT.visualizer_figure(vis)===fig
            @test isnothing(ctl.task)
            @test ctl.phase===:quiesced
            @test isnothing(ctl.owner_task)
            @test length(c.figures)==1

            # A real disposal request immediately revokes availability, before
            # the scheduled cleanup can run. Do not yield or query status here.
            BAT.dispose_visualizer!(vis)
            @test view.cleanup_pending[]
            @test !isempty(fig.content)
            @test BAT.visualizer_figure(vis)===nothing
            @test ctl.outcome===:complete && ctl.data_complete
            waitfor(() -> BAT.visualizer_status(vis).view_disposed &&
                !BAT.visualizer_status(vis).cleanup_pending)
            # Cleanup empties the borrowed object, rather than a separate copy.
            @test isempty(fig.content)
            @test BAT.visualizer_figure(vis)===nothing
            BAT.dispose_visualizer!(vis)
            @test BAT.visualizer_figure(vis)===nothing
        end
    finally
        isnothing(observer) || Makie.off(observer)
        cleanup(vis)
    end
end

function grid_case()
    # Feed controlled chain outputs through the real listener. It remains the
    # only owner of history and final publication throughout this session.
    vis,state,outputs,target=fixture(;backend=config(batch=1,limit=32))
    fig=nothing
    try
        fig=initialize(vis,state,outputs,target)
        graph=vis.content.graph
        cell(i,j)=only(filter(x -> x isa Axis,contents(fig[1,1][i,j])))
        axes=[cell(i,j) for i in 1:2,j in 1:2]
        source=Makie.ComputePipeline.get_observable!(graph[:gridlayout];use_deepcopy=false)
        view=vis.content.lifecycle.view
        handles=filter(h -> h.observable===source,view.subscriptions)
        @test length(handles)==1
        @test any(pair -> last(pair)===only(handles).f,Makie.Observables.listeners(source))
        prior_lo,prior_hi=copy(graph[:domain_lo][]),copy(graph[:domain_hi][])
        upper=axes[1,2]
        manual=nothing
        finite_limits=nothing
        midpoint=(prior_lo.+prior_hi)./2
        points=[midpoint,midpoint.+0.01,prior_hi.+10.0]
        for (step,point) in enumerate(points)
            chain=state.chain_state
            chain.stepno=step+1
            chain.output.v[1].=point
            chain.output.weight[1]=1
            chain.output.logd[1]=-1
            chain.output.info[1]=BAT.MCMCSampleID(Int32(1),Int32(1),Int32(1),Int64(step),Int32(1),true)
            previous=BAT.visualizer_status(vis).last_publication
            BAT.update_visualizer!(vis;chain_state=chain,nonzero_weights=true,is_sampling_step=true)
            waitfor(() -> BAT.visualizer_status(vis).last_publication>previous; seconds=120.0)
            @test length(graph[:samples][][1][1])==step
            @test length(filter(x -> x isa Axis,fig.content))==4
            @test all(cell(i,j)===axes[i,j] for i in 1:2,j in 1:2)
            if step==1
                # Initial live cells are inactive until listener selection and
                # first publication. Pan only once finite bounds are installed.
                finite_limits=deepcopy(upper.limits[])
                @test !any(isnothing,finite_limits)
                initial=upper.targetlimits[]
                manual=typeof(initial)(initial.origin .+ 0.2 .* initial.widths,0.5 .* initial.widths)
                upper.targetlimits[]=manual
            elseif step==2
                @test upper.targetlimits[]==manual
                @test upper.limits[]==finite_limits
                @test graph[:domain_lo][]==prior_lo && graph[:domain_hi][]==prior_hi
            else
                @test upper.targetlimits[] != manual
                @test graph[:domain_hi][]==point
                # Existing graph policy pads each nondegenerate domain by 5%.
                margin=0.05 .* (point.-prior_lo)
                @test upper.limits[]==((prior_lo[2]-margin[2],point[2]+margin[2]),
                    (prior_lo[1]-margin[1],point[1]+margin[1]))
            end
        end
        # Same-axis recipe and selection changes use the actual retained controls.
        lower_menu,diag_menu,upper_menu=filter(x -> x isa Menu,fig.content)
        for recipe in (KDE1D,Hist1D)
            diag_menu.i_selected[]=findfirst(option -> option[2]===recipe,diag_menu.options[])
            @test all(cell(i,j)===axes[i,j] for i in 1:2,j in 1:2)
            @test length(filter(x -> x isa Axis,fig.content))==4
        end
        checks=filter(x -> x isa Checkbox,fig.content)
        # Two-variable triangular picker: (1,1),(2,1),(2,2).
        checks[3].checked[]=false
        @test !axes[2,2].xlabelvisible[]
        checks[3].checked[]=true
        @test axes[2,2].xlabelvisible[]
        @test all(cell(i,j)===axes[i,j] for i in 1:2,j in 1:2)
        BAT.finish_visualizer!(vis;success=true)
        @test BAT.visualizer_figure(vis)===fig
        @test length(graph[:samples][][1][1])==length(points)
        @test graph[:domain_hi][]==last(points)
        @test all(axis -> axis in fig.content,axes)
        @test !isempty(Makie.colorbuffer(fig))
        BAT.dispose_visualizer!(vis)
        waitfor(() -> BAT.visualizer_status(vis).view_disposed &&
            !BAT.visualizer_status(vis).cleanup_pending)
        @test isempty(fig.content)
        @test isempty(view.subscriptions)
        @test all(h -> all(pair -> last(pair)!==h.f,Makie.Observables.listeners(source)),handles)
        @test isempty(Makie.Observables.listeners(source))
        # Deliberately update the retained graph after disposal. Detached binding
        # must not repopulate the deleted axes, even outside live admission.
        Makie.update!(graph;diagonal_recipe=Errorbars1D)
        @test all(axis -> isempty(axis.scene.plots),axes)
        @test isempty(fig.content)
        @test BAT.visualizer_figure(vis)===nothing
    finally
        cleanup(vis)
    end
end

const CASES=Dict("healthy"=>healthy,"callback"=>(() -> computational_failure(:callback)),
    "convergence"=>(() -> computational_failure(:convergence)),"sampler"=>(() -> computational_failure(:sampler)),
    "producer"=>producer,"snapshot"=>(() -> producer(snapshot_failure=true)),
    "listener"=>listener,"timeout"=>timeout_case,
    "transform"=>transform_failure,"publication"=>publication,"stop_publication"=>stop_publication,
    "startup"=>startup,"view"=>view_case,"figure"=>figure_case,"grid"=>grid_case)
case=get(ENV,"BAT_MAKIE_LIFECYCLE_CASE","")
haskey(CASES,case) || error("Run this file through test/optional/makie/runtests.jl")
@testset "Makie lifecycle: $case" begin
    CASES[case]()
end
end # module
