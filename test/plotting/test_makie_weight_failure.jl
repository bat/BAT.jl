# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Separate process-bounded optional case: a real numerical range failure in an
# active observer must leave fixed-work sampling unchanged and release history.
module TestMakieWeightFailure

using BAT, Test, Makie, Distributions, Random, Logging

const EXT = Base.get_extension(BAT, :BATMakieExt)

mutable struct Capture <: AbstractDisplay
    figures::Vector{Figure}
end
Base.displayable(::Capture, ::MIME"image/png") = true
Base.display(capture::Capture, figure::Figure) = (push!(capture.figures, figure); nothing)

function with_capture(f)
    capture = Capture(Figure[])
    old_inline = Makie.ALWAYS_INLINE_PLOTS[]
    Makie.inline!(true)
    pushdisplay(capture)
    try
        return with_logger(NullLogger()) do
            f(capture)
        end
    finally
        popdisplay(capture)
        Makie.inline!(old_inline)
    end
end

function wait_done(task; seconds=30.0)
    @test task isa Task
    task isa Task || error("Missing owned task in numerical-failure fixture")
    done = timedwait(() -> istaskdone(task), seconds; pollint=0.005) === :ok
    @test done
    done || error("Numerical-failure fixture did not terminate its owned task")
    @test !istaskfailed(task)
    wait(task)
end

function chain_snapshot(chain)
    (current=deepcopy(chain.current), proposed=deepcopy(chain.proposed),
        output=deepcopy(chain.output), accepted=copy(chain.accepted),
        order=copy(chain.walker_order), info=chain.info, stepno=chain.stepno,
        nattempts=copy(chain.nattempts), nsamples=copy(chain.nsamples),
        rng=copy(BAT.get_rng(chain.context)),
        walker_rngs=[copy(BAT.get_rng(context)) for context in chain.walker_genctxs])
end

function configuration()
    c = BATMakieVisualization(adaptive_batching=false)
    recipes = (upper=Hist2D, diagonal=Hist1D, lower=Hist2D)
    BATMakieVisualization(recipes, [1, 2], 2, 20_000, 40_000, false,
        c.batch_growth_rate, 0.01, c.dark,
        merge(c.triagonal_config, (nbins=(12, 12),)),
        merge(c.diagonal_config, (nbins=12,)); observer_timeout=60.0)
end

function algorithm(callback)
    TransformedMCMC(nchains=2, nsteps=24, strict=false, store_burnin=true,
        init=MCMCChainPoolInit(nsteps_init=20), callback=callback,
        convergence=AssumeConvergence(), adaptive_transform=BAT.NoAdaptiveTransform(),
        transform_tuning=NoMCMCTransformTuning(),
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=12, max_ncycles=1, nsteps_final=3))
end

@testset "Numerical observer failure preserves sampling and releases history" begin
    vis = BATVisualizer(configuration())
    plain_context = BATContext(rng=BAT.Philox4x((144, 72)))
    live_context = BATContext(rng=BAT.Philox4x((144, 72)), visualizer=vis)
    target = MvNormal(zeros(2), fill(1000.0, 2))
    plain_callbacks, live_callbacks = Any[], Any[]
    injected = Ref(false)
    listener, releaser = Ref{Union{Nothing,Task}}(nothing), Ref{Union{Nothing,Task}}(nothing)
    callback_steps = Ref{Vector{Int64}}(Int64[])

    plain_callback = function(tag, states)
        push!(plain_callbacks, (tag=tag, states=chain_snapshot.(getproperty.(states, :chain_state))))
        return nothing
    end
    live_callback = function(tag, states)
        push!(live_callbacks, (tag=tag, states=chain_snapshot.(getproperty.(states, :chain_state))))
        if !injected[]
            # Initialization and a real burn-in cycle have completed normally.
            # Change only observation configuration; never alter a sample or
            # override a production method to manufacture an exception.
            graph = EXT._live_graph(vis)
            @test !isempty(graph[:flat_weights][])
            @test isempty(graph[EXT.primitive_symbol(Std1D(), (1, 1))][].positions)
            config = merge(vis.backend.diagonal_config, (nsigma=floatmax(Float64),))
            @test EXT._live_graph_update!(vis; diagonal_config=config)
            @test vis.content.lifecycle.accepting[]
            @test EXT._live_graph_update!(vis;
                diagonal_config=vis.backend.diagonal_config, show_stats_diag=true)
            spread = graph[EXT.primitive_symbol(Std1D(), (1, 1))][].positions
            @test (last(spread) - first(spread)) / 2 > 1
            listener[] = lock(() -> vis.content.lifecycle.task, vis.content.buffer_lock)
            callback_steps[] = [state.chain_state.stepno for state in states]
            injected[] = true
            @test_throws ArgumentError EXT._stats_extent(
                (last(spread) - first(spread)) / 2, config.nsigma)
            @test !EXT._live_graph_update!(vis; diagonal_config=config)
            releaser[] = lock(() -> vis.content.lifecycle.disposal_task, vis.content.buffer_lock)
            @test !vis.content.lifecycle.accepting[]
        end
        return nothing
    end

    try
        Random.seed!(8921)
        plain = with_logger(NullLogger()) do
            BAT.evalmeasure(target, algorithm(plain_callback), plain_context)
        end
        default_after_plain = copy(Random.default_rng())
        Random.seed!(8921)
        observed = with_capture() do capture
            result = BAT.evalmeasure(target, algorithm(live_callback), live_context)
            @test length(capture.figures) == 1
            result
        end

        @test injected[]
        @test length(plain_callbacks) == length(live_callbacks) == 1
        @test live_callbacks == plain_callbacks
        @test BAT.samplesof(observed) == BAT.samplesof(plain)
        live_chains = BAT.samplegenof(observed).chain_states
        plain_chains = BAT.samplegenof(plain).chain_states
        @test chain_snapshot.(live_chains) == chain_snapshot.(plain_chains)
        @test all(chain -> chain.stepno == 24, live_chains)
        @test callback_steps[] == fill(12, 2)
        @test BAT.get_rng(live_context) == BAT.get_rng(plain_context)
        @test rand(UInt64, 16) == rand(default_after_plain, UInt64, 16)

        # No status polling or rescue disposal drives the release assertion.
        wait_done(listener[])
        wait_done(releaser[])
        @test isnothing(vis.content.graph[])
        status = BAT.visualizer_status(vis)
        @test status.outcome === :failed
        @test !status.data_complete
        @test status.quiesced && status.history_released
        @test !status.cleanup_pending
        # Makie wraps the arithmetic error at its graph boundary. H14 retains
        # only the bounded wrapper summary, never the graph-capturing exception.
        @test status.error.type == "ResolveException"
        @test status.error.phase === :view
        @test status.error.message == "Exception details are not retained"
        @test isnothing(status.cleanup_error)
        @test isnothing(BAT.visualizer_figure(vis))
    finally
        # Failure cleanup follows the outcome assertions and touches only the
        # visualizer owned by this fixture. No numerical task is detached.
        BAT.request_visualizer_stop!(vis; reason=:disabled)
        BAT.finish_visualizer!(vis; success=false)
        BAT.dispose_visualizer!(vis)
        coordinator = lock(() -> vis.content.lifecycle.disposal_task, vis.content.buffer_lock)
        isnothing(coordinator) || wait_done(coordinator)
    end
end

end # module
