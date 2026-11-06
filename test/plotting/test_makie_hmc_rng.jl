# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# The headless runner uses Cairo and a display capture. The separate native
# gl_hmc_rng.jl entry uses actual GL screens and runs cold/cached campaigns.
module TestMakieHMCRNG

using BAT, Test, Makie, Distributions, Random, Logging
include("makie_rng_testutils.jl")

const NATIVE_GL = get(ENV, "BAT_MAKIE_HMC_NATIVE", "false") == "true"

mutable struct FigureCapture <: AbstractDisplay
    count::Int
end
Base.displayable(::FigureCapture, ::MIME"image/png") = true
Base.display(d::FigureCapture, ::Makie.Figure) = (d.count += 1; nothing)

function with_capture(f)
    NATIVE_GL && return f(nothing)
    capture = FigureCapture(0)
    previous = Makie.ALWAYS_INLINE_PLOTS[]
    Makie.inline!(true)
    pushdisplay(capture)
    try
        return f(capture)
    finally
        popdisplay(capture)
        Makie.inline!(previous)
    end
end

function configuration()
    c = BATMakieVisualization(adaptive_batching=false)
    BATMakieVisualization((upper=Hist2D, diagonal=Hist1D, lower=Hist2D),
        [1], 1, 20_000, 40_000, false, c.batch_growth_rate, 0.01, c.dark,
        merge(c.triagonal_config, (nbins=(8, 8),)),
        merge(c.diagonal_config, (nbins=8,)); observer_timeout=30.0)
end

function algorithm(callback)
    # These HMC, step-size and transform components also exercise the actual
    # singleton/restart path in test_stanlike_lifecycle.jl. The present test
    # compares complete public campaigns, rather than another covariance oracle.
    tuning = MultiTrafoTuning((
        BAT.StanLikeTuning(init_buffer=0, term_buffer=0, window_size=1),
        BAT.StanLikeTuning(init_buffer=0, term_buffer=1, window_size=2),
    ))
    adaptive = AdaptiveTransformChain((
        BAT.TriangularAffineTransform(init=BAT.UnitTransformInit()),
        BAT.TriangularAffineTransform(init=BAT.UnitTransformInit()),
    ))
    TransformedMCMC(nchains=2, nwalkers=1, nsteps=16, strict=false,
        proposal=HamiltonianMC(step_size=0.01, max_depth=3),
        proposal_tuning=BAT.StepSizeAdaptor(gamma=1.0),
        adaptive_transform=adaptive, transform_tuning=tuning,
        pretransform=DoNotTransform(), store_burnin=true,
        init=MCMCRetryInit(nsteps_init=12, max_init_tries=2, strict=false),
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=12, max_ncycles=1, nsteps_final=2),
        convergence=AssumeConvergence(), callback=callback)
end

rng_snapshot(rng) = (state=copy(rng), next=rand(copy(rng), UInt64, 16))

function sample_snapshot(samples)
    (v=[collect(v) for v in samples.v], weight=collect(samples.weight),
        logd=collect(samples.logd), aux=collect(samples.aux),
        info=[(chainid=id.chainid, walkerid=id.walkerid, chaincycle=id.chaincycle,
            stepno=id.stepno, proposalid=id.proposalid, sampletype=id.sampletype)
            for id in samples.info])
end

function proposal_snapshot(proposal::BAT.HMCProposalState)
    d = proposal.diagnostics
    # The gradient callable owns target/context/AD machinery. Compare the
    # proposal's numerical controls and realized trajectory diagnostics, not
    # closure identity or a recursive traversal of that ownership graph.
    (step_size=proposal.step_size, step_jitter=proposal.step_jitter,
        max_depth=proposal.max_depth, max_delta_energy=proposal.max_delta_energy,
        target_acceptance=proposal.target_acceptance,
        target_acceptance_int=proposal.target_acceptance_int,
        diagnostics=(n_transitions=d.n_transitions, n_divergent=d.n_divergent,
            n_maxdepth=d.n_maxdepth, n_leapfrog=d.n_leapfrog,
            sum_p_accept=d.sum_p_accept, warmup_n_transitions=d.warmup_n_transitions,
            warmup_n_divergent=d.warmup_n_divergent, warmup_n_maxdepth=d.warmup_n_maxdepth,
            warmup_n_leapfrog=d.warmup_n_leapfrog, warmup_sum_p_accept=d.warmup_sum_p_accept))
end

function chain_snapshot(chain)
    info, partition = chain.info, chain.rngpart_cycle
    (info=(id=info.id, cycle=info.cycle, tuned=info.tuned, converged=info.converged),
        stepno=chain.stepno, nattempts=copy(chain.nattempts), nsamples=copy(chain.nsamples),
        accepted=copy(chain.accepted), walker_order=copy(chain.walker_order),
        current=map(sample_snapshot, chain.current), proposed=map(sample_snapshot, chain.proposed),
        output=sample_snapshot(chain.output),
        transforms=map(f -> (A=Matrix(f.A), b=collect(f.b)), BAT.fchainfs(chain.f_transform)),
        proposal=proposal_snapshot(chain.proposal),
        cycle_rng=(seed=partition.seed, counters=partition.partctrsbase,
            depth=partition.depth, indices=partition.idxs),
        context_rng=rng_snapshot(BAT.get_rng(chain.context)),
        walker_rngs=[rng_snapshot(BAT.get_rng(ctx)) for ctx in chain.walker_genctxs])
end

function window_snapshot(tuner::BAT.StanLikeTunerState)
    p, l = tuner.stats.param_stats, tuner.stats.logtf_stats
    (i=tuner.i, first=tuner.window_start, last=tuner.window_end,
        splits=copy(tuner.window_splits),
        mean=(sum_w=p.mean.sum_w, S=copy(p.mean.S), C=copy(p.mean.C)),
        cov=(n=p.cov.n, sum_w=p.cov.sum_w, sum_w2=p.cov.sum_w2,
            center=copy(p.cov.Mean_X), next_center=copy(p.cov.New_Mean_X), S=copy(p.cov.S)),
        minimum=copy(p.minimum), maximum=copy(p.maximum), mode=copy(tuner.stats.mode),
        logdensity=(sum_v=l.mean.sum_v, sum_w=l.mean.sum_w,
            n=l.var.n, var_sum_w=l.var.sum_w, sum_w2=l.var.sum_w2,
            center=l.var.mean_x, scatter=l.var.s, minimum=l.minimum, maximum=l.maximum))
end

function stepsize_snapshot(tuner::BAT.HMCStepSizeTunerState)
    (tuning=(gamma=tuner.tuning.gamma, t0=tuner.tuning.t0, kappa=tuner.tuning.kappa),
        m=tuner.m, log_mu=tuner.log_mu, log_stepsize_bar=tuner.log_stepsize_bar,
        H_bar=tuner.H_bar, run_nobs=tuner.run_nobs, run_accept_sum=tuner.run_accept_sum,
        run_accept_sqsum=tuner.run_accept_sqsum, run_ndivergent=tuner.run_ndivergent,
        run_skip=tuner.run_skip, min_run_nobs=tuner.min_run_nobs)
end

function owner_streams()
    full = copy(Random.default_rng())
    ordinary = rand(copy(full), UInt64, 16)
    children = Vector{Vector{UInt64}}(undef, 3)
    @sync for i in eachindex(children)
        Threads.@spawn children[i] = rand(UInt64, 16)
    end
    # These probes deliberately fork after the public-call checkpoint, equally
    # in each campaign. Preserve both full states; ordinary draws miss s4.
    return (; full, ordinary, children, after_children=copy(Random.default_rng()))
end

function join_disposal(vis)
    task = lock(() -> vis.content.lifecycle.disposal_task, vis.content.buffer_lock)
    if task isa Task
        done = timedwait(() -> istaskdone(task), 30.0; pollint=0.005) === :ok
        @test done
        done || error("HMC observation disposal timed out")
        @test !istaskfailed(task)
        wait(task)
    end
    return nothing
end

function campaign(; live=false)
    vis = live ? BATVisualizer(configuration()) : BATVisualizer()
    # BAT already imports its ForwardDiff dependency; this optional test does
    # not add another AD package to the optional project's direct dependencies.
    context = BATContext(rng=BAT.Philox4x((2816, 2841)), ad=BAT.ForwardDiff, visualizer=vis)
    owner = current_task()
    events = NamedTuple[]
    function callback(::Val{:mcmc_burnin}, states)
        # Legitimate caller-owned randomness must survive observation. Perform
        # precisely the same draws and fork in the two plain runs and live run.
        before_draw = rand(UInt64)
        child = @task rand(UInt64, 8)
        schedule(child)
        child_draws = fetch(child)
        after_draw = rand(UInt64)
        push!(events, (tag=:mcmc_burnin, owner=current_task() === owner,
            before_draw=before_draw, child_draws=child_draws, after_draw=after_draw,
            chains=chain_snapshot.(getproperty.(states, :chain_state)),
            transforms=[(last_change_restarts=s.trafo_tuner_state.last_change_restarts,
                windows=window_snapshot.(s.trafo_tuner_state.trafo_tuners)) for s in states],
            proposals=[stepsize_snapshot(s.proposal_tuner_state) for s in states]))
        return nothing
    end
    alg = algorithm(callback)
    target = MvNormal([0.0], ones(1, 1))
    released = false
    try
        result, streams = with_capture() do capture
            with_logger(NullLogger()) do
                # All fixture objects exist before the common seed/checkpoint.
                Random.seed!(0x483330)
                result = BAT.evalmeasure(target, alg, context)
                streams = owner_streams()
                isnothing(capture) || @test capture.count == (live ? 1 : 0)
                (result, streams)
            end
        end
        samples = sample_snapshot(BAT.samplesof(result))
        chains = chain_snapshot.(BAT.samplegenof(result).chain_states)
        @test !isempty(samples.weight)
        @test all(isfinite, samples.logd)
        @test length(chains) == 2
        @test all(c -> c.stepno == 16 && length(c.walker_rngs) == 1, chains)
        @test all(c -> isfinite(c.proposal.step_size) && c.proposal.step_size > 0, chains)
        @test length(events) == 1
        event = only(events)
        @test event.owner && event.tag === :mcmc_burnin
        @test all(c -> c.stepno == 12, event.chains)
        @test all(t -> length(t.windows) == 2, event.transforms)
        # The callback sees genuine completed windows, not frozen/no-op tuners.
        @test all(t -> all(w -> !isempty(w.splits) && w.i == last(w.splits) &&
            w.cov.n == 0, t.windows), event.transforms)
        if live
            status = BAT.visualizer_status(vis)
            @test status.outcome === :complete && status.data_complete && status.quiesced
            @test !status.history_released && !status.cleanup_pending
            fig = BAT.visualizer_figure(vis)
            @test fig isa Makie.Figure
            screen = NATIVE_GL ? Makie.getscreen(fig.scene) : nothing
            render_task = isnothing(screen) ? nothing : screen.rendertask
            if NATIVE_GL
                @test nameof(parentmodule(typeof(screen))) === :GLMakie
                @test isopen(screen)
                @test Makie.current_backend().renderloop_running(screen)
                @test !isempty(Makie.colorbuffer(screen))
            end
            @test vis.content.n_buffer_steps[] == 0
            @test isempty(vis.content.init_segments) && isempty(vis.content.invalid_init)

            before_disposal = copy(Random.default_rng())
            BAT.dispose_visualizer!(vis)
            @test Random.default_rng() == before_disposal
            coordinator = lock(() -> vis.content.lifecycle.disposal_task, vis.content.buffer_lock)
            @test coordinator isa Task
            coordinator isa Task || error("Completed HMC view lost its disposal coordinator")
            join_disposal(vis)
            @test istaskdone(coordinator) && !istaskfailed(coordinator)
            @test Random.default_rng() == before_disposal
            status = BAT.visualizer_status(vis)
            @test status.history_released && status.quiesced && !status.cleanup_pending
            @test status.outcome === :complete && status.data_complete
            @test BAT.visualizer_figure(vis) === nothing
            @test isnothing(vis.content.graph[])
            @test isnothing(vis.content.lifecycle.task) && isnothing(vis.content.lifecycle.timer)
            @test isnothing(vis.content.lifecycle.disposal_task)
            if NATIVE_GL
                @test !isopen(screen)
                @test render_task isa Task && istaskdone(render_task)
                @test !istaskfailed(render_task)
            end

            BAT.dispose_visualizer!(vis)
            @test Random.default_rng() == before_disposal
            @test BAT.visualizer_status(vis).history_released
            released = true
        end
        return (; samples, chains, events, context_rng=rng_snapshot(BAT.get_rng(context)), streams)
    finally
        # Healthy completion/release is asserted before fallback cleanup.
        if live && !released
            BAT.request_visualizer_stop!(vis; reason=:disabled)
            BAT.finish_visualizer!(vis; success=false)
            BAT.dispose_visualizer!(vis)
            join_disposal(vis)
        end
    end
end

@testset "Public HMC/MultiTrafo observation preserves numerical and RNG state" begin
    @test _prepare_makie_rng_test!() === nothing
    plain = campaign()
    plain_repeat = campaign()
    observed = campaign(; live=true)
    comparisons = [("plain/plain harness control", plain_repeat),
                   ("observed/plain public sampling", observed)]
    NATIVE_GL && push!(comparisons, ("cached GL public sampling", campaign(; live=true)))
    for (name, result) in comparisons
        @testset "$name" begin
            for field in (:samples, :chains, :events, :context_rng)
                @test isequal(getproperty(result, field), getproperty(plain, field))
            end
            @test result.streams.full == plain.streams.full
            @test result.streams.ordinary == plain.streams.ordinary
            @test result.streams.children == plain.streams.children
            @test result.streams.after_children == plain.streams.after_children
        end
    end
end

end # module
