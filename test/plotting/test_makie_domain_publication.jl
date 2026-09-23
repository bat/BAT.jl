# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieDomainPublication
using BAT, Makie, Test, Distributions
const EXT = Base.get_extension(BAT, :BATMakieExt)

function fixture()
    c = BATMakieVisualization(adaptive_batching=false)
    backend = BATMakieVisualization(c.recipes, [1,2], 2, 10_000, 20_000,
        false, c.batch_growth_rate, c.poll_interval, c.dark,
        c.triagonal_config, c.diagonal_config)
    vis = BATVisualizer(backend)
    target = BAT.BATDistMeasure(MvNormal(zeros(2), ones(2)))
    algorithm = TransformedMCMC(nchains=2,
        adaptive_transform=BAT.NoAdaptiveTransform(),
        transform_tuning=NoMCMCTransformTuning())
    context = BATContext(rng=BAT.Philox4x((144,72)), visualizer=vis)
    state = BAT.MCMCState(algorithm, target, Int32(1), [zeros(2)], context)
    outputs = BAT._empty_chain_outputs(state)
    EXT._claim_live_visualizer!(vis)
    vis.content.n_dof[] = 2
    EXT.register_state_for_vis!(vis, state, outputs)
    @test EXT._activate_live_visualizer!(vis)
    history = Ref((samples=vis.content.graph[:samples][],
        timing=[EXT._empty_walker_spans(outputs)]))
    return vis, state, history
end

snapshot(samples) = (v=copy(samples.v.data), logd=copy(samples.logd),
    weight=copy(samples.weight), info=copy(samples.info), aux=deepcopy(samples.aux))
snapshot_chains(samples) = [[snapshot(walker) for walker in chain] for chain in samples]

function set_sample!(samples, point, step)
    samples.weight[1] = isnothing(point) ? 0 : 1
    if !isnothing(point)
        samples.v[1] .= point
    end
    samples.logd[1] = -1
    samples.info[1] = BAT.MCMCSampleID(Int32(1), Int32(1), Int32(1),
        Int64(step), Int32(1), true)
    return nothing
end

@testset "Live domain publication scans the current candidate without mutating owners" begin
    vis, state, history = fixture()
    graph = vis.content.graph
    prior_lo, prior_hi = [-10.0,-20.0], [10.0,20.0]
    prior_before = (copy(prior_lo), copy(prior_hi))
    initial_history = history[]
    initial_before = snapshot_chains(initial_history.samples)
    sources = Tuple{Any,Any}[]
    generations = NamedTuple[]
    frames = [
        (fresh=nothing, pending=nothing, lo=[-10.0,-20.0], hi=[10.0,20.0],
            committed=zeros(2,0), published=zeros(2,0)),
        (fresh=[2.0,-3.0], pending=[50.0,-40.0], lo=[-10.0,-40.0], hi=[50.0,20.0],
            committed=reshape([2.0,-3.0],2,1), published=[2.0 50.0; -3.0 -40.0]),
        # Deliberately replace a transport snapshot without committing it. This
        # tests candidate ownership, not a particular MCMC transition history.
        (fresh=nothing, pending=[1.0,2.0], lo=[-10.0,-20.0], hi=[10.0,20.0],
            committed=reshape([2.0,-3.0],2,1), published=[2.0 1.0; -3.0 2.0]),
        (fresh=[15.0,-5.0], pending=nothing, lo=[-10.0,-20.0], hi=[15.0,20.0],
            committed=[2.0 15.0; -3.0 -5.0], published=[2.0 15.0; -3.0 -5.0]),
    ]
    try
        @test !vis.content.step_mode[]
        @test isnothing(vis.content.lifecycle.task)
        @test isnothing(vis.content.lifecycle.timer)
        for (step, frame) in enumerate(frames)
            chain = state.chain_state
            chain.stepno = step
            set_sample!(chain.output, frame.fresh, step)
            set_sample!(chain.current.x, frame.pending, step)
            producer_before = (snapshot(chain.output), snapshot(chain.current.x))
            BAT.update_visualizer!(vis; chain_state=chain,
                nonzero_weights=true, is_sampling_step=true)
            @test vis.content.buffer_dirty[]
            @test vis.content.n_buffer_steps[] == 1

            # Retain the real producer-owned transport arrays before detach.
            output_source = vis.content.output_buffer[1][1]
            pending_source = vis.content.pending_samples[1][1]
            push!(sources, (output_source, snapshot(output_source)))
            push!(sources, (pending_source, snapshot(pending_source)))
            before = BAT.visualizer_status(vis).last_publication
            committed_before = history[]
            EXT._flush_live_buffer!(vis, [state], identity, history, prior_lo, prior_hi; force=true)

            @test BAT.visualizer_status(vis).last_publication == before + 1
            @test !vis.content.buffer_dirty[]
            @test vis.content.n_buffer_steps[] == 0
            @test vis.content.output_buffer[1][1] !== output_source
            @test isempty(vis.content.output_buffer[1][1])
            @test (snapshot(chain.output), snapshot(chain.current.x)) == producer_before
            @test history[].samples[1][1].v.data == frame.committed
            @test graph[:samples][][1][1].v.data == frame.published
            @test graph[:current_idxs][] == [[size(frame.published,2)]]
            @test graph[:sample_timing][] === nothing
            @test isnothing(frame.fresh) ? history[] === committed_before : history[] !== committed_before
            @test graph[:domain_lo][] == frame.lo
            @test graph[:domain_hi][] == frame.hi
            @test graph[:domain_lo][] !== prior_lo
            @test graph[:domain_hi][] !== prior_hi
            # ComputePipeline may retain an earlier input when the new vector
            # has equal values. Ownership requires old values to remain intact,
            # not a new graph-stored identity on every publication.
            @test (prior_lo, prior_hi) == prior_before
            @test snapshot_chains(initial_history.samples) == initial_before
            @test all(pair -> snapshot(pair[1]) == pair[2], sources)
            @test all(g -> snapshot_chains(g.samples) == g.before &&
                g.lo == g.expected_lo && g.hi == g.expected_hi, generations)

            push!(generations, (samples=graph[:samples][], before=snapshot_chains(graph[:samples][]),
                lo=graph[:domain_lo][], hi=graph[:domain_hi][],
                expected_lo=copy(frame.lo), expected_hi=copy(frame.hi)))
            latest = last(generations)
            EXT._flush_live_buffer!(vis, [state], identity, history, prior_lo, prior_hi; force=true)
            @test BAT.visualizer_status(vis).last_publication == before + 1
            @test graph[:samples][] === latest.samples
            @test graph[:domain_lo][] === latest.lo
            @test graph[:domain_hi][] === latest.hi
        end
        @test BAT.visualizer_status(vis).outcome == :running
        @test isnothing(vis.content.lifecycle.task)
        @test isnothing(vis.content.lifecycle.timer)
    finally
        BAT.request_visualizer_stop!(vis; reason=:disabled)
        BAT.finish_visualizer!(vis; success=false)
        BAT.dispose_visualizer!(vis)
    end
    @test BAT.visualizer_status(vis).quiesced
    @test !BAT.visualizer_status(vis).cleanup_pending
end
end
