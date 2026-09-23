# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Optional Makie regression entry point; run in the bounded application suite.
module TestMakieDwell
using BAT, Test
using Distributions, LinearAlgebra, Makie, Random
const EXT = Base.get_extension(BAT, :BATMakieExt)

mutable struct Capture <: AbstractDisplay
    count::Int
end
Base.displayable(::Capture, ::MIME"image/png") = true
Base.display(d::Capture, ::Makie.Figure) = (d.count += 1; nothing)
function initialize(vis, states, outputs, target)
    cap = Capture(0)
    old_inline = Makie.ALWAYS_INLINE_PLOTS[]
    Makie.inline!(true)
    pushdisplay(cap)
    try
        BAT.init_visualizer!(vis; rng = BAT.get_rng(states[1].chain_state.context),
            mcmc_states = states, outputs = outputs, target = target, f_pretransform = identity)
        @test cap.count == 1
    finally
        popdisplay(cap)
        Makie.inline!(old_inline)
    end
end
function config(; batch = 4, limit = 8)
    c = BATMakieVisualization(adaptive_batching = false)
    BATMakieVisualization(c.recipes, [1, 2], 2, batch, limit, false,
        c.batch_growth_rate, 0.01, c.dark, c.triagonal_config, c.diagonal_config)
end
function fixture(; nchains = 2, nwalkers = 2, weighting = RepetitionWeighting(), batch = 4, limit = 8)
    vis = BATVisualizer(config(; batch, limit))
    target = BAT.BATDistMeasure(MvNormal(zeros(2), 1e-24 .* Matrix{Float64}(I, 2, 2)))
    alg = TransformedMCMC(; nchains, nwalkers, sample_weighting = weighting,
        adaptive_transform = BAT.NoAdaptiveTransform(), transform_tuning = NoMCMCTransformTuning())
    states = [BAT.MCMCState(alg, target, Int32(i), [zeros(2) for _ in 1:nwalkers],
        BATContext(rng = BAT.Philox4x((i, 71)), visualizer = vis)) for i in 1:nchains]
    return vis, target, states, BAT._empty_chain_outputs.(states)
end
function ingest(vis, state, step; nonzero = true)
    BAT.update_visualizer_impl!(vis; chain_state = state.chain_state,
        nonzero_weights = nonzero, is_sampling_step = step)
end
function stop(vis)
    BAT.finish_visualizer!(vis; success = true)
    status = BAT.visualizer_status(vis)
    @test status.quiesced
    @test status.outcome == :complete
    @test status.data_complete
    @test isnothing(status.error)
end
function cleanup(vis)
    BAT.request_visualizer_stop!(vis; reason = :disabled)
    BAT.finish_visualizer!(vis; success = false)
    BAT.dispose_visualizer!(vis)
end

mass(vis) = sum(vis.content.graph[:flat_weights][])
# These regression watchdogs include cold compilation. Short lifecycle deadline
# behavior is checked separately in test_makie_lifecycle.jl.
function waitmass(vis, expected)
    @test timedwait(() -> mass(vis) == expected, 120.0) == :ok
    @test mass(vis) == expected
end
function reject_steps!(vis, states, outputs, chain, n; nonzero = true)
    # The fixed seed and tiny target produce checked, real zero-acceptance steps.
    producer = @async begin
        for _ in 1:n
            states[chain] = BAT.mcmc_step!!(states[chain])
            state = states[chain]
            @test !any(state.chain_state.accepted)
            BAT.get_samples!(outputs[chain], state, nonzero)
            before = deepcopy(state.chain_state.current.x)
            rng = copy(BAT.get_rng(state.chain_state.context))
            ingest(vis, state, true; nonzero)
            @test state.chain_state.current.x == before
            @test BAT.get_rng(state.chain_state.context) == rng
        end
    end
    @test timedwait(() -> istaskdone(producer), 120.0) == :ok
    fetch(producer)
end
function terminal!(vis, states, outputs, chain; nonzero = true)
    states[chain] = BAT.flush_samples!!(states[chain])
    BAT.get_samples!(outputs[chain], states[chain], nonzero)
    ingest(vis, states[chain], false; nonzero)
end

@testset "Live dwell progress" begin
    @testset "Snapshot ownership, zero policy and fractional weights" begin
        for weighting in (RepetitionWeighting(), ARPWeighting())
            vis, target, states, outputs = fixture(; nchains = 1, weighting, limit = 100)
            state = states[1]
            chain = state.chain_state
            chain.current.x.weight .= weighting isa RepetitionWeighting ? [7, 0] : [0.75, 0.0]
            snapshot = EXT._pending_walker_outputs(chain, true)
            all_snapshot = EXT._pending_walker_outputs(chain, false)
            @test length.(snapshot) == [1, 0]
            @test length.(all_snapshot) == [1, 1]
            old_v = copy(snapshot[1].v[1])
            old_weight = snapshot[1].weight[1]
            chain.current.x.v[1] .= 123
            chain.current.x.weight[1] = 99
            @test snapshot[1].v[1] == old_v
            @test snapshot[1].weight[1] == old_weight
            # Compose fractional pending state twice from the same emitted base:
            # no accumulation, no conversion of weights to time, no input mutation.
            base = BAT._empty_chain_outputs(state)
            base[1] = deepcopy(snapshot[1])
            combined = BAT._append_walker_outputs(state, base, snapshot)
            repeated = BAT._append_walker_outputs(state, base, snapshot)
            @test combined[1].weight == repeated[1].weight == [2 * old_weight]
            @test base[1].weight == [old_weight]
        end
        @test EXT._add_buffer_steps(typemax(Int) - 1, 2) == typemax(Int)
        @test EXT._add_buffer_steps(10, 2) == 12
    end

    @testset "Fractional and zero weights do not define progress" begin
        vis, target, states, outputs = fixture(; nchains = 1, weighting = ARPWeighting(), limit = 100)
        state = states[1]
        push!(vis.content.chain_ids, Int32(1))
        push!(vis.content.output_buffer, BAT._empty_chain_outputs(state))
        push!(vis.content.pending_samples, BAT._empty_chain_outputs(state))
        # This transport-only fixture has no listener/figure, but ingestion
        # still requires the same claimed and active session as real startup.
        EXT._claim_live_visualizer!(vis)
        EXT._activate_live_visualizer!(vis)
        try
            state.chain_state.output.weight .= [0.25, 0.0]
            state.chain_state.current.x.weight .= [0.75, 0.0]
            ingest(vis, state, true)
            @test vis.content.n_buffer_steps[] == 2
            @test length.(vis.content.pending_samples[1]) == [1, 0]
            ingest(vis, state, false; nonzero = false)
            @test vis.content.n_buffer_steps[] == 2
            @test all(isempty, vis.content.pending_samples[1])
            @test vis.content.buffer_dirty[]
        finally
            cleanup(vis)
        end
    end

    @testset "Real iterator preserves rejected samples and every RNG stream" begin
        for nonzero in (true, false)
            vis, target, states, outputs = fixture(; batch = 4, limit = 8)
            plain_vis, _, plain_states, plain_outputs = fixture()
            try
                initialize(vis, states, outputs, target)
                plain_states = BAT.mcmc_iterate!!(plain_outputs, plain_states;
                    max_nsteps = 12, nonzero_weights = nonzero)
                producer = @async BAT.mcmc_iterate!!(outputs, states;
                    max_nsteps = 12, nonzero_weights = nonzero, update_visualizer = true)
                @test timedwait(() -> istaskdone(producer), 120.0) == :ok
                states = fetch(producer)
                stop(vis)
                @test mass(vis) == 48
                @test vis.content.step_mode[]
                @test vis.content.graph[:max_step][] == 12
                Makie.update!(vis.content.graph; window_steps=(11,12),show_trace_upper=true)
                @test mass(vis) == 8 # Two steps, two chains, two walkers.
                @test length(vis.content.graph[:trace_groups][]) == 4
                Makie.update!(vis.content.graph; window_steps=nothing)
                @test mass(vis) == 48
                @test vis.content.n_buffer_steps[] == 0
                @test !vis.content.buffer_dirty[]
                for chain in 1:2
                    live, plain = states[chain].chain_state, plain_states[chain].chain_state
                    @test live.current.x == plain.current.x
                    @test live.output == plain.output
                    @test BAT.get_rng(live.context) == BAT.get_rng(plain.context)
                    @test all(BAT.get_rng(a) == BAT.get_rng(b) for
                        (a, b) in zip(live.walker_genctxs, plain.walker_genctxs))
                    for walker in 1:2
                        @test outputs[chain][walker] == plain_outputs[chain][walker]
                        @test vis.content.graph[:samples][][chain][walker] == outputs[chain][walker]
                    end
                end
            finally
                cleanup(vis)
            end
        end
    end

    @testset "Emitted weight-only merges still advance steps" begin
        vis, target, states, outputs = fixture(; nchains = 1, nwalkers = 1, limit = 100)
        state = states[1]
        push!(vis.content.chain_ids, Int32(1))
        push!(vis.content.output_buffer, BAT._empty_chain_outputs(state))
        push!(vis.content.pending_samples, BAT._empty_chain_outputs(state))
        EXT._claim_live_visualizer!(vis)
        EXT._activate_live_visualizer!(vis)
        try
            chain = state.chain_state
            chain.output .= chain.current.x
            chain.output.weight .= 1
            ingest(vis, state, true)
            ingest(vis, state, true)
            @test length(only(vis.content.output_buffer[1])) == 1
            @test sum(only(vis.content.output_buffer[1]).weight) == 2
            @test vis.content.n_buffer_steps[] == 2
            @test vis.content.buffer_dirty[]
        finally
            cleanup(vis)
        end
    end

    @testset "Real rejections, idle chain, replacement and final drain" begin
        for nonzero in (true, false)
            vis, target, states, outputs = fixture()
            try
                initialize(vis, states, outputs, target)
                reject_steps!(vis, states, outputs, 1, 4; nonzero)
                waitmass(vis, 8)
                Makie.update!(vis.content.graph; window_steps=(3,4))
                @test mass(vis) == 4 # Clips the still-provisional rejection tail.
                Makie.update!(vis.content.graph; window_steps=nothing)
                @test mass(vis) == 8
                @test all(states[1].chain_state.current.x.weight .== 4)
                @test all(states[1].chain_state.output.weight .== 0)
                nonzero && @test all(isempty, outputs[1])
                # Chain 1 becomes idle; chain 2's flush must retain chain 1's tail.
                reject_steps!(vis, states, outputs, 2, 4; nonzero)
                waitmass(vis, 16)
                reject_steps!(vis, states, outputs, 1, 2; nonzero)
                waitmass(vis, 20)
                @test vis.content.n_buffer_steps[] == 0
                @test BAT.visualizer_status(vis).phase == :running
                terminal!(vis, states, outputs, 1; nonzero)
                terminal!(vis, states, outputs, 2; nonzero)
                @test vis.content.n_buffer_steps[] == 0
                @test vis.content.buffer_dirty[]
                stop(vis)
                @test !vis.content.buffer_dirty[]
                @test all(chain -> all(isempty, chain), vis.content.pending_samples)
                @test mass(vis) == 20
                plotted = vis.content.graph[:samples][]
                for chain in 1:2, walker in 1:2
                    @test plotted[chain][walker].v == outputs[chain][walker].v
                    @test plotted[chain][walker].weight == outputs[chain][walker].weight
                end
            finally
                cleanup(vis)
            end
        end
    end

    @testset "Repeated equal pending snapshots never accumulate" begin
        for nonzero in (true, false)
            vis, target, states, outputs = fixture(; nchains = 1)
            try
                initialize(vis, states, outputs, target)
                reject_steps!(vis, states, outputs, 1, 4; nonzero)
                waitmass(vis, 8)
                previous_view = deepcopy(vis.content.graph[:samples][])
                ingest(vis, states[1], true; nonzero)
                ingest(vis, states[1], true; nonzero)
                @test timedwait(() -> vis.content.n_buffer_steps[] == 0, 120.0) == :ok
                # Drain completion precedes publication. Joining the listener
                # ensures composition finished, even if Makie elides equal data.
                stop(vis)
                @test mass(vis) == 8
                @test vis.content.graph[:samples][] == previous_view
            finally
                cleanup(vis)
            end
        end
    end

    @testset "Acceptance replaces the dwell; cycle and zero-step flush" begin
        for same_coordinate in (false, true)
            vis, target, states, outputs = fixture(; nchains = 1, nwalkers = 1, batch = 2, limit = 4)
            try
                initialize(vis, states, outputs, target)
                reject_steps!(vis, states, outputs, 1, 4)
                waitmass(vis, 4)
                chain = states[1].chain_state
                # Controlled acceptance packet has the exact layout produced by
                # mcmc_step!!: old dwell in output, new state of unit weight in current.
                chain.output .= chain.current.x
                chain.current.x.v[1] .= same_coordinate ? 0.0 : 1.0
                chain.current.x.weight .= 1
                chain.stepno += 1
                chain.current.x.info[1] = BAT.MCMCSampleID(Int32(1), Int32(1), Int32(1), chain.stepno, Int32(1), true)
                BAT.get_samples!(outputs[1], states[1], true)
                ingest(vis, states[1], true)
                # The next rejection updates only the current weight.
                chain.output.weight .= 0
                chain.current.x.weight .+= 1
                chain.stepno += 1
                ingest(vis, states[1], true)
                waitmass(vis, 6)
                terminal!(vis, states, outputs, 1)
                BAT.next_cycle!(states[1])
                # A new-cycle packet; stepno resets, but batching counts the event.
                chain = states[1].chain_state
                chain.output.weight .= 0
                chain.current.x.weight .= 1
                chain.stepno = 1
                ingest(vis, states[1], true)
                chain.current.x.weight .+= 1
                chain.stepno = 2
                ingest(vis, states[1], true)
                waitmass(vis, 8)
                @test vis.content.n_buffer_steps[] == 0
                terminal!(vis, states, outputs, 1)
                @test vis.content.buffer_dirty[]
                @test vis.content.n_buffer_steps[] == 0
                stop(vis)
                @test mass(vis) == 8
                @test only(vis.content.graph[:samples][])[1].v == outputs[1][1].v
                @test only(vis.content.graph[:samples][])[1].weight == outputs[1][1].weight
                @test vis.content.graph[:max_step][] == 6
                Makie.update!(vis.content.graph; window_steps=(2,2),show_trace_upper=true)
                @test mass(vis) == 2 # Local step two from each cycle, even if rows merge.
                @test length(vis.content.graph[:trace_groups][]) == 1
                @test only(vis.content.graph[:trace_groups][]).recency == [1.0]
                Makie.update!(vis.content.graph; window_steps=nothing)
                @test mass(vis) == 8
            finally
                cleanup(vis)
            end
        end
    end
end
end # module
