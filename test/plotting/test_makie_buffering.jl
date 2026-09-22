# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Optional Makie test entry point, run by the process-bounded application suite.
module TestMakieBuffering

using BAT, Test
using Distributions, LinearAlgebra, Makie
const EXT = Base.get_extension(BAT, :BATMakieExt)

mutable struct FigureCapture <: AbstractDisplay
    count::Int
end
Base.displayable(::FigureCapture, ::MIME"image/png") = true
Base.display(d::FigureCapture, ::Makie.Figure) = (d.count += 1; nothing)

function initialize(vis, states, outputs, target, rng)
    capture = FigureCapture(0)
    old_inline = Makie.ALWAYS_INLINE_PLOTS[]
    Makie.inline!(true)
    pushdisplay(capture)
    try
        BAT.init_visualizer!(vis; rng = rng, mcmc_states = states, outputs = outputs,
            f_pretransform = identity, target = target)
        @test capture.count == 1
    finally
        popdisplay(capture)
        Makie.inline!(old_inline)
    end
end

function buffer_is_partial(vis)
    lock(vis.content.buffer_lock) do
        vis.content.n_buffer_steps[] < vis.content.effective_batch_size[]
    end
end

function config(limit, adaptive, rate)
    cfg = BATMakieVisualization(max_buffered = limit, adaptive_batching = adaptive, batch_growth_rate = rate)
    BATMakieVisualization(cfg.recipes, [1, 2], 2, cfg.n_batch, cfg.max_buffered,
        cfg.adaptive_batching, cfg.batch_growth_rate, cfg.poll_interval, cfg.dark,
        cfg.triagonal_config, cfg.diagonal_config)
end

@testset "Makie buffer thresholds" begin
    @testset "Threshold arithmetic" begin
        for batch in (1, 49, 50, 60, 200, 2^53-1, 2^53+1, typemax(Int)-1024, typemax(Int)),
            factor in (1.0, nextfloat(1.0), 1.2, 4.0, 1e100, floatmax(Float64))
            threshold = EXT._scaled_buffer_threshold(batch, factor)
            @test batch <= threshold <= typemax(Int)
            @test EXT._scaled_buffer_threshold(threshold, factor) >= threshold
        end
        @test EXT._scaled_buffer_threshold(50, 1.2) == 60
        @test EXT._scaled_buffer_threshold(50, 4.0) == 200
        @test EXT._scaled_buffer_threshold(2^53+1, 1.0) == 2^53+1
        @test EXT._scaled_buffer_threshold(typemax(Int), 1.2) == typemax(Int)
        for ratio in (1.0, 4.0)
            batch = 50
            for _ in 1:100
                batch = EXT._scaled_buffer_threshold(batch, 1.2)
                @test batch <= EXT._scaled_buffer_threshold(batch, ratio)
            end
        end
    end

    # Ordinary cases force several flushes by producing more than the combined
    # first three stop thresholds. Extreme cases use only a few real rows.
    cases = [(limit, adaptive, 1.2, 250) for limit in (50, 200) for adaptive in (false, true)]
    append!(cases, [(50, true, floatmax(Float64), 17), (typemax(Int), false, 1.2, 17)])
    for (limit, adaptive, rate, nsteps) in cases
        @testset "limit=$limit adaptive=$adaptive rate=$rate" begin
            vis = BATVisualizer(config(limit, adaptive, rate))
            target = BAT.BATDistMeasure(MvNormal(zeros(2), Matrix{Float64}(I, 2, 2)))
            alg = TransformedMCMC(nchains = 2, nwalkers = 2,
                adaptive_transform = BAT.NoAdaptiveTransform(), transform_tuning = NoMCMCTransformTuning())
            states = [BAT.MCMCState(alg, target, Int32(id), [zeros(2), zeros(2)],
                BATContext(rng = BAT.Philox4x((id, 71)), visualizer = vis)) for id in 1:2]
            outputs = BAT._empty_chain_outputs.(states)
            expected = [[Vector{Float64}[] for _ in 1:2] for _ in 1:2]
            for (id, state) in enumerate(states), walker in 1:2
                state.chain_state.current.x.weight[walker] = 1
                push!(outputs[id][walker], deepcopy(state.chain_state.current.x[walker]))
                push!(expected[id][walker], zeros(2))
                state.chain_state.current.x.weight[walker] = 0 # initial sample was already emitted
            end
            updates = Int[]
            observer = nothing
            try
                initialize(vis, states, outputs, target, BAT.get_rng(states[1].chain_state.context))
                observer = Makie.on(vis.content.graph[:current_idx]) do value
                    push!(updates, Int(value))
                end
                producer = @async begin
                    @sync for (id, state) in enumerate(states)
                        Threads.@spawn begin
                            chain = state.chain_state
                            for step in 1:nsteps
                                for walker in 1:2
                                    v = [10.0 * id + walker + step / 10, sin(step + walker)]
                                    chain.output.v[walker] .= v
                                    chain.output.weight[walker] = 1
                                    chain.output.logd[walker] = -1
                                    chain.output.info[walker] = BAT.MCMCSampleID(Int32(id), Int32(walker),
                                        Int32(1), Int64(step), Int32(1), true)
                                    push!(expected[id][walker], v)
                                end
                                chain.stepno = step
                                BAT.update_visualizer_impl!(vis; chain_state = chain, nonzero_weights = true, is_sampling_step = true)
                            end
                        end
                    end
                end
                outcome = timedwait(() -> istaskdone(producer), 30.0)
                @test outcome == :ok
                outcome == :ok || error("Producer did not finish within the test deadline")
                fetch(producer)
                @test timedwait(() -> sum(vis.content.graph[:flat_weights][]) >= 54, 30.0) == :ok
                @test !isempty(updates)
                @test vis.content.is_live[]
                @test !istaskdone(vis.content.listener_task[])
                # Let any already-reachable ordinary flush finish before testing
                # the forced drain. Producers have joined, so no new rows arrive.
                @test timedwait(() -> buffer_is_partial(vis), 30.0) == :ok
                before_final = vis.content.effective_batch_size[]
                if rate == floatmax(Float64)
                    @test before_final == typemax(Int)
                    @test sum(vis.content.graph[:flat_weights][]) >= 54 # initial four + first flush
                elseif adaptive
                    @test before_final >= 87 # at least three ordinary growth cycles
                else
                    @test before_final == 50
                end
                vis.content.is_live[] = false
                @test timedwait(() -> istaskdone(vis.content.listener_task[]), 30.0) == :ok
                @test vis.content.n_buffer_steps[] == 0
                @test vis.content.effective_batch_size[] == before_final # forced drain does not grow
                @test sum(vis.content.graph[:flat_weights][]) == 4 * (nsteps + 1)
                plotted = vis.content.graph[:samples][]
                for id in 1:2, walker in 1:2
                    @test plotted[id][walker].v == expected[id][walker]
                    @test all(==(1), plotted[id][walker].weight)
                end
            finally
                isnothing(observer) || Makie.off(observer)
                # Emergency cleanup only after assertions; R11 cancellation is a
                # separate issue. The outer process deadline contains stuck tasks.
                vis.content.is_live[] = false
                lock(vis.content.buffer_lock) do
                    notify(vis.content.buffer_cond, all = true)
                end
                task = vis.content.listener_task[]
                isnothing(task) || timedwait(() -> istaskdone(task), 5.0)
            end
        end
    end
end
end # module
