# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestVisualizerInitSegments
using BAT, Test, Distributions, Random

struct RecordingBackend <: BAT.BATVisBackend end
BAT.capture_visualizer_init(::BATVisualizer{RecordingBackend}) = true
function BAT.begin_visualizer_init_segment(vis::BATVisualizer{RecordingBackend}, chain, output)
    (step=chain.stepno, cycle=chain.info.cycle, weights=copy(chain.current.x.weight))
end
function BAT.end_visualizer_init_segment!(vis::BATVisualizer{RecordingBackend}, before, chain, output)
    lock(vis.content.lock) do
        push!(vis.content.records, (before=before, step=chain.stepno, cycle=chain.info.cycle,
            weights=copy(chain.current.x.weight)))
    end
    nothing
end

@testset "Initialization visualization boundaries" begin
    for retry in (false,true)
        init = retry ? MCMCRetryInit(nsteps_init=30) : MCMCChainPoolInit(nsteps_init=30)
        alg = TransformedMCMC(nchains=2,nwalkers=retry ? 2 : 1, init=init,
            adaptive_transform=BAT.NoAdaptiveTransform(),transform_tuning=NoMCMCTransformTuning())
        target = BAT.BATDistMeasure(MvNormal(zeros(2), ones(2)))
        vis = BATVisualizer(RecordingBackend(),(records=NamedTuple[],lock=ReentrantLock()))
        observed_ctx = BATContext(rng=BAT.Philox4x((54,731)),visualizer=vis)
        plain_ctx = BATContext(rng=BAT.Philox4x((54,731)))
        unexpected(args...) = error("Initialization emitted a public callback")
        Random.seed!(937)
        plain = BAT.mcmc_init!(alg,target,init,unexpected,plain_ctx)
        Random.seed!(937)
        observed = BAT.mcmc_init!(alg,target,init,unexpected,observed_ctx)
        @test observed.outputs == plain.outputs
        @test BAT.get_rng(observed_ctx) == BAT.get_rng(plain_ctx)
        @test !isempty(vis.content.records)
        @test all(r -> r.before.step <= r.step && r.before.cycle==r.cycle &&
            all(iszero,r.before.weights) && all(iszero,r.weights),vis.content.records)
        for (a,b) in zip(observed.mcmc_states,plain.mcmc_states)
            @test a.chain_state.current.x == b.chain_state.current.x
            @test BAT.get_rng(a.chain_state.context) == BAT.get_rng(b.chain_state.context)
        end
        # Normal iterator calls do not record just because a backend exists.
        empty!(vis.content.records)
        state = BAT.mcmc_iterate!!(observed.outputs[1],observed.mcmc_states[1];max_nsteps=1)
        @test isempty(vis.content.records)
        state = BAT.mcmc_iterate!!(observed.outputs[1],state;max_nsteps=0,record_visualizer_init=true)
        @test length(vis.content.records)==1
        @test only(vis.content.records).before.step==only(vis.content.records).step
    end
    @test !BAT.capture_visualizer_init(BATVisualizer())
end
end
