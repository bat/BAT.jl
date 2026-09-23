# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestVisualizerLifecycle
using BAT, Test, Distributions, Logging

mutable struct LifecycleRecord
    events::Vector{Symbol}
    claimed::Bool
    fail_prepare::Bool
    fail_init::Bool
    fail_clear::Bool
    fail_finish::Bool
end
LifecycleRecord(; fail_prepare=false, fail_init=false, fail_clear=false, fail_finish=false) =
    LifecycleRecord(Symbol[], false, fail_prepare, fail_init, fail_clear, fail_finish)
struct RecordingBackend <: BAT.BATVisBackend end
function BAT.prepare_visualizer_init!(vis::BATVisualizer{RecordingBackend}, alg)
    c = vis.content
    c.claimed && error("session already acquired")
    c.fail_prepare && error("prepare failure")
    c.claimed = true
    push!(c.events, :prepare)
    nothing
end
function BAT.init_visualizer!(vis::BATVisualizer{RecordingBackend}; kwargs...)
    push!(vis.content.events, :init)
    vis.content.fail_init && error("init failure")
    nothing
end
function BAT.clear_visualizer_init!(vis::BATVisualizer{RecordingBackend})
    push!(vis.content.events, :clear)
    vis.content.fail_clear && error("clear failure")
    nothing
end
function BAT.request_visualizer_stop!(vis::BATVisualizer{RecordingBackend}; reason=:disabled, exception=nothing)
    push!(vis.content.events, reason)
    nothing
end
function BAT.finish_visualizer!(vis::BATVisualizer{RecordingBackend}; success::Bool=true)
    push!(vis.content.events, success ? :complete : :incomplete)
    vis.content.fail_finish && error("finish failure")
    nothing
end

struct ThrowingLogger <: AbstractLogger end
Logging.min_enabled_level(::ThrowingLogger) = Logging.Warn
# Fail observation diagnostics only; unrelated tuning warnings retain their
# computational semantics and are not part of the cleanup containment boundary.
Logging.shouldlog(::ThrowingLogger, level, mod, group, id) = group === :bat_visualizer
Logging.catch_exceptions(::ThrowingLogger) = false
Logging.handle_message(::ThrowingLogger, args...; kwargs...) = error("logger failure")

struct FailingInit <: BAT.MCMCInitAlgorithm
    exception::Exception
end
BAT.mcmc_init!(alg::TransformedMCMC, target::BAT.BATMeasure, init::FailingInit, callback::Function, ctx::BATContext) = throw(init.exception)

struct InterruptingBackend <: BAT.BATVisBackend end
BAT.finish_visualizer!(::BATVisualizer{InterruptingBackend}; success::Bool=true) = throw(InterruptException())
BAT.clear_visualizer_init!(::BATVisualizer{InterruptingBackend}) = throw(InterruptException())

function sample(vis; callback=BAT.nop_func, init=MCMCChainPoolInit(nsteps_init=30))
    alg = TransformedMCMC(nchains=2, nsteps=4, strict=false, store_burnin=true,
        adaptive_transform=BAT.NoAdaptiveTransform(), transform_tuning=NoMCMCTransformTuning(),
        init=init, convergence=AssumeConvergence(),
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=10, max_ncycles=1, nsteps_final=0),
        callback=callback)
    bat_sample(Normal(), alg, BATContext(rng=BAT.Philox4x((124, 95)), visualizer=vis)).result
end

mutable struct BlockingRecord
    cond::Threads.Condition
    stopped::Bool
    stop_calls::Int
    throw_stop::Bool
    entered::Channel{Nothing}
end
BlockingRecord(; throw_stop=false) = BlockingRecord(Threads.Condition(), false, 0, throw_stop, Channel{Nothing}(1))
struct BlockingBackend <: BAT.BATVisBackend end
function BAT.request_visualizer_stop!(vis::BATVisualizer{BlockingBackend}; kwargs...)
    c = vis.content
    lock(c.cond) do
        c.stopped = true
        c.stop_calls += 1
        notify(c.cond; all=true)
    end
    c.throw_stop && error("stop failure")
    nothing
end
function BAT.update_visualizer_impl!(vis::BATVisualizer{BlockingBackend}; kwargs...)
    c = vis.content
    lock(c.cond) do
        put!(c.entered, nothing)
        while !c.stopped
            wait(c.cond)
        end
    end
    nothing
end

# Minimal numerical iterator: the real scalar/vector loops, storage boundary,
# observation dispatch and child catch execute unchanged. Only the numerical
# step is replaced to place a deterministic error behind sibling wait barriers.
mutable struct TestIterator <: BAT.MCMCIterator
    context::BATContext
    info::NamedTuple{(:id,),Tuple{Int32}}
    stepno::Int
    step::Function
end
BAT.nsteps(c::TestIterator) = c.stepno
BAT.nsamples(::TestIterator) = 0
function BAT.mcmc_step!!(s::BAT.MCMCState{TestIterator})
    s.chain_state.step()
    s.chain_state.stepno += 1
    s
end
BAT.flush_samples!!(s::BAT.MCMCState{TestIterator}) = s
BAT.get_samples!(out, s::BAT.MCMCState{TestIterator}, nonzero::Bool) = out
function state(vis, id, step=()->nothing)
    BAT.MCMCState(TestIterator(BATContext(visualizer=vis), (id=Int32(id),), 0, step),
        BAT.NoMCMCProposalTunerState(), BAT.NoMCMCTransformTuningState(), BAT.MCMCNoTemperingState())
end

function contains_exception(err, expected)
    err === expected && return true
    err isa CompositeException && return any(e -> contains_exception(e, expected), err.exceptions)
    err isa TaskFailedException && return any(e -> contains_exception(e.exception, expected), Base.current_exceptions(err.task))
    false
end

@testset "Generic visualization lifecycle" begin
    vis = BATVisualizer()
    @test BAT.request_visualizer_stop!(vis) === nothing
    @test BAT.finish_visualizer!(vis) === nothing
    @test BAT.dispose_visualizer!(vis) === nothing
    @test BAT.visualizer_status(vis) == (
        phase=:inactive, outcome=:inactive, reason=:none, data_complete=false,
        last_publication=0, quiesced=true, cleanup_pending=false,
        view_disposed=false, presentation_valid=false, error=nothing, cleanup_error=nothing)

    @testset "Owned exits and rejected acquisition" begin
        observed = BATVisualizer(RecordingBackend(), LifecycleRecord())
        expected = sample(BATVisualizer())
        @test sample(observed) == expected
        @test first(observed.content.events) == :prepare
        @test count(==(:complete), observed.content.events) == 1
        @test :clear in observed.content.events
        before = copy(observed.content.events)
        @test_throws ErrorException sample(observed)
        @test observed.content.events == before

        finish_fault = BATVisualizer(RecordingBackend(), LifecycleRecord(fail_finish=true))
        @test with_logger(() -> sample(finish_fault), ThrowingLogger()) == expected
        @test :complete in finish_fault.content.events
        @test :cleanup_failed in finish_fault.content.events

        rejected = BATVisualizer(RecordingBackend(), LifecycleRecord(fail_prepare=true))
        @test_throws ErrorException sample(rejected)
        @test isempty(rejected.content.events)

        startup = BATVisualizer(RecordingBackend(), LifecycleRecord(fail_init=true))
        @test_throws "init failure" sample(startup)
        @test :clear in startup.content.events
        @test :aborted in startup.content.events
        @test last(startup.content.events) == :incomplete

        init_failure = ErrorException("MCMC initialization failure")
        initializing = BATVisualizer(RecordingBackend(), LifecycleRecord(fail_clear=true, fail_finish=true))
        caught_init = with_logger(ThrowingLogger()) do
            try
                sample(initializing; init=FailingInit(init_failure))
                nothing
            catch err
                err
            end
        end
        @test caught_init === init_failure
        @test :init ∉ initializing.content.events
        @test :clear in initializing.content.events
        @test :aborted in initializing.content.events
        @test :incomplete in initializing.content.events
        @test last(initializing.content.events) == :cleanup_failed

        failure = ErrorException("public callback failure")
        broken = BATVisualizer(RecordingBackend(), LifecycleRecord(fail_clear=true, fail_finish=true))
        caught = with_logger(ThrowingLogger()) do
            try
                sample(broken; callback=(args...)->throw(failure))
                nothing
            catch err
                err
            end
        end
        @test caught === failure
        @test :aborted in broken.content.events
        @test :incomplete in broken.content.events
        @test last(broken.content.events) == :cleanup_failed
    end

    @testset "Owner interruption remains primary" begin
        vis = BATVisualizer(InterruptingBackend())
        @test_throws InterruptException BAT._safe_finish_visualizer!(vis)
        @test_throws InterruptException BAT._safe_clear_visualizer_init!(vis)
        with_logger(ThrowingLogger()) do
            @test BAT._safe_finish_visualizer!(vis; success=false) === nothing
            @test BAT._safe_clear_visualizer_init!(vis; suppress_interrupt=true) === nothing
        end
    end

    @testset "Child error releases every observing session before join" begin
        firstvis = BATVisualizer(BlockingBackend(), BlockingRecord(throw_stop=true))
        secondvis = BATVisualizer(BlockingBackend(), BlockingRecord())
        failure = ErrorException("numerical step failure")
        failstep = () -> begin
            take!(firstvis.content.entered)
            take!(secondvis.content.entered)
            throw(failure)
        end
        states = [state(firstvis, 1), state(secondvis, 2), state(firstvis, 3, failstep)]
        outputs = [DensitySampleVector[] for _ in states]
        caught = with_logger(ThrowingLogger()) do
            try
                BAT.mcmc_iterate!!(outputs, states; max_nsteps=1, update_visualizer=true)
                nothing
            catch err
                err
            end
        end
        @test contains_exception(caught, failure)
        @test firstvis.content.stopped && secondvis.content.stopped
        @test firstvis.content.stop_calls == secondvis.content.stop_calls == 1
    end

    @testset "Non-observing groups do not stop sessions" begin
        for output_enabled in (false, true), observer_enabled in (false, true)
            output_enabled && observer_enabled && continue
            vis = BATVisualizer(BlockingBackend(), BlockingRecord())
            failure = ErrorException("non-observing step failure")
            states = [state(vis, 1, ()->throw(failure))]
            outputs = output_enabled ? [DensitySampleVector[]] : nothing
            caught = try
                BAT.mcmc_iterate!!(outputs, states; max_nsteps=1, update_visualizer=observer_enabled)
                nothing
            catch err
                err
            end
            @test contains_exception(caught, failure)
            @test vis.content.stop_calls == 0
        end
    end
end
end
