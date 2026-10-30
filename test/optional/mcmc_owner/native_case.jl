# This file is a part of BAT.jl, licensed under the MIT License (MIT).
# Run only through runtests.py: native SIGINT and hangs are process-isolated.
using BAT, Test, Distributions

@test realpath(pathof(BAT)) == realpath(joinpath(@__DIR__, "..", "..", "..", "src", "BAT.jl"))

Base.exit_on_sigint(false) # Test application policy, not a BAT library setting.
const CASE = ARGS[1]
const GATES = ARGS[2]
const PRINT_LOCK = ReentrantLock()
const INIT_CASES = ("init_pool_first", "init_pool_second", "init_retry")

function event(parts...)
    lock(PRINT_LOCK) do
        println("BAT_OWNER ", parts...)
        flush(stdout)
    end
end

function await_gate(name)
    deadline = time() + 30
    while !isfile(joinpath(GATES, name))
        time() < deadline || error("external driver did not release $name")
        sleep(0.005)
    end
end

mutable struct NativeRecord
    condition::Threads.Condition
    workers::Vector{Task}
    stopped::Bool
    stop_attempts::Int
    primary::Any
    primary_backtrace::Any
    finish_outcomes::Vector{Bool}
end
NativeRecord() = NativeRecord(Threads.Condition(), Task[], false, 0,
    nothing, nothing, Bool[])
struct NativeBackend <: BAT.BATVisBackend end

function BAT.update_visualizer_impl!(vis::BATVisualizer{NativeBackend}; kwargs...)
    record = vis.content
    lock(record.condition) do
        worker = current_task()
        @test Threads.threadid() != 1 # Native SIGINT must target the owner.
        if all(t -> t !== worker, record.workers)
            push!(record.workers, worker)
            event("ENTERED ", length(record.workers))
        end
        while !record.stopped
            wait(record.condition)
        end
    end
    CASE == "vector_drain" && await_gate("release_workers")
    nothing
end

function BAT.request_visualizer_stop!(vis::BATVisualizer{NativeBackend};
    reason=:disabled, exception=nothing)
    record = vis.content
    record.stop_attempts += 1
    @test reason === :aborted
    if isnothing(record.primary) && exception isa CapturedException && exception.ex isa InterruptException
        record.primary = exception.ex
        # This dispatched hook executes inside the real vector owner's catch.
        # Observe its original raw trace without replacing production methods.
        current = filter(e -> e.exception === record.primary, Base.current_exceptions())
        @test !isempty(current)
        record.primary_backtrace = copy(last(current).backtrace)
    end
    if CASE == "vector_stop"
        event("STOP_WAIT ", record.stop_attempts)
        await_gate("release_stop")
    end
    lock(record.condition) do
        Base.disable_sigint() do
            record.stopped = true
            notify(record.condition; all=true)
        end
    end
    event("STOPPED")
    nothing
end

function BAT.finish_visualizer!(vis::BATVisualizer{NativeBackend}; success::Bool=true)
    record = vis.content
    @test all(istaskdone, record.workers)
    push!(record.finish_outcomes, success)
    event("FINISHED ", success)
    nothing
end

# Only the numerical step/storage are replaced. The production scalar/vector
# loops, observation dispatch, cancellation and joining remain unchanged.
mutable struct NativeIterator <: BAT.MCMCIterator
    context::BATContext
    info::NamedTuple{(:id,),Tuple{Int32}}
    stepno::Int
end
BAT.nsteps(c::NativeIterator) = c.stepno
BAT.nsamples(::NativeIterator) = 0
function BAT.mcmc_step!!(s::BAT.MCMCState{NativeIterator})
    s.chain_state.stepno += 1
    s
end
BAT.flush_samples!!(s::BAT.MCMCState{NativeIterator}) = s
BAT.get_samples!(out, s::BAT.MCMCState{NativeIterator}, nonzero::Bool) = out
state(vis, id) = BAT.MCMCState(
    NativeIterator(BATContext(visualizer=vis), (id=Int32(id),), 0),
    BAT.NoMCMCProposalTunerState(), BAT.NoMCMCTransformTuningState(),
    BAT.MCMCNoTemperingState())

function public_sample(vis)
    algorithm = TransformedMCMC(nchains=2, nsteps=8, strict=false,
        store_burnin=true, adaptive_transform=BAT.NoAdaptiveTransform(),
        transform_tuning=NoMCMCTransformTuning(),
        init=MCMCChainPoolInit(nsteps_init=30), convergence=AssumeConvergence(),
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=10, max_ncycles=1, nsteps_final=0))
    bat_sample(Normal(), algorithm,
        BATContext(rng=BAT.Philox4x((124, 95)), visualizer=vis))
end

function run_case()
    @test Threads.threadid() == 1
    @test Threads.threadpool() === :interactive
    record = NativeRecord()
    vis = BATVisualizer(NativeBackend(), record)
    states = CASE == "public_owner" ? nothing : [state(vis, i) for i in 1:2]
    caught = false
    try
        if isnothing(states)
            public_sample(vis)
        else
            BAT.mcmc_iterate!!([DensitySampleVector[] for _ in states], states;
                max_nsteps=8, update_visualizer=true)
        end
    catch err
        caught = true
        # Assert before any test-owned release/cleanup: no producer may outlive
        # the real production boundary, including after the second SIGINT.
        @test length(record.workers) == 2
        @test all(istaskdone, record.workers)
        @test err isa InterruptException
        @test err === record.primary
        @test catch_backtrace() == record.primary_backtrace
        if isnothing(states)
            @test record.finish_outcomes == [false]
        else
            @test [s.chain_state.stepno for s in states] == [1, 1]
            before = [s.chain_state.stepno for s in states]
            sleep(0.05)
            @test [s.chain_state.stepno for s in states] == before
            @test isempty(record.finish_outcomes)
        end
        @test CASE == "vector_stop" ? record.stop_attempts == 2 : record.stop_attempts >= 1
    end
    @test caught
    event("RESULT ", CASE)
end

mutable struct InitRecord
    lock::ReentrantLock
    workers::Vector{Task}
    active::Int
    segments::Vector{Tuple{Int,Int}}
    clear_checks::Vector{Bool}
    finish_outcomes::Vector{Bool}
    callbacks::Int
    primary::Any
    primary_backtrace::Any
end
InitRecord() = InitRecord(ReentrantLock(), Task[], 0, Tuple{Int,Int}[],
    Bool[], Bool[], 0, nothing, nothing)
struct InitBackend <: BAT.BATVisBackend end
BAT.capture_visualizer_init(::BATVisualizer{InitBackend}) = true
BAT.begin_visualizer_init_segment(::BATVisualizer{InitBackend}, chain, output) = chain.info.cycle

function BAT.end_visualizer_init_segment!(vis::BATVisualizer{InitBackend}, before, chain, output)
    record = vis.content
    selected = before == (CASE == "init_pool_second" ? 2 : 1)
    if selected
        lock(record.lock) do
            @test Threads.threadid() != 1
            push!(record.workers, current_task())
            record.active += 1
            event("INIT_ENTERED ", length(record.workers))
        end
        try
            await_gate("release_init")
            lock(record.lock) do
                push!(record.segments, (chain.info.id, chain.info.cycle))
            end
        finally
            lock(record.lock) do
                record.active -= 1
            end
        end
    else
        lock(record.lock) do
            push!(record.segments, (chain.info.id, chain.info.cycle))
        end
    end
    nothing
end

function BAT.request_visualizer_stop!(vis::BATVisualizer{InitBackend}; reason=:disabled, exception=nothing)
    record = vis.content
    @test reason === :aborted
    # Initialization groups have update_visualizer=false. Their vector owner
    # therefore does not call this hook; public cleanup reaches it after joins.
    @test all(istaskdone, record.workers)
    @test record.active == 0
    @test exception isa CapturedException
    @test exception.ex isa InterruptException
    record.primary = exception.ex
    active_exceptions = filter(e -> e.exception === record.primary, Base.current_exceptions())
    @test !isempty(active_exceptions)
    record.primary_backtrace = copy(last(active_exceptions).backtrace)
    event("INIT_STOPPED")
    nothing
end

function BAT.clear_visualizer_init!(vis::BATVisualizer{InitBackend})
    record = vis.content
    lock(record.lock) do
        settled = record.active == 0 && all(istaskdone, record.workers)
        push!(record.clear_checks, settled)
        @test settled
        empty!(record.segments)
    end
    nothing
end

function BAT.finish_visualizer!(vis::BATVisualizer{InitBackend}; success::Bool=true)
    record = vis.content
    @test record.active == 0
    @test all(istaskdone, record.workers)
    push!(record.finish_outcomes, success)
    nothing
end

function run_init_case()
    @test Threads.threadid() == 1
    @test Threads.threadpool() === :interactive
    record = InitRecord()
    vis = BATVisualizer(InitBackend(), record)
    init = CASE == "init_retry" ? MCMCRetryInit(nsteps_init=30) :
        MCMCChainPoolInit(nsteps_init=30, init_tries_per_chain=BAT.ClosedInterval(1, 1))
    callback = (args...) -> (record.callbacks += 1; nothing)
    algorithm = TransformedMCMC(nchains=2, nsteps=8, strict=false, store_burnin=true,
        proposal=RandomWalk(proposaldist=Normal(0, 0.2)),
        adaptive_transform=BAT.NoAdaptiveTransform(), transform_tuning=NoMCMCTransformTuning(),
        init=init, convergence=AssumeConvergence(), callback=callback,
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=10, max_ncycles=1, nsteps_final=5))
    caught = false
    try
        bat_sample(Normal(), algorithm,
            BATContext(rng=BAT.Philox4x((124, 95)), visualizer=vis))
    catch err
        caught = true
        @test err isa InterruptException
        @test err === record.primary
        @test catch_backtrace() == record.primary_backtrace
        @test length(record.workers) == 2
        @test all(istaskdone, record.workers)
        @test record.active == 0
        @test record.clear_checks == [true]
        @test record.finish_outcomes == [false]
        @test record.callbacks == 0
        @test isempty(record.segments)
        sleep(0.05)
        @test record.active == 0
        @test isempty(record.segments)
        @test record.callbacks == 0
    end
    @test caught
    event("RESULT ", CASE)
end

CASE in ("vector_owner", "vector_stop", "vector_drain", "public_owner", INIT_CASES...) || error("unknown case: $CASE")
println("Julia ", VERSION, "; BAT=", pathof(BAT), "; threads=", Threads.nthreads())
CASE in INIT_CASES ? run_init_case() : run_case()
