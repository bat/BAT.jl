# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestMCMCOwner

using BAT, Test, Distributions, Logging

# Every injected exception below originates in a dispatched hook or a condition
# owned by this fixture. Never schedule an exception into an arbitrary task.
mutable struct Gate
    condition::Threads.Condition
    entered::Int
    open::Bool
end
Gate() = Gate(Threads.Condition(), 0, false)
function enter!(gate::Gate)
    lock(gate.condition) do
        gate.entered += 1
        notify(gate.condition; all=true)
        while !gate.open
            wait(gate.condition)
        end
    end
    nothing
end
function release!(gate::Gate)
    lock(gate.condition) do
        gate.open = true
        notify(gate.condition; all=true)
    end
    nothing
end
entered(gate::Gate) = lock(() -> gate.entered, gate.condition)
function interrupt_waiter!(gate::Gate, exception)
    lock(gate.condition) do
        notify(gate.condition, exception; all=true, error=true)
    end
end
await(predicate) = timedwait(predicate, 20; pollint=0.001) == :ok

function settle!(owner, gates; workers=())
    foreach(release!, gates)
    # Cleanup releases only fixture-owned waits. Assertions about premature
    # return and live children must be made before entering this finally path.
    await(() -> istaskdone(owner)) || error("MCMC owner fixture did not settle")
    for worker in workers
        isnothing(worker) && continue
        await(() -> istaskdone(worker)) || error("MCMC producer fixture did not settle")
    end
    nothing
end
function finished_result(owner)
    istaskdone(owner) || error("MCMC owner did not finish before the test deadline")
    fetch(owner)
end

contains_exception(err, expected) = err === expected ||
    (err isa CompositeException && any(e -> contains_exception(e, expected), err.exceptions)) ||
    (err isa TaskFailedException && any(e -> contains_exception(e.exception, expected), Base.current_exceptions(err.task)))

struct HookBackend <: BAT.BATVisBackend end
BAT.request_visualizer_stop!(vis::BATVisualizer{HookBackend}; kwargs...) = vis.content.stop()
BAT.update_visualizer_impl!(vis::BATVisualizer{HookBackend}; kwargs...) = vis.content.update()

struct BrokenDiagnosticLogger <: AbstractLogger end
Logging.min_enabled_level(::BrokenDiagnosticLogger) = Logging.Warn
Logging.shouldlog(::BrokenDiagnosticLogger, level, mod, group, id) = group === :bat_visualizer || group === :mcmc_algorithm
Logging.catch_exceptions(::BrokenDiagnosticLogger) = false
Logging.handle_message(::BrokenDiagnosticLogger, args...; kwargs...) = error("diagnostic failure")

mutable struct OwnerIterator <: BAT.MCMCIterator
    context::BATContext
    info::NamedTuple{(:id,),Tuple{Int32}}
    stepno::Int
    flushes::Int
    worker::Union{Nothing,Task}
    step::Function
end
BAT.nsteps(c::OwnerIterator) = c.stepno
BAT.nsamples(::OwnerIterator) = 0
function BAT.mcmc_step!!(s::BAT.MCMCState{OwnerIterator})
    c = s.chain_state
    c.worker = current_task()
    c.step()
    c.stepno += 1
    s
end
function BAT.flush_samples!!(s::BAT.MCMCState{OwnerIterator})
    s.chain_state.flushes += 1
    s
end
BAT.get_samples!(out, ::BAT.MCMCState{OwnerIterator}, ::Bool) = out
function state(id, step=()->nothing; vis=BATVisualizer())
    BAT.MCMCState(OwnerIterator(BATContext(visualizer=vis), (id=Int32(id),), 0, 0, nothing, step),
        BAT.NoMCMCProposalTunerState(), BAT.NoMCMCTransformTuningState(), BAT.MCMCNoTemperingState())
end
outputs(states) = [DensitySampleVector[] for _ in states]
function catch_result(f)
    try
        f()
        (exception=nothing, backtrace=nothing)
    catch err
        (exception=err, backtrace=catch_backtrace())
    end
end
owner_stop_interrupt(exception) = throw(exception)

@testset "MCMC producer ownership" begin
    @testset "Empty and invalid groups launch no work" begin
        empty_states = BAT.MCMCState[]
        @test BAT.mcmc_iterate!!(nothing, empty_states) === empty_states
        states = [state(1), state(2)]
        @test_throws DimensionMismatch BAT.mcmc_iterate!!(outputs(states)[1:1], states)
        @test all(s -> isnothing(s.chain_state.worker), states)
    end

    @testset "Owner interruption preserves active producers until join" begin
        numerical_gate = Gate()
        stop_calls = Threads.Atomic{Int}(0)
        primary = InterruptException()
        child_failure = ErrorException("later numerical child failure")
        stop = () -> begin
            n = Threads.atomic_add!(stop_calls, 1)
            n < 3 && owner_stop_interrupt(primary)
            nothing
        end
        vis = BATVisualizer(HookBackend(), (stop=stop, update=()->nothing))
        states = [state(1, () -> enter!(numerical_gate); vis),
                  state(2, () -> begin
                      await(() -> entered(numerical_gate) == 1) || error("first producer did not enter")
                      throw(child_failure)
                  end; vis)]
        owner = @async with_logger(BrokenDiagnosticLogger()) do
            catch_result(() -> BAT.mcmc_iterate!!(outputs(states), states;
                max_nsteps=8, update_visualizer=true))
        end
        try
            @test await(() -> stop_calls[] >= 4)
            @test !istaskdone(owner)
            @test !istaskdone(states[1].chain_state.worker)
            release!(numerical_gate)
            @test await(() -> istaskdone(owner))
            result = finished_result(owner)
            @test result.exception === primary
            @test any(frame -> frame.func === :owner_stop_interrupt, stacktrace(result.backtrace))
            @test all(s -> istaskdone(s.chain_state.worker), states)
            @test states[1].chain_state.stepno == 1
            @test states[1].chain_state.flushes == 1
            @test states[2].chain_state.flushes == 0
            snapshot = [(s.chain_state.stepno, s.chain_state.flushes) for s in states]
            yield()
            @test [(s.chain_state.stepno, s.chain_state.flushes) for s in states] == snapshot
        finally
            settle!(owner, (numerical_gate,); workers=[s.chain_state.worker for s in states])
        end
    end

    @testset "Distinct blocked sessions release before join: retained=$retained" for retained in (false, true)
        gates = [Gate(), Gate()]
        calls = [Threads.Atomic{Int}(0), Threads.Atomic{Int}(0)]
        primary = InterruptException()
        sessions = [BATVisualizer(HookBackend(), (
            update=() -> enter!(gates[k]),
            stop=() -> begin
                attempt = Threads.atomic_add!(calls[k], 1)
                k == 1 && attempt == 0 && owner_stop_interrupt(primary)
                release!(gates[k])
                k == 1 && error("stop failed after releasing its producer")
                nothing
            end)) for k in 1:2]
        failure = ErrorException("numerical trigger")
        states = [state(1; vis=sessions[1]), state(2; vis=sessions[2]),
            state(3, () -> begin
                await(() -> all(g -> entered(g) == 1, gates)) || error("observers did not block")
                throw(failure)
            end; vis=sessions[1])]
        owner = @async with_logger(BrokenDiagnosticLogger()) do
            catch_result(() -> BAT.mcmc_iterate!!(retained ? outputs(states) : nothing, states;
                max_nsteps=8, update_visualizer=true))
        end
        try
            @test await(() -> istaskdone(owner))
            @test finished_result(owner).exception === primary
            @test calls[1][] == 2
            @test calls[2][] == 1
            @test all(s -> istaskdone(s.chain_state.worker), states)
            @test [s.chain_state.stepno for s in states] == [1, 1, 0]
        finally
            settle!(owner, gates; workers=[s.chain_state.worker for s in states])
        end
    end

    @testset "Multiple child failures retain launch order and original stacks" begin
        gate = Gate()
        failures = [ErrorException("first chain failure"), ErrorException("second chain failure")]
        states = [state(k, () -> (enter!(gate); throw(failures[k]))) for k in 1:2]
        owner = @async catch_result(() -> BAT.mcmc_iterate!!(nothing, states; max_nsteps=8))
        try
            @test await(() -> entered(gate) == 2)
            release!(gate)
            @test await(() -> istaskdone(owner))
            result = finished_result(owner).exception
            @test result isa CompositeException
            @test length(result.exceptions) == 2
            for k in 1:2
                @test result.exceptions[k].task === states[k].chain_state.worker
                @test contains_exception(result.exceptions[k], failures[k])
                @test !isempty(Base.current_exceptions(result.exceptions[k].task))
            end
            @test all(s -> istaskdone(s.chain_state.worker), states)
        finally
            settle!(owner, (gate,); workers=[s.chain_state.worker for s in states])
        end
    end

    @testset "Only the enrolled prefix is drained" begin
        for n in (0, 1, 2)
            gate = Gate()
            enrolled = [Threads.@spawn enter!(gate) for _ in 1:n]
            unstarted = Task(() -> error("an unlaunched task must not run"))
            ledger = [enrolled; unstarted]
            owner = @async Base.disable_sigint() do
                BAT._mcmc_wait_tasks(ledger, n; unwinding=true)
            end
            try
                @test await(() -> entered(gate) == n)
                n > 0 && @test !istaskdone(owner)
                release!(gate)
                @test await(() -> istaskdone(owner))
                @test !istaskfailed(owner)
                @test all(istaskdone, enrolled)
                @test !istaskstarted(unstarted)
            finally
                settle!(owner, (gate,); workers=enrolled)
            end
        end
    end

    @testset "A failed task is detected without entering its body" begin
        gate = Gate()
        pending = Threads.@spawn enter!(gate)
        body_entered = Ref(false)
        failure = ErrorException("failure before the fixture task body")
        failed = Task(() -> (body_entered[] = true))
        # This task is fresh and owned exclusively by the fixture. Scheduling
        # its initial exception is supported; no running task is interrupted.
        schedule(failed, failure; error=true)
        ledger = [pending, failed]
        try
            @test await(() -> entered(gate) == 1)
            @test await(() -> istaskdone(failed))
            @test istaskfailed(failed)
            @test !body_entered[]
            @test contains_exception(TaskFailedException(failed), failure)
            all_done, detected = Base.disable_sigint() do
                BAT._mcmc_scan_tasks(ledger, length(ledger))
            end
            @test !all_done
            @test detected === failed
            @test !istaskdone(pending)

            release!(gate)
            @test await(() -> istaskdone(pending))
            all_done, detected = Base.disable_sigint() do
                BAT._mcmc_scan_tasks(ledger, length(ledger))
            end
            @test all_done
            @test detected === failed
        finally
            settle!(pending, (gate,); workers=(failed,))
        end
    end

    @testset "The owner closes its live repeating timer" begin
        for unwinding in (false, true)
            @test Base.disable_sigint() do
                BAT._mcmc_close_timer(nothing; unwinding) === nothing
            end
            # A long first delay keeps the resource live throughout this test;
            # closure, not a naturally expired tick, must make isopen false.
            timer = Timer(60.0; interval=60.0)
            try
                @test isopen(timer)
                @test Base.disable_sigint() do
                    BAT._mcmc_close_timer(timer; unwinding) === nothing
                end
                @test !isopen(timer)
                @test Base.disable_sigint() do
                    BAT._mcmc_close_timer(timer; unwinding) === nothing
                end
                @test !isopen(timer)
            finally
                # Independently release the fixture resource even if a broken
                # close helper caused an assertion above to fail.
                close(timer)
            end
        end
    end
end

# Exercise real pool/retry initialization, burn-in, stabilization and main
# sampling. The numerical state is genuine; only observation hooks supply the
# barriers. Initialization-only groups intentionally have no stop participant.
struct PhaseBackend <: BAT.BATVisBackend end
mutable struct PhaseRecord
    lock::ReentrantLock
    phase::Symbol
    producer_gate::Gate
    stop_gate::Gate
    active::Int
    workers::Vector{Task}
    events::Vector{Symbol}
    clear_active::Vector{Int}
    clear_workers_done::Vector{Bool}
    finish_active::Vector{Int}
    segments::Vector{Tuple{Int,Int}}
    stop_calls::Int
    failure::Exception
    stop_owner::Union{Nothing,Task}
end
PhaseRecord(phase) = PhaseRecord(ReentrantLock(), phase, Gate(), Gate(), 0, Task[],
    Symbol[], Int[], Bool[], Int[], Tuple{Int,Int}[], 0, ErrorException("phase producer failure: $phase"), nothing)
snapshot(c::PhaseRecord) = lock(c.lock) do
    (active=c.active, workers=copy(c.workers), events=copy(c.events),
        clear_active=copy(c.clear_active), finish_active=copy(c.finish_active),
        clear_workers_done=copy(c.clear_workers_done),
        segments=copy(c.segments), stop_calls=c.stop_calls, stop_owner=c.stop_owner)
end
BAT.prepare_visualizer_init!(::BATVisualizer{PhaseBackend}, alg) = nothing
BAT.capture_visualizer_init(::BATVisualizer{PhaseBackend}) = true
BAT.begin_visualizer_init_segment(::BATVisualizer{PhaseBackend}, chain, output) = chain.info.cycle
function BAT.clear_visualizer_init!(vis::BATVisualizer{PhaseBackend})
    c = vis.content
    lock(c.lock) do
        push!(c.clear_active, c.active)
        push!(c.clear_workers_done, all(istaskdone, c.workers))
        push!(c.events, :clear)
        empty!(c.segments)
    end
    nothing
end
function BAT.init_visualizer!(vis::BATVisualizer{PhaseBackend}; kwargs...)
    lock(() -> push!(vis.content.events, :initialized), vis.content.lock)
    nothing
end
function BAT.finish_visualizer!(vis::BATVisualizer{PhaseBackend}; success=true)
    c = vis.content
    lock(c.lock) do
        push!(c.finish_active, c.active)
        push!(c.events, success ? :complete : :incomplete)
    end
    nothing
end
function phase_barrier!(c::PhaseRecord, chain)
    lock(c.lock) do
        c.active += 1
        push!(c.workers, current_task())
    end
    try
        if chain.info.id == 1
            enter!(c.producer_gate)
        else
            await(() -> entered(c.producer_gate) == 1) || error("phase producer did not enter")
            throw(c.failure)
        end
    finally
        lock(c.lock) do
            c.active -= 1
            push!(c.events, :producer_exited)
        end
    end
    nothing
end
function BAT.end_visualizer_init_segment!(vis::BATVisualizer{PhaseBackend}, before, chain, output)
    c = vis.content
    selected = (c.phase in (:pool_first, :retry) && before == 1) ||
        (c.phase === :pool_second && before == 2)
    selected && phase_barrier!(c, chain)
    lock(c.lock) do
        push!(c.segments, (chain.info.id, chain.info.cycle))
    end
    nothing
end
function BAT.update_visualizer_impl!(vis::BATVisualizer{PhaseBackend}; chain_state, is_sampling_step, kwargs...)
    c = vis.content
    cycle = c.phase === :burnin ? 2 : c.phase === :stabilization ? 3 : c.phase === :main ? 4 : 0
    is_sampling_step && chain_state.stepno == 1 && chain_state.info.cycle == cycle &&
        phase_barrier!(c, chain_state)
    nothing
end
function BAT.request_visualizer_stop!(vis::BATVisualizer{PhaseBackend}; kwargs...)
    c = vis.content
    first_stop = lock(c.lock) do
        c.stop_calls += 1
        c.stop_owner = current_task()
        c.stop_calls == 1
    end
    if c.phase in (:burnin, :stabilization, :main)
        # The test notifies an error only while the owner is waiting on this
        # fixture-owned condition. Retried stop then releases the real producer.
        first_stop && enter!(c.stop_gate)
        release!(c.producer_gate)
    end
    nothing
end

function sample_phase(c::PhaseRecord; store_burnin=true)
    init = c.phase in (:pool_first, :pool_second) ?
        MCMCChainPoolInit(nsteps_init=30, init_tries_per_chain=BAT.ClosedInterval(1, 1)) :
        MCMCRetryInit(nsteps_init=30)
    callback = (args...) -> lock(() -> push!(c.events, :callback), c.lock)
    alg = TransformedMCMC(nchains=2, nsteps=8, strict=false, store_burnin=store_burnin,
        proposal=RandomWalk(proposaldist=Normal(0, 0.2)),
        adaptive_transform=BAT.NoAdaptiveTransform(), transform_tuning=NoMCMCTransformTuning(),
        init=init, convergence=AssumeConvergence(), callback=callback,
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=10, max_ncycles=1, nsteps_final=5))
    bat_sample(Normal(), alg, BATContext(rng=BAT.Philox4x((124, 95)),
        visualizer=BATVisualizer(PhaseBackend(), c)))
end

@testset "Public MCMC phase cleanup follows producer termination" begin
    for phase in (:pool_first, :pool_second, :retry, :burnin, :stabilization, :main),
        retained in (phase === :stabilization ? (false, true) : (true,))
        @testset "$phase retained=$retained" begin
            c = PhaseRecord(phase)
            owner = @async with_logger(NullLogger()) do
                catch_result(() -> sample_phase(c; store_burnin=retained))
            end
            observing = phase in (:burnin, :stabilization, :main)
            try
                @test await(() -> entered(c.producer_gate) == 1)
                if observing
                    @test await(() -> entered(c.stop_gate) == 1)
                    @test snapshot(c).stop_owner === owner
                else
                    @test await(() -> length(snapshot(c).workers) == 2)
                end
                before = snapshot(c)
                @test !istaskdone(owner)
                @test before.active >= 1
                @test length(before.clear_active) == (observing ? 1 : 0)
                @test isempty(before.finish_active)
                if observing
                    interrupt_waiter!(c.stop_gate, InterruptException())
                else
                    release!(c.producer_gate)
                end
                @test await(() -> istaskdone(owner))
                result = finished_result(owner).exception
                @test observing ? result isa InterruptException : contains_exception(result, c.failure)
                after = snapshot(c)
                @test after.active == 0
                @test all(istaskdone, after.workers)
                @test all(iszero, after.clear_active)
                @test all(after.clear_workers_done)
                @test after.finish_active == [0]
                @test isempty(after.segments)
                @test :complete ∉ after.events
                @test last(after.events) === :incomplete
                @test count(==(:callback), after.events) == (retained && phase in (:stabilization, :main) ? 1 : 0)
                observing && @test after.stop_calls >= 2
                yield()
                @test snapshot(c).events == after.events
                @test isempty(snapshot(c).segments)
            finally
                settle!(owner, (c.producer_gate, c.stop_gate); workers=snapshot(c).workers)
            end
        end
    end
end

end
