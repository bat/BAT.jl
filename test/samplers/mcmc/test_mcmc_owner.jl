# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestMCMCOwner

using BAT, Test

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
function state(id, step=()->nothing)
    BAT.MCMCState(OwnerIterator(BATContext(), (id=Int32(id),), 0, 0, nothing, step),
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

@testset "MCMC producer ownership" begin
    @testset "Empty and invalid groups launch no work" begin
        empty_states = BAT.MCMCState[]
        @test BAT.mcmc_iterate!!(nothing, empty_states) === empty_states
        states = [state(1), state(2)]
        @test_throws DimensionMismatch BAT.mcmc_iterate!!(outputs(states)[1:1], states)
        @test all(s -> isnothing(s.chain_state.worker), states)
    end

    @testset "A child failure cancels and joins another active worker" begin
        gate = Gate()
        failure = ErrorException("numerical child failure")
        states = [state(1, () -> (enter!(gate); yield())), state(2, () -> begin
            await(() -> entered(gate) == 1) || error("first worker did not enter")
            throw(failure)
        end)]
        owner = @async catch_result(() -> BAT.mcmc_iterate!!(outputs(states), states; max_nsteps=10_000))
        try
            @test await(() -> !isnothing(states[2].chain_state.worker) && istaskdone(states[2].chain_state.worker))
            @test !istaskdone(owner)
            @test !istaskdone(states[1].chain_state.worker)
            release!(gate)
            @test await(() -> istaskdone(owner))
            @test contains_exception(finished_result(owner).exception, failure)
            @test all(s -> istaskdone(s.chain_state.worker), states)
            @test 1 <= states[1].chain_state.stepno < 10_000
            @test states[1].chain_state.flushes == 1
            snapshot = [(s.chain_state.stepno, s.chain_state.flushes) for s in states]
            yield()
            @test [(s.chain_state.stepno, s.chain_state.flushes) for s in states] == snapshot
        finally
            settle!(owner, (gate,); workers=[s.chain_state.worker for s in states])
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

end # module
