# This file is a part of BAT.jl, licensed under the MIT License (MIT).
# Run only through runtests.py: native SIGINT and hangs are process-isolated.
using BAT, Test, Logging

@test realpath(pathof(BAT)) == realpath(joinpath(@__DIR__, "..", "..", "..", "src", "BAT.jl"))

Base.exit_on_sigint(false) # Test application policy, not a BAT library setting.
const CASE = ARGS[1]
const GATES = ARGS[2]
const PRINT_LOCK = ReentrantLock()

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
    lock::ReentrantLock
    workers::Vector{Task}
end
NativeRecord() = NativeRecord(ReentrantLock(), Task[])

struct BrokenDiagnosticLogger <: AbstractLogger end
Logging.min_enabled_level(::BrokenDiagnosticLogger) = Logging.Warn
Logging.shouldlog(::BrokenDiagnosticLogger, level, mod, group, id) = group === :mcmc_algorithm
Logging.catch_exceptions(::BrokenDiagnosticLogger) = false
Logging.handle_message(::BrokenDiagnosticLogger, args...; kwargs...) = error("diagnostic failure")

# Only fixture-owned numerical steps/storage are replaced. The production
# scalar/vector loops, cancellation and joining remain unchanged.
mutable struct NativeIterator <: BAT.MCMCIterator
    context::BATContext
    info::NamedTuple{(:id,),Tuple{Int32}}
    stepno::Int
    flushes::Int
    record::NativeRecord
end
BAT.nsteps(c::NativeIterator) = c.stepno
BAT.nsamples(::NativeIterator) = 0
function BAT.mcmc_step!!(s::BAT.MCMCState{NativeIterator})
    c = s.chain_state
    lock(c.record.lock) do
        @test Threads.threadid() != 1 # Native SIGINT must target the owner.
        push!(c.record.workers, current_task())
        event("ENTERED ", length(c.record.workers))
    end
    await_gate("release_workers")
    CASE == "vector_drain" && c.info.id == 2 && error("later numerical child failure")
    c.stepno += 1
    s
end
function BAT.flush_samples!!(s::BAT.MCMCState{NativeIterator})
    s.chain_state.flushes += 1
    s
end
BAT.get_samples!(out, ::BAT.MCMCState{NativeIterator}, ::Bool) = out
state(record, id) = BAT.MCMCState(
    NativeIterator(BATContext(), (id=Int32(id),), 0, 0, record),
    BAT.NoMCMCProposalTunerState(), BAT.NoMCMCTransformTuningState(),
    BAT.MCMCNoTemperingState())

function run_case()
    @test Threads.threadid() == 1
    @test Threads.threadpool() === :interactive
    record = NativeRecord()
    states = [state(record, i) for i in 1:2]
    caught = false
    try
        with_logger(BrokenDiagnosticLogger()) do
            BAT.mcmc_iterate!!([DensitySampleVector[] for _ in states], states; max_nsteps=8)
        end
    catch err
        caught = true
        # Assert before any fixture cleanup: no worker may outlive the real
        # production boundary, including after the second SIGINT and child error.
        @test err isa InterruptException
        @test length(record.workers) == 2
        @test all(istaskdone, record.workers)
        @test count(istaskfailed, record.workers) == Int(CASE == "vector_drain")
        @test [s.chain_state.stepno for s in states] == (CASE == "vector_drain" ? [1, 0] : [1, 1])
        @test [s.chain_state.flushes for s in states] == (CASE == "vector_drain" ? [1, 0] : [1, 1])
        before = [(s.chain_state.stepno, s.chain_state.flushes) for s in states]
        sleep(0.05)
        @test [(s.chain_state.stepno, s.chain_state.flushes) for s in states] == before
    end
    @test caught
    event("RESULT ", CASE)
end

CASE in ("vector_owner", "vector_drain") || error("unknown case: $CASE")
println("Julia ", VERSION, "; BAT=", pathof(BAT), "; threads=", Threads.nthreads())
run_case()
