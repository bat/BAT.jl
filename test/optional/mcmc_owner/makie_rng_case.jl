# This file is a part of BAT.jl, licensed under the MIT License (MIT).
# Opt-in child of runtests.py; use a project containing BAT, Makie, Random, Test.
# The native signal targets a stable real finish wait, not an artificially
# enlarged task-enrollment instruction interval. No renderer is constructed.
# This is explicitly a scheduler-initialized gate; the runtime's first idle
# wait is characterized below before any child-stream measurement.
using BAT, Makie, Random, Test

@test realpath(pathof(BAT)) == realpath(joinpath(@__DIR__, "..", "..", "..", "src", "BAT.jl"))
const EXT = Base.get_extension(BAT, :BATMakieExt)
isnothing(EXT) && error("BATMakieExt failed to load")
const CASE = ARGS[1]
const GATES = ARGS[2]
CASE == "makie_rng" || error("unknown case: $CASE")
Base.exit_on_sigint(false) # Only this isolated test application's signal policy.

function event(parts...)
    println("BAT_OWNER ", parts...)
    flush(stdout)
    return nothing
end

function waitfor(predicate; seconds=20.0)
    ready = timedwait(predicate, seconds; pollint=0.005) === :ok
    @test ready
    ready || error("Native Makie fixture barrier timed out")
    return nothing
end

function join_owned(task)
    isnothing(task) && return nothing
    waitfor(() -> istaskdone(task))
    @test !istaskfailed(task)
    wait(task)
    return nothing
end

function characterize_scheduler_wait!()
    before = copy(Random.default_rng())
    first_result = timedwait(() -> false, 0.02; pollint=0.001)
    after = copy(Random.default_rng())
    println("BASE_WAIT Julia=", VERSION, " before=", before, " after=", after,
        " full_state_equal=", before == after)
    flush(stdout)
    @test first_result === :timed_out
    @test rand(copy(before), UInt64, 8) == rand(copy(after), UInt64, 8)
    # Julia 1.13 may initialize a per-thread scheduler Task on the first idle
    # wait, charging whichever task first reaches that path. Never rewind that
    # real runtime work or require its cold full-state effect to be neutral.
    @test timedwait(() -> false, 0.02; pollint=0.001) === :timed_out
    @test Random.default_rng() == after
    event("SCHEDULER_CONTROL complete")
    return nothing
end

# This counterfactual measurement creates real children and restores its own
# complete measuring-task state. It does not wrap production work. Retaining
# and joining every child also avoids attributing test-harness forks to BAT.
function child_vectors_from(state)
    current = copy(Random.default_rng())
    children = Task[]
    try
        copy!(Random.default_rng(), state)
        for _ in 1:3
            child = @task rand(UInt64, 8)
            push!(children, child)
            schedule(child)
        end
        return fetch.(children)
    finally
        copy!(Random.default_rng(), current)
        foreach(join_owned, children)
    end
end

function run_case()
    @test Threads.threadid() == 1
    @test Threads.threadpool() === :interactive
    # This visible Base-only control precedes both the observation checkpoint
    # and the child-vector probe, which itself yields to joined child tasks.
    characterize_scheduler_wait!()
    vis = BATVisualizer(BATMakieVisualization(
        adaptive_batching=false, observer_timeout=20.0))
    ctl = vis.content.lifecycle
    vis.content.n_dof[] = 3
    released = Threads.Atomic{Bool}(false)
    listener_release = Channel{Nothing}(1)
    function release_listener!()
        if !Threads.atomic_cas!(released, false, true)
            put!(listener_release, nothing)
        end
        return nothing
    end
    # Establish the harness's numerical-pool task before the RNG checkpoint.
    # It cannot receive the owner-thread SIGINT and only releases our barrier.
    controller = Threads.@spawn begin
        @test Threads.threadid() != 1
        deadline = time() + 30
        try
            while !released[] && !isfile(joinpath(GATES, "release_listener"))
                time() < deadline || error("external driver did not release listener")
                sleep(0.005)
            end
        finally
            release_listener!()
        end
    end
    listener = nothing
    timer = nothing
    coordinator = nothing
    try
        EXT._claim_live_visualizer!(vis)
        before = copy(Random.default_rng())
        expected_children = child_vectors_from(before)
        @test Random.default_rng() == before
        private_before = copy(ctl.task_rng)
        finish! = () -> begin
            event("MAKIE_FINISH_WAIT")
            take!(listener_release)
            return nothing
        end
        EXT._start_live_listener!(vis, () -> nothing, finish!)
        listener, timer = lock(vis.content.buffer_lock) do
            ctl.task, ctl.timer
        end
        @test listener isa Task
        @test timer isa Timer
        @test Random.default_rng() == before
        @test ctl.task_rng != private_before
        private_after_start = copy(ctl.task_rng)

        interrupted = false
        try
            # The listener's finish callable blocks on our owned channel. The
            # external driver sends SIGINT only after that callable announces
            # its barrier, while this real production finish waits for it.
            BAT.finish_visualizer!(vis; success=true)
        catch err
            interrupted = true
            @test err isa InterruptException
            @test Random.default_rng() == before
            @test !istaskdone(listener)
            @test !released[]
            @test ctl.phase === :stopping
            @test ctl.outcome === :aborted
            @test ctl.task_rng == private_after_start

            # This is the real current-owner cleanup path after delivery:
            # coordinator construction, storage and scheduling are unchanged.
            BAT.dispose_visualizer!(vis)
            coordinator = lock(() -> ctl.disposal_task, vis.content.buffer_lock)
            @test coordinator isa Task
            @test !istaskdone(coordinator)
            @test !istaskdone(listener)
            @test Random.default_rng() == before
            @test ctl.task_rng != private_after_start
            private_after_dispose = copy(ctl.task_rng)
            event("MAKIE_DISPOSING")

            join_owned(controller)
            join_owned(listener)
            join_owned(coordinator)
            # Outcome checks precede the finally block's emergency releases.
            @test !isopen(timer)
            @test isnothing(ctl.timer)
            @test isnothing(ctl.task)
            @test isnothing(vis.content.graph[])
            status = BAT.visualizer_status(vis)
            @test status.quiesced
            @test status.history_released
            @test !status.cleanup_pending
            @test status.outcome === :aborted
            @test ctl.task_rng == private_after_dispose
            @test Random.default_rng() == before
            @test child_vectors_from(copy(Random.default_rng())) == expected_children
            @test Random.default_rng() == before

            # A completed repeated disposal must neither fork nor reseed.
            BAT.dispose_visualizer!(vis)
            @test ctl.task_rng == private_after_dispose
            @test Random.default_rng() == before
        end
        @test interrupted
    finally
        # Release/join only fixture-owned resources. These operations cannot
        # turn an earlier failed assertion or a missing native signal into pass.
        release_listener!()
        BAT.dispose_visualizer!(vis)
        join_owned(controller)
        join_owned(listener)
        pending = lock(() -> ctl.disposal_task, vis.content.buffer_lock)
        join_owned(pending)
        isnothing(coordinator) || coordinator === pending || join_owned(coordinator)
    end
    event("RESULT ", CASE)
    return nothing
end

println("Julia ", VERSION, "; BAT=", pathof(BAT), "; threads=", Threads.nthreads())
@testset "Native SIGINT with initialized scheduler preserves BAT observer RNG ownership" begin
    run_case()
end
