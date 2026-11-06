# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Keep Base's one-time scheduler Task creation outside seeded RNG comparisons.
function _prepare_makie_rng_test!()
    Threads.threadid() == 1 || error("Makie RNG tests must prepare their owning thread")
    if isdefined(Base, :get_sched_task)
        scheduler = Base.get_sched_task()
        scheduler isa Task && !istaskdone(scheduler) ||
            error("Unsupported Julia scheduler state for Makie RNG tests")
    elseif VERSION >= v"1.13.0-DEV"
        error("Julia scheduler setup changed; update the Makie RNG test helper")
    end
    return nothing
end
