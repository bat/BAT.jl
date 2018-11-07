# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc """
    struct ExecContext
        use_threads::Bool
        onprocs::Vector{Int64}
    end

Functions that take an `ExecContext` argument must limit their use of
threads and processes accordingly. Depending on `use_threads`, the function
may use all (or only a single) thread(s) on each process in `onprocs` (in
addition to the current thread on the current process).

The caller may choose to change the `ExecContext` from call to call, based on
execution time and latency measurements, etc.

Functions can announce their [`BAT.ExecCapabilities`](@ref) via [`exec_capabilities`](@ref).
"""
struct ExecContext
    use_threads::Bool
    onprocs::Vector{Int64}
end

export ExecContext

ExecContext() = ExecContext(true, Distributed.workers())


@doc """
    struct ExecCapabilities
        nthreads::Int
        threadsafe::Bool
        nprocs::Int
        remotesafe::Bool
    end

Specifies the execution capabilities of functions that support an
`ExecContext` argument.

`nthreads` specifies the maximum number of threads the function can utilize
efficiently, internally. If `nthreads <= 1`, the function implementation is
single-threaded. `nthreads == 0` indicates that the function is cheap and
that when used in combination with other functions, their capabilities should
dominate.

`threadsafe` specifies whether the function is thread-safe, and can be can be
run on multiple threads in parallel by the caller.

`nprocs` specifies the maximum number of worker processes the function can
utilize efficiently, internally. If `procs <= 1`, the function cannot
use worker processes. `nthreads == 0` carries equivalent meaning to
`nthreads == 0`.

`remotesafe` specifies that the function can be run on a remote thread,
it implies that the function arguments can be (de-)serialized safely.

Functions with an `ExecContext` argument should announce their capabilities
via methods of `exec_capabilities`. Functions should, ideally, either support
internal multithreading (`nthreads > 1`) or be thread-safe
(`threadsafe` == true). Likewise, functions should either utilize worker
processes (`nprocs` > 1) internally or support remote execution
(`remotesafe` == true) by the caller.
"""
struct ExecCapabilities
    nthreads::Int
    threadsafe::Bool
    nprocs::Int
    remotesafe::Bool
end

export ExecCapabilities



@doc """
    intersect(a:ExecCapabilities, b:ExecCapabilities)

Get the intersection of execution capabilities of a and b, i.e. the
`ExecCapabilities` that should be used when to functions are used in
combination (e.g. in sequence).
"""
Base.intersect(a::ExecCapabilities, b::ExecCapabilities) = ExecCapabilities(
    if a.nthreads == 0
        b.nthreads
    elseif b.nthreads == 0
        a.nthreads
    else
        # ToDo/Decision: Better use maximum?
        min(a.nthreads, b.nthreads)
    end,
    a.threadsafe && b.threadsafe,
    if a.nprocs == 0
        b.nprocs
    elseif b.nprocs == 0
        a.nprocs
    else
        # ToDo/Decision: Better use maximum?
        min(a.nprocs, b.nprocs)
    end,
    a.remotesafe && a.remotesafe
)



@doc """
    exec_capabilities(f, args...)::ExecCapabilities

Determines the execution capabilities of a function `f` that supports an
`ExecContext` argument, when called with arguments `args...`. The
`ExecContext` argument itself is excluded from `args...` for
`exec_capabilities`.

Before calling `f`, the caller must use

    exec_capabilities(f, args...)

to determine the execution capabilities of `f` with the intended arguments,
and take the resulting `ExecCapabilities` into account. If `f` is not
thread-safe (but remote-safe), and the caller needs to run it on multiple
threads, the caller may deep-copy the function arguments.
"""
function exec_capabilities end



function negotiate_exec_context(context::ExecContext, target_caps::AbstractVector{ExecCapabilities})
    target_caps_threadsafe = all(x -> x.threadsafe, target_caps)
    target_caps_nthreads = minimum(x -> x.nthreads, target_caps)

    if context.use_threads
        target_use_threads = target_caps_nthreads > 1
        self_use_threads = target_caps_threadsafe && !target_use_threads
    else
        target_use_threads = false
        self_use_threads = false
    end

    target_caps_remotesafe = all(x -> x.remotesafe, target_caps)

    # ToDo: Take different nprocs into account
    target_caps_nprocs = minimum(x -> x.nprocs, target_caps)

    nprocs_avail = length(context.onprocs)
    if isempty(context.onprocs)
        target_onprocs = [myid()]
        self_onprocs = [myid()]
    elseif nprocs_avail == 1 && context.onprocs[1] == myid()
        target_onprocs = context.onprocs
        self_onprocs = context.onprocs
    else
        if target_caps_remotesafe
            # ToDo: Distribute available processes for targets
            error("Not implemented yet")
            target_onprocs = error("Not implemented yet")
            self_onprocs = error("Not implemented yet")
        else
            # ToDo: Distribute available processes over targets
            target_onprocs = error("Not implemented yet")
            self_onprocs = [myid()]
        end
    end

    self_context = ExecContext(self_use_threads, target_onprocs)
    # ToDo: Improve naive implementation:
    target_context = map(x -> ExecContext(target_use_threads, self_onprocs), target_caps)

    (self_context, target_context)
end
