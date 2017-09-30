# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct ExecContext
        use_threads::Bool
        onprocs::Vector{Int64}
    end

Functions that take an `ExecContext` argument must limit their use of
threads and processes accordingly. Depending on `use_threads`, the
function may use all (or only a single) thread(s) on each process in `onprocs`
(in addition to the current thread on the current process).

The caller may choose to change the `ExecContext` from call to call,
based on execution time and latency measurements, etc.

Functions can announce their `ExecCapabilities` via `exec_capabilities`.
"""
struct ExecContext
    use_threads::Bool
    onprocs::Vector{Int64}
end

export ExecContext

ExecContext() = ExecContext(true, workers())


"""
    struct ExecCapabilities
        nthreads::Int
        threadsafe::Bool
        nprocs::Int
        remotesafe::Bool
    end

Specifies the execution capabilities of functions that support an
`ExecContext` argument. 

`nthreads` specifies the maximum number of threads the function can
utilize efficiently, internally. If `nthreads <= 1`, the function
implementation is single-threaded.

`threadsafe` specifies whether the function is thread-safe, and can be can be
run on multiple threads in parallel by the caller.  

`nprocs` specifies the maximum number of worker processes the function can
utilize efficiently, internally. If `procs <= 1`, the function cannot
use worker processes.

`remotesafe` specifies that the function can be run on a remote thread,
it implies that the function arguments can be (de-)serialized safely.

Functions with an `ExecContext` argument should announce their capabilities
via methods of `exec_capabilities`. Functions should, ideally, either support
internal multithreading (`nthreads > 1`) of be thread-safe
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

ExecCapabilities() = ExecCapabilities(0, false, 0, false)


"""
    exec_capabilities(f, args...)::ExecCapabilities

Determines the execution capabilities of a function `f` that supports an
`ExecContext` argument, when called with arguments `args...`. The
`ExecContext` argument itself is excluded from `args...`, for
`exec_capabilities`.

Before calling `f`, the caller must use

    exec_capabilities(f, args...)

to determine the execution capabilities of `f` with the intended arguments,
and take the resulting `ExecCapabilities` into account. If `f` is not
thread-safe (but remote-safe), and the caller needs to run it on multiple
threads, the caller may deep-copy the function arguments.
"""
function exec_capabilities end



# abstract type AbstractExecutor end
# export AbstractExecutor
# 
# 
# struct SerialExecutor{RNG<:AbstractRNG} <: AbstractExecutor
#     rng::RNG
#     ec::ExecContext
# end
# 
# export SerialExecutor
# 
# SerialExecutor(rng::RNG) where {RNG<:AbstractRNG} =
#     SerialExecutor{RNG}(rng)
