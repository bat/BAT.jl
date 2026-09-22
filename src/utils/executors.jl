# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type BATExecutor end


default_executor() = MultiThreadedExec()


struct SequentialExec <: BATExecutor end

function exec_map!(f::Base.Callable, executor::SequentialExec, Y::AbstractVector, X::AbstractVector)
    @argcheck length(eachindex(X)) == length(eachindex(X))
    for i in 0:(length(eachindex(Y)) - 1)
        Y[firstindex(Y) + i] = f(X[firstindex(X) + i])
    end
    return Y
end


"""
    MultiThreadedExec(; ntasks = Threads.nthreads())

Execute work in at most `ntasks` Julia tasks. The default uses the number of
Julia worker threads. Task count does not set the number of sampler candidates.
"""
struct MultiThreadedExec <: BATExecutor
    ntasks::Int

    function MultiThreadedExec(; ntasks::Integer = Threads.nthreads())
        @argcheck ntasks > 0
        return new(ntasks)
    end
end

function exec_map!(f::F, executor::MultiThreadedExec, Y::AbstractVector, X::AbstractVector) where {F<:Base.Callable}
    @argcheck length(X) == length(Y)
    n = length(Y)
    ntasks = min(executor.ntasks, n)
    ntasks <= 1 && return exec_map!(f, SequentialExec(), Y, X)
    @sync for task in 1:ntasks
        Threads.@spawn for i in fld((task - 1) * n, ntasks):(fld(task * n, ntasks) - 1)
            Y[firstindex(Y) + i] = f(X[firstindex(X) + i])
        end
    end
    return Y
end


@with_kw struct DistributedExec{WP<:AbstractWorkerPool} <: BATExecutor
    workers::WP = WorkerPool(workers())
    batchsize::Int = 1
end

function exec_map!(f::Base.Callable, executor::DistributedExec, Y::AbstractVector, X::AbstractVector)
    Y .= pmap(f, executor.workers, X, distributed = true, batch_size = executor.batchsize)
    return Y
end
