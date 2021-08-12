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


struct MultiThreadedExec <: BATExecutor end

function exec_map!(f::Base.Callable, executor::MultiThreadedExec, Y::AbstractVector, X::AbstractVector)
    @argcheck length(eachindex(X)) == length(eachindex(X))
    @threads for i in 0:(length(eachindex(Y)) - 1)
        Y[firstindex(Y) + i] = f(X[firstindex(X) + i])
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
