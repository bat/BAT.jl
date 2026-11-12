# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type BATExecutor end


default_executor() = MultiThreadedExec()


function exec_map(f::Base.Callable, executor::BATExecutor, X::AbstractVector)
    T = Base.promote_op(f, eltype(X))
    # Let map widen storage when the result type depends on the input value.
    isconcretetype(T) || return map(f, X)
    return exec_map!(f, executor, Vector{T}(undef, length(X)), X)
end


struct SequentialExec <: BATExecutor end

exec_map(f::Base.Callable, ::SequentialExec, X::AbstractVector) = map(f, X)

function exec_map!(f::Base.Callable, executor::SequentialExec, Y::AbstractVector, X::AbstractVector)
    @argcheck length(X) == length(Y) throw(ArgumentError("Input and output arrays must have equal lengths."))
    for i in 0:(length(Y) - 1)
        Y[firstindex(Y) + i] = f(X[firstindex(X) + i])
    end
    return Y
end


struct MultiThreadedExec <: BATExecutor end

function exec_map!(f::Base.Callable, executor::MultiThreadedExec, Y::AbstractVector, X::AbstractVector)
    @argcheck length(X) == length(Y) throw(ArgumentError("Input and output arrays must have equal lengths."))
    @threads for i in 0:(length(Y) - 1)
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
