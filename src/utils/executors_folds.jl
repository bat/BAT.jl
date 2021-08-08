# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function exec_map!(f::Base.Callable, executor::Transducers.Executor, Y::AbstractVector, X::AbstractVector)
    Folds.map!(f, Y, X, executor)
    return Y
end


function exec_map!(f::Base.Callable, executor::Transducers.DistributedEx, Y::AbstractVector, X::AbstractVector)
    Y .= Folds.map(f, X, executor)
    return Y
end
