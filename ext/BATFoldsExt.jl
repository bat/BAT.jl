# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATFoldsExt

using Folds, Transducers

using BAT


function BAT.exec_map!(f::Base.Callable, executor::Transducers.Executor, Y::AbstractVector, X::AbstractVector)
    Folds.map!(f, Y, X, executor)
    return Y
end


function BAT.exec_map!(f::Base.Callable, executor::Transducers.DistributedEx, Y::AbstractVector, X::AbstractVector)
    Y .= Folds.map(f, X, executor)
    return Y
end

end # module BATFoldsExt
