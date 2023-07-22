# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATFoldsExt

@static if isdefined(Base, :get_extension)
    using Folds, Transducers
else
    using ..Folds, ..Transducers
end

using BAT
BAT.pkgext(::Val{:Folds}) = BAT.PackageExtension{:Folds}()


function BAT.exec_map!(f::Base.Callable, executor::Transducers.Executor, Y::AbstractVector, X::AbstractVector)
    Folds.map!(f, Y, X, executor)
    return Y
end


function BAT.exec_map!(f::Base.Callable, executor::Transducers.DistributedEx, Y::AbstractVector, X::AbstractVector)
    Y .= Folds.map(f, X, executor)
    return Y
end

end # module BATFoldsExt
