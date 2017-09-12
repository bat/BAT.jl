# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct ExecContext
    multithreaded::Bool
    onprocs::StepRange{Int,Int}
end

export ExecContext

ExecContext() = ExecContext(false, myid():1:myid())


struct ExecCompat
    multithreading::Bool
    max_procs::Int # Value of zero indicates that execution should happen on current process
end

export ExecCompat

ExecCompat() = ExecCompat(false, 0)


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
