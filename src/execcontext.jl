# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct ExecContext
    multithreaded::Bool
    onprocs::StepRange{Int,Int}
end

ExecContext() = ExecContext(false, myid():1:myid())


struct ExecCompat
    multithreading::Bool
    max_procs::Int # Value of zero indicates that execution should happen on current process
end

ExecCompat() = ExecCompat(false, 0)
