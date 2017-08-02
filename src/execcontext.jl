# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct ExecContext
    multithreaded::Bool
    onprocs::StepRange{Int,Int}
end

ExecContext() = ExecContext(false, myid():1:myid())
