# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct MCMCRetryInit <: MCMCInitAlgorithm

TODO

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MCMCRetryInit <: MCMCInitAlgorithm
    init_tries_per_chain::Int64 = 16
    nsteps_init::Int64 = 20
    initval_alg::InitvalAlgorithm = InitFromTarget()
    strict::Bool = true
end

export MCMCRetryInit




# Draw a random init point for each walker for each chain
# And let the chains run for nsteps_init steps and unviable walkers get a new random position and let their chains run until 
# Leave the chains with viable walkers as is 
# strict lets the init fail if nothing moves

