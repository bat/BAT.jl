# This file is a part of BAT.jl, licensed under the MIT License (MIT).

__precompile__(true)

module BAT

include.([
    "shims.jl",
    "rand.jl",
    "util.jl",
    "execcontext.jl",
    "onlinestats.jl",
    "parambounds.jl",
    "targetfunction.jl",
    "proposaldist.jl",
    "mhsampler.jl",
])

end # module
