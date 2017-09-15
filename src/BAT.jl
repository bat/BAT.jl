# This file is a part of BAT.jl, licensed under the MIT License (MIT).

__precompile__(true)

module BAT

include.([
    "shims.jl",
    "rng.jl",
    "distributions.jl",
    "util.jl",
    "extendablearray.jl",
    "execcontext.jl",
    "onlineuvstats.jl",
    "onlinemvstats.jl",
    "parambounds.jl",
    "proposaldist.jl",
    "targetdensity.jl",
    "targetsubject.jl",
    "mcmc.jl",
    "mhsampler.jl",
])

end # module
