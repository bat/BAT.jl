# This file is a part of BAT.jl, licensed under the MIT License (MIT).

import Compat.Test
Test.@testset "Package BAT" begin
    include("mcmc_rand.jl")
    include("target_distribution.jl")
    include("target_density.jl")
    include("distributions.jl")
    include("spatialvolume.jl")    
    include("parambounds.jl")
    include("onlinemvstats.jl")
    include("onlineuvstats.jl")
end
