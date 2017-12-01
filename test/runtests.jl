# This file is a part of BAT.jl, licensed under the MIT License (MIT).

import Compat.Test
Test.@testset "Package BAT" begin
    include("mcmc_rand.jl")
    include("density.jl")
    include("distributions.jl")
    include("spatialvolume.jl")
    include("parambounds.jl")
    include("onlinemvstats.jl")
    include("onlineuvstats.jl")
    include("tjelmeland.jl")
end
