# This file is a part of BAT.jl, licensed under the MIT License (MIT).

import Compat.Test
Test.@testset "Package BAT" begin
    include("mcmc_rand.jl")
    include("data_vector.jl")
    include("rng.jl")
    include("density.jl")
    include("distributions.jl")
    include("proposaldist.jl")
    include("spatialvolume.jl")
    include("parambounds.jl")
    include("onlinemvstats.jl")
    include("onlineuvstats.jl")
    include("const_density.jl")
    include("density_product.jl")
    include("mvdist_density.jl")
    include("density_sample.jl")        
end
