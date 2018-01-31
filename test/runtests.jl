# This file is a part of BAT.jl, licensed under the MIT License (MIT).

import Compat.Test
Test.@testset "Package BAT" begin
    include("shims.jl")
    include("rng.jl")
    include("distributions.jl")
    include("util.jl")
    include("onlineuvstats.jl")
    include("onlinemvstats.jl")
    include("spatialvolume.jl")
    include("parambounds.jl")
    include("proposaldist.jl")
    include("density.jl")
    include("const_density.jl")
    include("density_product.jl")
    include("mvdist_density.jl")
    include("data_vector.jl")
    include("density_sample.jl")        
    include("mcmc_rand.jl")
    # include("tjelmeland.jl")
end
