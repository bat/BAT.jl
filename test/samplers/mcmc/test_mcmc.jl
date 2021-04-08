# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "mcmc" begin
    include("test_proposaldist.jl")
    include("test_mcmc_sample.jl")
    include("test_mcmc_convergence.jl")
    include("test_mh.jl")
    include("test_hmc.jl")
    include("test_mcmc_sampleid.jl")
end
