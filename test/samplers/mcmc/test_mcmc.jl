# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "mcmc" begin
    include("test_proposaldist.jl")
    include("test_mcmc_rand.jl")
    include("algorithms/test_algorithms.jl")
end
