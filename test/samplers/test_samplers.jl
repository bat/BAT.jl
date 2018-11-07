# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "samplers" begin
    include("mcmc/test_mcmc.jl")
end
