# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "samplers" begin
    include("test_bat_sample.jl")
    include("mcmc/test_mcmc.jl")
end
