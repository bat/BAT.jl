# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "densities" begin
    include("test_logdval.jl")
    include("test_abstract_density.jl")
    include("test_joint_likelihood.jl")
end
