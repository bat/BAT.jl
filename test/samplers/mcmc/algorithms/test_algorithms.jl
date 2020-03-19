# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "algorithms" begin
    include("test_ahmc.jl")
end


#StaticTrajectory(int, n_steps)
