# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "algorithms" begin
    include("test_mh.jl")
    include("test_hmc.jl")
end
