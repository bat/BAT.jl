# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "transforms" begin
    include("test_distribution_transform.jl")
    include("test_trafo_utils.jl")
end
