# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "utils" begin
    include("test_array_utils.jl")
    include("test_coord_utils.jl")
end
