# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "algotypes" begin
    include("test_bat_context.jl")
    include("test_makie_visualization_config.jl")
end
