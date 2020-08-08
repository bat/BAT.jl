# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "distributions" begin
    include("test_ahmi_integration.jl")
    include("test_cuba_integration.jl")
end
