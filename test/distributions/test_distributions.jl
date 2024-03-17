# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "distributions" begin
    include("test_distribution_functions.jl")
    include("test_standard_uniform.jl")
    include("test_standard_normal.jl")
    include("test_hierarchical_distribution.jl")
end
