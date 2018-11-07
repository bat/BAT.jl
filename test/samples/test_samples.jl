# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "samples" begin
    include("test_data_vector.jl")
    include("test_density_sample.jl")        
end
