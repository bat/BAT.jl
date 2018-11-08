# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "parameters" begin
    include("test_spatialvolume.jl")
    include("test_parambounds.jl")
    include("test_data_vector.jl")
    include("test_density_sample.jl")
    include("test_parameter_mapping.jl")
end
