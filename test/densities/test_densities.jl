# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "densities" begin
    include("test_density.jl")
    include("test_const_density.jl")
    include("test_density_product.jl")
    include("test_mvdist_density.jl")
    include("test_parameter_mapped_density.jl")
    include("test_external_density.jl")
end
