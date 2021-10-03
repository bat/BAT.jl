# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "densities" begin
    include("test_logdval.jl")
    include("test_abstract_density.jl")
    include("test_generic_density.jl")
    include("test_distribution_density.jl")
    include("test_parameter_mapped_density.jl")
    include("test_renormalize_density.jl")
    include("test_truncate_density.jl")
    include("test_transformed_density.jl")
    include("test_external_density.jl")
end
