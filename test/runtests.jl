# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "Package BAT" begin
    include("utils/test_utils.jl")
    include("rngs/test_rngs.jl")
    include("distributions/test_distributions.jl")
    include("parameters/test_parameters.jl")
    include("densities/test_densities.jl")
    include("statistics/test_statistics.jl")
    include("optimization/test_optimization.jl")
    include("samplers/test_samplers.jl")
    include("io/test_io.jl")
    include("plotting/test_plotting.jl")
end
