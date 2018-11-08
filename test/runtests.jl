# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "Package BAT" begin
    include("Logging/test_Logging.jl")

    include("utils/test_utils.jl")
    include("rngs/test_rngs.jl")
    include("distributions/test_distributions.jl")
    include("scheduling/test_scheduling.jl")
    include("parameters/test_parameters.jl")
    include("statistics/test_statistics.jl")
    include("densities/test_densities.jl")
    include("samplers/test_samplers.jl")
    include("plotting/test_plotting.jl")
end
