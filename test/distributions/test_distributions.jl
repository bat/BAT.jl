# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "distributions" begin
    include("test_distribution_functions.jl")
    include("test_bat_sampler.jl")
    include("test_HistogramAsUvDistribution.jl")
    include("test_HistogramAsMvDistribution.jl")
end
