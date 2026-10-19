# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "plotting" begin
    include("test_MarginalDist.jl")
    include("test_histogram_weights.jl")
    include("test_histogram_weight_regions.jl")
    include("test_histogram_regions.jl")
end
