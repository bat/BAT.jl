# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "optimization" begin
    include("test_mode_estimators.jl")
    include("test_binned_mode_estimator.jl")
end
