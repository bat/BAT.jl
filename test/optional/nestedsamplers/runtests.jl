# This file is a part of BAT.jl, licensed under the MIT License (MIT).

import Test
import Random

Random.seed!(0x424154)

Test.@testset "BAT with NestedSamplers" begin
    include("test_ellipsoidal_nested_sampling.jl")
end
