# This file is a part of BAT.jl, licensed under the MIT License (MIT).

import Test
import Random

Random.seed!(0x424154)

Test.@testset "BAT with UltraNest" begin
    include("test_ultranest.jl")
end
