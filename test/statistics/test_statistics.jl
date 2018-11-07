# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "statistics" begin
    include("test_onlineuvstats.jl")
    include("test_onlinemvstats.jl")
end
