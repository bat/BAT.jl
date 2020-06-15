# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "statistics" begin
    include("test_onlineuvstats.jl")
    include("test_onlinemvstats.jl")
    include("test_autocor.jl")
    include("test_effective_sample_size.jl")
    include("test_whiten.jl")
end
