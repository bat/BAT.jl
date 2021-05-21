# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "integration" begin
    include("test_forwarddiff.jl")
    include("test_zygote.jl")
end
