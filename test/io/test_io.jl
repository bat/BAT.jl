# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "io" begin
    include("test_hdf5.jl")
    include("test_bat_io.jl")
end
