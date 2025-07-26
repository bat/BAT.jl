# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "utils" begin
    include("test_error_log.jl")
    include("test_util_functions.jl")
    include("test_array_utils.jl")
end
