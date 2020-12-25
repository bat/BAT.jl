# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "nested_sampling" begin
    # Python ultranest package doesn't seem to be available via Conda on
    # 32-bit systems:
    if Int == Int64
        include("test_ultranest.jl")
    end
end
