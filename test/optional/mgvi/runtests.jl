# This file is a part of BAT.jl, licensed under the MIT License (MIT).

import Test

# Keep MGVI separate while SparseColumnPivotedQR fails on 32-bit Julia.
Test.@testset "BAT with MGVI" begin
    include("test_mgvi.jl")
end
