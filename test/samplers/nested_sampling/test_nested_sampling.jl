# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

import Pkg

Test.@testset "nested_sampling" begin
    # Python ultranest package doesn't seem to be available via Conda on
    # 32-bit systems, GLIBCXX version conflicts cause trouble for CI
    # on older Julia versions:
    if Int == Int64 && ("UltraNest" in keys(Pkg.project().dependencies))
        # include("test_ultranest.jl")
    end
    if ("NestedSamplers" in keys(Pkg.project().dependencies))
        include("test_ellipsoidal_nested_sampling.jl")
    end
end
