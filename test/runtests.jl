# This file is a part of BAT.jl, licensed under the MIT License (MIT).

@Base.Test.testset "Package BAT" begin
    include.([
        "distributions.jl",
        "parambounds.jl",
        "onlinemvstats.jl",
    ])
end
