# This file is a part of BAT.jl, licensed under the MIT License (MIT).

@Base.Test.testset "Package BAT" begin
    include.([
        "rand.jl",
        "parambounds.jl",
    ])
end
