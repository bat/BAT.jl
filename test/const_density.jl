# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Compat.Test

@testset "const_density" begin
    @testset "ConstDensity" begin
        cd = @inferred ConstDensity(
            BAT.HyperRectBounds([-1., 0.5], [2.,1], BAT.hard_bounds),
            one)

    end
end
