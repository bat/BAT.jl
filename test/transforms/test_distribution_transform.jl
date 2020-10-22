# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using ValueShapes, Distributions, ForwardDiff

@testset "test_distribution_transform" begin
    #=
    using Cuba
    function integrate_over_unit(density::AbstractDensity)
        vs = varshape(density)
        f_cuba(source_x, y) = y[1] = exp(logvalof(density)(vs(source_x)))
        Cuba.vegas(f_cuba, 1, 1).integral[1]
    end
    =#
end
