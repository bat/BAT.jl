# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test


@testset "DEMove" begin
    check_ensemble_gaussian_moments(DEMove(); seed = 26201)
end
