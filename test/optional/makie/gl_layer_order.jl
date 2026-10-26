# This file is a part of BAT.jl, licensed under the MIT License (MIT).
# Run separately with an external deadline and an actual native GL display.
using BAT, Makie, GLMakie, Test
@test realpath(pkgdir(BAT)) == realpath(joinpath(@__DIR__, "../../.."))
GLMakie.activate!()
Makie.inline!(false)
include("../../plotting/test_makie_layer_order.jl")
