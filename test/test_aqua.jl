# This file is a part of BAT.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import BAT

# ToDo: Fix ambiguities and enable ambiguity testing:
#=
Test.@testset "Package ambiguities" begin
    Test.@test isempty(Test.detect_ambiguities(BAT))
end # testset
=#

Test.@testset "Aqua tests" begin
    Aqua.test_all(
        BAT,
        ambiguities = false,
        unbound_args = false,
        project_toml_formatting = VERSION≥v"1.7"
    )
end # testset
