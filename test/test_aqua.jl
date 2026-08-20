# This file is a part of BAT.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import BAT

Test.@testset "Package ambiguities" begin
    Test.@test isempty(Test.detect_ambiguities(BAT))
end # testset

Test.@testset "Aqua tests" begin
    Aqua.test_all(
        BAT,
        ambiguities = false,
        unbound_args = false,
        # ToDo: Re-enable once ScopedSettings v0.2 is registered. The
        # persistent-tasks check resolves BAT's dependencies from the
        # registries in a fresh project, which fails while a required
        # version is unregistered:
        persistent_tasks = false
    )
end # testset
