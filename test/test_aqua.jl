# This file is a part of BAT.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import BAT
import MeasureBase

Test.@testset "Package ambiguities" begin
    Test.@test isempty(Test.detect_ambiguities(BAT))
end # testset

# BAT defines moments and modes of MeasureBase measures and transports of
# shaped sample arrays, which is type piracy until MeasureBase provides
# the former and the sample storage is reworked for the latter:
const mb_types_treated_as_own = [
    MeasureBase.AbstractMeasure,
    MeasureBase.AbstractProductMeasure,
    MeasureBase.AsMeasure,
    MeasureBase.Dirac,
    MeasureBase.WeightedMeasure,
    MeasureBase.TransportFunction,
]

Test.@testset "Aqua tests" begin
    Aqua.test_all(
        BAT,
        ambiguities = false,
        unbound_args = false,
        piracies = (treat_as_own = mb_types_treated_as_own,)
    )
end # testset
