# This file is a part of BAT.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import BAT
import MeasureBase

Test.@testset "Package ambiguities" begin
    Test.@test isempty(Test.detect_ambiguities(BAT))
end # testset

# BAT connects MeasureBase's measures to ValueShapes and to the statistics
# functions, which is type piracy until that support moves into a
# ValueShapes extension of MeasureBase (see src/measures/measure_shapes.jl):
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
