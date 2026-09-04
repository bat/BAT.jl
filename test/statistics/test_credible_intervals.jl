# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using IntervalSets
using StatsBase
using Test

@testset "empirical credible intervals" begin
    values = [-10, -9, 9, 10]
    weights = [20, 1, 1, 20]

    @test BAT.smallest_credible_intervals(values, Weights(weights)) == [
        ClosedInterval(-10, -10), ClosedInterval(10, 10)
    ]
    @test only(BAT.smallest_credible_intervals(values, Weights(weights); mode = :connected)) ==
        ClosedInterval(-10, 10)

    reference = BAT.smallest_credible_intervals(values, Weights(weights))
    @test BAT.smallest_credible_intervals(reverse(values), Weights(reverse(weights))) == reference
    @test BAT.smallest_credible_intervals(values, Weights(7 .* weights)) == reference

    logweights = exp.(BAT.ULogarithmic, [-1000.0, 0.0])
    @test only(BAT.smallest_credible_intervals([0, 1], Weights(logweights); mode = :connected)) ==
        ClosedInterval(1, 1)
end
