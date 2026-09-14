# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using IntervalSets
using StatsBase
using Test

@testset "empirical credible intervals" begin
    values = [-10, -9, 9, 10]
    weights = [200, 1, 1, 200]

    @test BAT.smallest_credible_intervals(values, Weights(weights)) == [
        ClosedInterval(-10, -10), ClosedInterval(10, 10)
    ]
    @test only(BAT.smallest_credible_intervals(values, Weights(weights); mode = :connected)) ==
        ClosedInterval(-10, 10)
    @test only(BAT.smallest_credible_intervals(0:3; p = 0.5, mode = :connected)) == ClosedInterval(0, 1)
    @test only(BAT.smallest_credible_intervals([0, 1], Weights([6827, 3173]);
        nsigma_equivalent = 1, mode = :connected)) == ClosedInterval(0, 0)
    @test only(BAT.smallest_credible_intervals([0, 1], Weights([369, 1]);
        nsigma_equivalent = 3, mode = :connected)) == ClosedInterval(0, 1)

    reference = BAT.smallest_credible_intervals(values, Weights(weights))
    @test BAT.smallest_credible_intervals(reverse(values), Weights(reverse(weights))) == reference
    @test BAT.smallest_credible_intervals(values, Weights(7 .* weights)) == reference

    logweights = exp.(BAT.ULogarithmic, [-1.0, 0.0])
    @test only(BAT.smallest_credible_intervals([0, 1], Weights(logweights); mode = :connected)) ==
        ClosedInterval(1, 1)
    logweights = exp.(BAT.ULogarithmic, [zeros(10); -1000.0])
    @test only(BAT.smallest_credible_intervals([zeros(Int, 9); 1; 2], Weights(logweights);
        p = 9//10, mode = :connected)) == ClosedInterval(0, 1)

    weights = [BAT.Double64(9), BAT.Double64(1.0, 2.0^-1000)]
    @test only(BAT.smallest_credible_intervals([0, 1], Weights(weights);
        p = 9//10, mode = :connected)) == ClosedInterval(0, 1)
    values = [BAT.Double64(0), BAT.Double64(1.0, 2.0^-1000), BAT.Double64(3), BAT.Double64(4)]
    @test only(BAT.smallest_credible_intervals(values; mode = :connected)) ==
        ClosedInterval(values[2], values[4])
end
