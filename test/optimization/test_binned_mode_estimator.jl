using BAT
using Test

@testset "BinnedModeEstimator" begin
    @testset "FixedNBins" begin
        samples = BAT.DensitySampleVector(
            [[1.0], [2.0], [3.0], [4.0], [5.0]],
            zeros(5),
        )

        result = BAT.bat_marginalmode(
            samples,
            BAT.BinnedModeEstimator(binning = BAT.FixedNBins(nbins = 2)),
            BAT.BATContext(),
        )

        @test result.result == [3.0]
    end
end
