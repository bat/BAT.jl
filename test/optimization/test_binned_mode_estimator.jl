using BAT
using Test

@test BAT.bat_marginalmode(
    BAT.DensitySampleVector([[1.0], [2.0], [3.0], [4.0], [5.0]], zeros(5)),
    BAT.BinnedModeEstimator(binning = BAT.FixedNBins(nbins = 2)),
    BAT.BATContext(),
).result == [3.0]
