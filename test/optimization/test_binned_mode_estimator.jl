using BAT
using Test

@test BAT.bat_marginalmode(
    BAT.DensitySampleVector([[1.0], [2.0], [3.0], [4.0], [5.0]], zeros(5)),
    BAT.BinnedModeEstimator(binning = BAT.FixedNBins(nbins = 2)),
    BAT.BATContext(),
).result == [3.0]

using Distributions, ValueShapes

let prior = distprod(x = (a = Uniform(0, 1), b = (c = Uniform(2, 3), d = MvNormal([1.0 0.0; 0.0 1.0]))), e = 7.0)
    samples = bat_sample(prior, IIDSampling(nsamples = 200)).result
    mode = BAT.bat_marginalmode(samples, BAT.BinnedModeEstimator(binning = BAT.FixedNBins(nbins = 4)), BAT.BATContext()).result
    @test keys(mode) == (:x, :e) && keys(mode.x) == (:a, :b) && keys(mode.x.b) == (:c, :d)
    @test mode.e == 7.0
    @test isfinite(logpdf(prior, mode))
end
