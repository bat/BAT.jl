using BAT
using Test
using Statistics

@testset "Funnel Distribution" begin
    #Tests different forms of instantiate Funnel Distribution
    @test @inferred(BAT.FunnelDistribution(1., 2., 3)) isa BAT.FunnelDistribution
    @test @inferred(BAT.FunnelDistribution()) isa BAT.FunnelDistribution
    @test @inferred(BAT.FunnelDistribution(a = 1.0, b = 0.5, n = 3)) isa BAT.FunnelDistribution

    funnel = BAT.FunnelDistribution(a = Float32(1.0), b = Float32(0.5), n = Int32(3))

    #Check Parameters
    @test @inferred(StatsBase.params(funnel)) == (Float32(1.0), Float32(0.5), Int32(3))

    @test @inferred(Base.length(funnel)) == 3 #Number of dimensions
    @test @inferred(Base.eltype(funnel)) == Float32 #Element Type

    #Check mean and covariance
    @test @inferred(Statistics.mean(funnel)) == [0.0, 0.0, 0.0] #Check mean
    @test isapprox(@inferred(Statistics.cov(funnel)), [1.00256 -0.0124585 -0.00373376; -0.0124585 7.04822 -0.165097;  -0.00373376  -0.165097  7.1126])

    #logpdf
    @test isapprox(@inferred(Distributions._logpdf(funnel, [0., 0., 0.])), -2.75681, atol = 1e-5)

    #How to call ._rand!? Tried Distributions._rand!(BAT.bat_rng(), funnel, []) but didnt work
end