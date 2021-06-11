using BAT
using Test
using Statistics
using StatsBase
using Distributions

@testset "Gaussian Shell Distribution" begin
    #Tests different ways of instantiate GaussianShell Distribution
    @test @inferred(BAT.GaussianShell(r = 3., w = 0.1, n = 3)) isa BAT.GaussianShell
    @test @inferred(BAT.GaussianShell()) isa BAT.GaussianShell

    gs = BAT.GaussianShell(r = 3., w = 0.1, n = 3)

    @test @inferred(Base.length(gs)) == 3
    @test @inferred(Base.eltype(gs)) == Float64

    @test @inferred(StatsBase.params(gs)) == (3.0, 0.1, [0.0, 0.0, 0.0])

    @test isapprox(@inferred(Statistics.cov(gs)), [3.02718 0.00166016 0.0113455; 
    0.00166016 3.01293 0.00453035; 0.0113455   0.00453035  3.01046], atol = 1e-5)

    @test isapprox(@inferred(Distributions._logpdf(gs, [0., 0., 0.])), -453.34571, atol = 1e-5)

end