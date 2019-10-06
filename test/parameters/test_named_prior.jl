# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, PDMats, ValueShapes, IntervalSets

@testset "named_prior" begin
    prior = @inferred NamedPrior(a = 5, b = Normal(), c = -4..5, d = MvNormal([1.2 0.5; 0.5 2.1]), e = [Normal(1.1, 0.2)] )

    @test typeof(@inferred VarShapes(prior)) <: VarShapes

    varshapes = VarShapes(prior)

    @test (@inferred logpdf(prior, [0.2, -0.4, 0.3, -0.5, 0.9])) == logpdf(Normal(), 0.2) + logpdf(Uniform(-4, 5), -0.4) + logpdf(MvNormal([1.2 0.5; 0.5 2.1]), [0.3, -0.5]) + logpdf(Normal(1.1, 0.2), 0.9)

    @test (@inferred logpdf(prior, varshapes([0.2, -0.4, 0.3, -0.5, 0.9]))) == logpdf(Normal(), 0.2) + logpdf(Uniform(-4, 5), -0.4) + logpdf(MvNormal([1.2 0.5; 0.5 2.1]), [0.3, -0.5]) + logpdf(Normal(1.1, 0.2), 0.9)

    @test all([rand(prior) in param_bounds(prior) for i in 1:10^4])

    @test (@inferred cov(prior)) ≈
        [1.0  0.0   0.0  0.0 0.0;
         0.0  6.75  0.0  0.0 0.0;
         0.0  0.0   1.2  0.5 0.0;
         0.0  0.0   0.5  2.1 0.0; 
         0.0  0.0   0.0  0.0 0.04 ]

    # ToDo: Add more tests
end
