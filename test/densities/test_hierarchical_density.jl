# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, StatsBase, IntervalSets, ValueShapes, ArraysOfArrays

@testset "hierarchial_density" begin
    let
        parent_density = NamedTupleDist(
            foo = Exponential(3.5),
            bar = Normal(2.0, 1.0)
        )

        f = v -> NamedTupleDist(baz = fill(Normal(v.bar, v.foo), 3))

        @test typeof(@inferred HierarchicalDensity(f, parent_density)) <: HierarchicalDensity

        hd = HierarchicalDensity(f, parent_density)

        @test @inferred(sampler(hd)) == BAT.HierarchicalDensitySampler(hd)
        @test @inferred(rand(sampler(hd))) isa AbstractVector{<:Real}
        @test @inferred(varshape(hd)) == NamedTupleShape(foo = ScalarShape{Real}(), bar = ScalarShape{Real}(), baz = ArrayShape{Real}(3))

        @test @inferred(BAT.eval_logval_unchecked(hd, [2.7, 4.3, 8.7, 8.7, 8.7])) ≈ logpdf(parent_density.foo, 2.7) + logpdf(parent_density.bar, 4.3) + 3 * logpdf(Normal(4.3, 2.7), 8.7)
        @test @inferred(BAT.eval_logval_unchecked(hd, (foo = 2.7, bar = 4.3, baz = fill(8.7, 3)))) ≈ logpdf(parent_density.foo, 2.7) + logpdf(parent_density.bar, 4.3) + 3 * logpdf(Normal(4.3, 2.7), 8.7)

        @test @inferred(BAT.var_bounds(hd)) == BAT.HierarchicalDensityBounds(hd)

        hd_bounds = @inferred(BAT.var_bounds(hd))
        @test all(in.(nestedview(@inferred rand(sampler(hd), 10^3)), Ref(hd_bounds)))
        @test @inferred(fill(-1.0, totalndof(hd)) in hd_bounds) == false

        @test (@inferred(BAT.renormalize_variate!(fill(NaN, totalndof(hd)), hd_bounds, fill(-1.0, totalndof(hd)))) in hd_bounds) == false

        posterior = PosteriorDensity(LogDVal(0), hd)
        samples = bat_sample(posterior, 10^5, MCMCSampling(sampler = MetropolisHastings())).result
        isapprox(cov(unshaped.(samples)), cov(hd), rtol = 0.05)
    end

    let
        parent_density = BAT.DistributionDensity(NamedTupleDist(
            foo = 2..4,
            bar = Normal(2.0, 1.0)
        ), bounds_type = BAT.reflective_bounds)

        f = v -> BAT.DistributionDensity(NamedTupleDist(baz = fill(Normal(v.bar, v.foo), 3)), bounds_type = BAT.reflective_bounds)

        hd = HierarchicalDensity(f, parent_density)

        hd_bounds = @inferred(BAT.var_bounds(hd))
        @test @inferred(fill(1.5, totalndof(hd)) in hd_bounds) == false
        @test @inferred(BAT.renormalize_variate!(fill(NaN, totalndof(hd)), hd_bounds, fill(1.5, totalndof(hd)))) == [2.5, 1.5, 1.5, 1.5, 1.5]
        @test (@inferred(BAT.renormalize_variate!(fill(NaN, totalndof(hd)), hd_bounds, fill(1.5, totalndof(hd)))) in hd_bounds) == true
    end

    let
        hd = HierarchicalDensity(
            v -> NamedTupleDist(b = Normal(v.a, 1.2)),
            NamedTupleDist(a = Normal(2.3, 1.9))
        )

        cov_expected = [1.9^2 1.9^2; 1.9^2 1.9^2 + 1.2^2]

        @test isapprox(cov(hd), cov_expected, rtol = 0.05)
        @test isapprox(mean(nestedview(rand(sampler(hd), 10^5))), [2.3, 2.3], rtol = 0.05)
    end
end
