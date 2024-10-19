# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using DensityInterface, ValueShapes
using ArraysOfArrays, Distributions, StatsBase, IntervalSets

@testset "truncate_batmeasure" begin
    prior_dist = unshaped(NamedTupleDist(
        a = truncated(Normal(), -2, 2),
        b = Exponential(),
        c = [1 2; 3 4],
        d = [-3..3, -4..4]
    ))
    prior = convert(AbstractMeasureOrDensity, prior_dist)

    likelihood = v -> (logval = 0,)

    posterior = PosteriorMeasure(likelihood, prior)

    bounds = [-1..3, 1..4, -2..2, 0..4]

    @test (@inferred BAT.truncate_dist_hard(Exponential(), -1..5)).dist isa Truncated{<:Exponential}

    let
        orig_dist = Exponential()
        trunc_dist, logweight = @inferred BAT.truncate_dist_hard(orig_dist, -6..6)
        @test trunc_dist.lower == 0 && trunc_dist.upper == 6
        @test logpdf(trunc_dist, 3) + logweight ≈ logpdf(orig_dist, 3)
    end

    let
        orig_dist = truncated(Exponential(), 1, 5)
        trunc_dist, logweight = @inferred BAT.truncate_dist_hard(orig_dist, 0..4)
        @test trunc_dist.lower == 1 && trunc_dist.upper == 4
        @test logpdf(trunc_dist, 3) + logweight ≈ logpdf(orig_dist, 3)
    end

    let
        orig_dist = truncated(Exponential(), 1, 5)
        trunc_dist, logweight = @inferred BAT.truncate_dist_hard(orig_dist, 2..6)
        @test trunc_dist.lower == 2 && trunc_dist.upper == 5
        @test logpdf(trunc_dist, 3) + logweight ≈ logpdf(orig_dist, 3)
    end
    
    @test @inferred(BAT.truncate_dist_hard(prior_dist, bounds)).dist isa ValueShapes.UnshapedNTD
    let
        trunc_dist, logweight = @inferred BAT.truncate_dist_hard(prior_dist, bounds)
        @test logpdf(unshaped(trunc_dist), [1, 2, 0, 3]) + logweight ≈ logpdf(unshaped(prior_dist), [1, 2, 0, 3])
    end

    @test @inferred(truncate_density(prior, bounds)) isa BAT.BATWeightedMeasure


    @test BAT.checked_logdensityof(unshaped(truncate_density(prior, bounds)), [1, 2, 0, 3]) ≈ BAT.checked_logdensityof(unshaped(prior), [1, 2, 0, 3])
    @test BAT.checked_logdensityof(truncate_density(prior, bounds), varshape(prior)([1, 2, 0, 3])) ≈ BAT.checked_logdensityof(prior, varshape(prior)([1, 2, 0, 3]))
    @test varshape(truncate_density(prior, bounds)) == varshape(prior)

    @test @inferred(truncate_density(posterior, bounds)) isa PosteriorMeasure

    trunc_pstr = truncate_density(posterior, bounds)
    @test @inferred(BAT.checked_logdensityof(unshaped(trunc_pstr), [1, 2, 0, 3])) ≈ BAT.checked_logdensityof(unshaped(posterior), [1, 2, 0, 3])
    @test @inferred(BAT.checked_logdensityof(unshaped(trunc_pstr), [-1, -1, -1, -1])) ≈ -Inf
    @test varshape(trunc_pstr) == varshape(posterior)

    let
        trunc_prior_dist = parent(BAT.getprior(trunc_pstr)).dist
        s = bat_sample(trunc_pstr, MCMCSampling(mcalg = RandomWalk(), trafo = DoNotTransform(), nsteps = 10^5)).result
        s_flat = flatview(unshaped.(s))
        @test all(minimum.(bounds) .< minimum(s_flat))
        @test all(maximum.(bounds) .> maximum(s_flat))
        cov_est = cov(unshaped.(s))
        @test isapprox(cov_est[1,1], var(trunc_prior_dist.shaped.a), rtol = 0.05)
        @test isapprox(cov_est[4,4], var(trunc_prior_dist.shaped.d.v[2]), rtol = 0.05)      
    end
end
