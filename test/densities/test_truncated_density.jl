# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using ArraysOfArrays, Distributions, StatsBase, IntervalSets, ValueShapes

@testset "truncated_density" begin
    prior_dist = NamedTupleDist(
        a = truncated(Normal(), -2, 2),
        b = Exponential(),
        c = [1 2; 3 4],
        d = [-3..3, -4..4]
    )
    prior = convert(AbstractDensity, prior_dist)

    likelihood = LogDVal(0)

    posterior = PosteriorDensity(likelihood, prior)

    intervals = [-1..3, 1..4, -2..2, 0..4]
    bounds = BAT.HyperRectBounds(intervals, BAT.hard_bounds)

    @test (@inferred BAT.truncate_dist_hard(Exponential(), -1..5)).dist isa Truncated{<:Exponential}

    let
        orig_dist = Exponential()
        trunc_dist, logscalecorr = @inferred BAT.truncate_dist_hard(orig_dist, -6..6)
        @test trunc_dist.lower == 0 && trunc_dist.upper == 6
        @test logpdf(trunc_dist, 3) + logscalecorr ≈ logpdf(orig_dist, 3)
    end

    let
        orig_dist = truncated(Exponential(), 1, 5)
        trunc_dist, logscalecorr = @inferred BAT.truncate_dist_hard(orig_dist, 0..4)
        @test trunc_dist.lower == 1 && trunc_dist.upper == 4
        @test logpdf(trunc_dist, 3) + logscalecorr ≈ logpdf(orig_dist, 3)
    end

    let
        orig_dist = truncated(Exponential(), 1, 5)
        trunc_dist, logscalecorr = @inferred BAT.truncate_dist_hard(orig_dist, 2..6)
        @test trunc_dist.lower == 2 && trunc_dist.upper == 5
        @test logpdf(trunc_dist, 3) + logscalecorr ≈ logpdf(orig_dist, 3)
    end
    
    @test @inferred(BAT.truncate_dist_hard(prior_dist, intervals)).dist isa NamedTupleDist
    let
        trunc_dist, logscalecorr = @inferred BAT.truncate_dist_hard(prior_dist, intervals)
        @test logpdf(unshaped(trunc_dist), [1, 2, 0, 3]) + logscalecorr ≈ logpdf(unshaped(prior_dist), [1, 2, 0, 3])
    end

    @test @inferred(BAT.truncate_density(prior, bounds)) isa BAT.TruncatedDensity


    @test BAT.eval_logval(BAT.truncate_density(prior, bounds), [1, 2, 0, 3]) ≈ BAT.eval_logval(prior, [1, 2, 0, 3])
    @test varshape(BAT.truncate_density(prior, bounds)) == varshape(prior)

    @test @inferred(BAT.truncate_density(posterior, bounds)) isa PosteriorDensity

    trunc_pstr = BAT.truncate_density(posterior, bounds)
    @test @inferred(BAT.eval_logval(trunc_pstr, [1, 2, 0, 3])) ≈ BAT.eval_logval(posterior, [1, 2, 0, 3])
    @test @inferred(BAT.eval_logval(trunc_pstr, [-1, -1, -1, -1])) ≈ -Inf
    @test @inferred(BAT.eval_gradlogval(trunc_pstr, [1, 2, 0, 3])).grad_logd ≈ BAT.eval_gradlogval(posterior, [1, 2, 0, 3]).grad_logd
    @test varshape(trunc_pstr) == varshape(posterior)

    let
        trunc_prior_dist = parent(BAT.getprior(trunc_pstr)).dist
        s = bat_sample(trunc_pstr, 10^5, MCMCSampling()).result
        s_flat = flatview(unshaped.(s))
        @test all(minimum.(intervals) .< minimum(s_flat))
        @test all(maximum.(intervals) .> maximum(s_flat))
        cov_est = cov(unshaped.(s))
        @test isapprox(cov_est[1,1], var(trunc_prior_dist.a), rtol = 0.05)
        @test isapprox(cov_est[4,4], var(trunc_prior_dist.d.v[2]), rtol = 0.05)      
    end

    @test @inferred(BAT.truncate_density(BAT.ConstDensity(LogDVal(0)), bounds)) == BAT.ConstDensity(LogDVal(0), bounds)
end
