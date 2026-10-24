# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, StatsBase, ValueShapes, ArraysOfArrays, StructArrays
using Random123: Philox4x
import RecipesBase

using BAT: MarginalDist, flatbatch

# Plots.jl defines this, `apply_recipe` needs it for recipes with keyword
# arguments:
hasmethod(RecipesBase.is_key_supported, Tuple{Symbol}) || @eval RecipesBase.is_key_supported(::Symbol) = true

recipe_series(args...; kwargs...) = RecipesBase.apply_recipe(Dict{Symbol,Any}(kwargs), args...)
series_attr(series, key) = [get(s.plotattributes, key, nothing) for s in series]
with_label(series, label) = only(filter(s -> get(s.plotattributes, :label, nothing) == label, series))
marg_hist(marg::MarginalDist) = convert(Histogram, marg.dist)

@testset "recipes" begin
    prior = distprod(a = Normal(), b = [Normal(1, 1), Normal(2, 0.5)], c = 4.2, d = Exponential())
    samples = bat_sample(prior, IIDSampling(nsamples = 10^4), BATContext(rng = Philox4x((7, 8)))).result
    X = flatbatch(samples.v)

    @testset "samples 1D" begin
        series = recipe_series(samples, :(b[2]); mean = true, globalmode = true)
        marg = series[1].args[1]
        @test marg isa MarginalDist && isapprox(mean(marg.dist), 2, atol = 0.05)
        @test series[1].plotattributes[:xguide] == "b[2]" && series[1].plotattributes[:yguide] == "p(b[2])"
        @test first.(with_label(series, "mean").args[1]) ≈ fill(mean(samples).b[2], 2)
        @test first.(with_label(series, "global mode").args[1]) ≈ fill(mode(samples).b[2], 2)
        @test with_label(series, "marginal mode").args[1][1][1] ≈ only(only(BAT.find_marginalmodes(marg)))

        # Integer selectors are positions in the standard-variate layout:
        @test marg_hist(recipe_series(samples, 3)[1].args[1]) == marg_hist(marg)
        @test recipe_series(unshaped.(samples), 3)[1].plotattributes[:xguide] == "v_3"
        @test_throws ArgumentError recipe_series(samples, :b)
        @test_throws ArgumentError recipe_series(samples, :c)
    end

    @testset "samples 2D" begin
        series = recipe_series(samples, (:a, :(b[2])); seriestype = :marginal, mean = true, std = Dict(), globalmode = true)
        @test series[1].args[1] isa MarginalDist && series[1].args[2] == (1, 2)
        @test keys(varshape(series[1].args[1])) == (:a, :b⌞2⌟)
        @test all(==("a"), series_attr(series, :xguide)) && all(==("b[2]"), series_attr(series, :yguide))
        markers = series[2:end]
        @test all(==(3), series_attr(markers, :subplot))
        @test all(!isnothing, series_attr(markers, :markercolor))
        mean_series = with_label(series, "mean")
        @test collect(only.(mean_series.args)) ≈ [mean(samples).a, mean(samples).b[2]]
        @test mean_series.plotattributes[:xerror] ≈ std(samples.v.a)
        @test with_label(series, "global mode").plotattributes[:markeralpha] == 1

        @test series_attr(recipe_series(samples, (1, 4); mean = true)[2:end], :subplot) == [nothing, nothing]

        scatter = only(recipe_series(samples, (1, 4); seriestype = :scatter, marginalmode = false))
        @test scatter.args == (samples.v.a, samples.v.d)
    end

    @testset "samples overview" begin
        series = recipe_series(samples; vsel = [:(b[2]), :a], bins = (a = 10,))
        diagonal = series[[1, 4]]
        @test [s.args[2] for s in diagonal] == [3, 1]
        @test series_attr(diagonal, :xguide) == ["b[2]", "a"]
        @test series_attr(diagonal, :bins) == [200, 10]
        @test series[2].args[2] == (3, 1) && series[2].plotattributes[:bins] == (200, 10)

        # Positions beyond the free parameters are skipped:
        @test length(recipe_series(samples)) == 4^2
        @test series_attr(recipe_series(samples; vsel = (4, :a), bins = -2:0.5:2)[[1, 4]], :bins) == [-2:0.5:2, -2:0.5:2]
        @test [s.args[2] for s in recipe_series(samples; vsel = 2:3)[[1, 4]]] == [2, 3]
        @test_throws ArgumentError recipe_series(samples; vsel = [:a], bins = (b = 10,))
    end

    @testset "MarginalDist" begin
        marg = MarginalDist(samples, :a, bins = 20)
        hist = marg_hist(marg)
        @test only(recipe_series(marg; seriestype = :stephist)).args == (hist.edges[1][1:end-1], hist.weights)
        intervals = recipe_series(marg; seriestype = :smallest_intervals)
        @test length(intervals) == 4 && intervals[end].args == (hist.edges[1][1:end-1], hist.weights)

        marg_2d = MarginalDist(samples, (:d, :a), bins = 20)
        @test only(recipe_series(marg_2d, 2; seriestype = :stephist)).args[2] ≈ hist.weights

        # The panels of marginal plots show the selected parameters:
        marg_3d = MarginalDist(samples, [:a, :(b[1]), :d], bins = 20)
        panels = recipe_series(marg_3d, (3, 1); seriestype = :marginal, layout = 4)
        @test keys(varshape(panels[1].args[1])) == (:d, :a)
        @test [s.args[2] for s in panels[[1, 3, 4]]] == [1, (1, 2), 2]
    end

    @testset "MCMC diagnostics" begin
        info = StructArray(chainid = repeat(Int32[1, 2], inner = 50))
        mcmc_samples = DensitySampleVector(v = nestedview(randn(2, 100)), logd = zeros(100), info = info)
        series = recipe_series(BAT.MCMCDiagnostics(mcmc_samples); layout = 1)
        @test length(series) == 2 * 2 * 5
        trace = filter(s -> get(s.plotattributes, :title, nothing) == "Trace", series)
        @test [s.args[2] for s in trace] == [flatview(mcmc_samples.v)[p, r] for r in (1:50, 51:100) for p in 1:2]

        # Sample weights act like repeated samples in the kernel density estimates:
        kde_densities(smpls) = [s.args[2] for s in recipe_series(
            BAT.MCMCDiagnostics(smpls); layout = 1, description = false, diagnostics = [:kde],
            kde = Dict("bandwidth" => 0.3, "boundary" => (-6.0, 6.0))
        )]
        weights = repeat([1, 3], 50)
        weighted = DensitySampleVector(v = mcmc_samples.v, logd = mcmc_samples.logd, weight = weights, info = info)
        repeated = mcmc_samples[reduce(vcat, fill.(eachindex(weights), weights))]
        @test kde_densities(weighted) ≈ kde_densities(repeated)
        @test !(kde_densities(weighted) ≈ kde_densities(mcmc_samples))
    end
end
