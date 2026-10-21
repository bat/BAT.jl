# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestBinnedMoments
using BAT, Test, StatsBase, Statistics, Distributions, EmpiricalDistributions
using LinearAlgebra, Random, ValueShapes

const Q = Rational{BigInt}
samples(columns, weights) = DensitySampleVector(
    [collect(row) for row in zip(columns...)], zeros(length(weights)); weight=copy(weights))
binned(marginal) = marginal.dist isa UvBinnedDist ? marginal.dist : unshaped(marginal.dist)

# Exact bin-center population moments, independent of dependency normalization,
# moment routines and the production checked arithmetic.
function oracle(edges, weights)
    centers = map(e -> [(Q(e[i]) + Q(e[i + 1])) / 2 for i in 1:length(e)-1], edges)
    p = Q.(weights)
    p ./= sum(p)
    indices = CartesianIndices(p)
    n = length(edges)
    mu = [sum(p[I] * centers[d][I[d]] for I in indices) for d in 1:n]
    covariance = [sum(p[I] * (centers[d][I[d]] - mu[d]) *
        (centers[e][I[e]] - mu[e]) for I in indices) for d in 1:n, e in 1:n]
    return Float64.(mu), Float64.(covariance)
end

function moment_error(f)
    err = try
        f()
        nothing
    catch exception
        exception
    end
    @test err isa ArgumentError
    @test err isa ArgumentError && occursin("Binned coordinate moments:", sprint(showerror, err))
end

# Old allowed dependencies draw bin selection from the default RNG (H32).
# Preserve both input streams and their consumption without blessing that defect.
function draws(dist, seed)
    Random.seed!(619)
    explicit = Xoshiro(seed)
    values = rand(explicit, dist, 16)
    return values, explicit, copy(Random.default_rng())
end

@testset "Checked binned coordinate moments" begin
    @testset "Public finite-wrong and overflow regressions" begin
        fixtures = (
            (([-2.0^-600, nextfloat(2.0^-600)], [0.0, 2.0^1023]), ones(1, 1)),
            (([-2.0^-500, 0.0, 2.0^-500], [0.0, 2.0^1023]), fill(0.5, 2, 1)),
            (([-2.0^520, 0.0, 2.0^520],), [2.0^-50, 1.0 - 2.0^-50]),
            (([2.0^100, 2.0^100 + 2.0^50], [0.0, 2.0^-1000]), ones(1, 1)),
        )
        mu, covariance = oracle(fixtures[1]...)
        @test mu[1] == 2.0^-653
        @test iszero(covariance)
        @test oracle(fixtures[2]...)[2][1, 1] == 2.0^-1002
        @test isfinite(oracle(fixtures[3]...)[2][1, 1])
        for (edges, weights) in fixtures, closed in (:left, :right)
            hist = Histogram(edges, weights, closed, false)
            before = deepcopy(hist)
            moment_error(() -> BAT._plot_binned_dist(hist, length(edges) == 1))
            @test hist == before
            columns = ntuple(length(edges)) do d
                [Float64((Q(edges[d][I[d]]) + Q(edges[d][I[d] + 1])) / 2)
                    for I in vec(collect(CartesianIndices(weights)))]
            end
            source = samples(columns, vec(weights))
            original = deepcopy(source)
            ranges = map(e -> range(first(e), last(e); length=length(e)), edges)
            selection = length(edges) == 1 ? 1 : Tuple(1:length(edges))
            bins = length(edges) == 1 ? only(ranges) : ranges
            moment_error(() -> BAT.MarginalDist(source, selection; bins, closed))
            @test source == original
        end
    end

    @testset "Independent moments, shapes and ordinary public entries" begin
        rng = Xoshiro(711)
        for n in (1, 2, 3), k in 1:16
            edges = ntuple(_ -> cumsum([rand(rng) - 2; rand(rng, 3) .+ 0.25]), n)
            counts = rand(rng, 1:9, ntuple(_ -> 3, n)...)
            weights = Float64.(counts) ./ sum(counts)
            hist = Histogram(edges, weights, isodd(k) ? :left : :right, false)
            before = deepcopy(hist)
            dist = BAT._plot_binned_dist(hist, n == 1)
            mu, covariance = oracle(edges, weights)
            @test [mean(dist)...] ≈ mu rtol=3e-14 atol=1e-15
            @test n == 1 ? isapprox(var(dist), covariance[1, 1]; rtol=5e-14) :
                isapprox(cov(dist), covariance; rtol=5e-14, atol=1e-15)
            @test hist == before
        end

        for exponent in (-500, -100, 0, 100, 500), n in (1, 2)
            edges = ntuple(_ -> [-2.0, 0.0, 2.0] .* 2.0^exponent, n)
            weights = fill(2.0^-n, ntuple(_ -> 2, n))
            dist = BAT._plot_binned_dist(Histogram(edges, weights, :left, false), n == 1)
            mu, covariance = oracle(edges, weights)
            @test [mean(dist)...] == mu
            @test n == 1 ? var(dist) == covariance[1, 1] : cov(dist) == covariance
        end
        # Larger nonuniform grids also exercise the dependency's reductions.
        edges = (collect(range(-3., 4.; length=101)), collect(range(-1., 2.; length=91)))
        weights = rand(rng, 100, 90)
        weights ./= sum(weights)
        hist = Histogram(edges, weights, :left, false)
        dist = BAT._plot_binned_dist(hist, false)
        reference = MvBinnedDist(hist)
        @test mean(dist) == mean(reference)
        @test cov(dist) == cov(reference)
        for closed in (:left, :right)
            source = samples(([0.25, 0.75, 1.25, 1.75], [1.75, 1.25, 0.75, 0.25]), [1., 2., 3., 2.])
            before = deepcopy(source)
            for selection in (1, (1, 2))
                bins = selection == 1 ? (0.0:1.0:2.0) : (0.0:1.0:2.0, 0.0:1.0:2.0)
                marg = BAT.MarginalDist(source, selection; bins, closed)
                dist = binned(marg)
                @test dist isa (selection == 1 ? UvBinnedDist : MvBinnedDist)
                hist = convert(Histogram, dist)
                masses = selection == 1 ? [3/8, 5/8] : [0.0 3/8; 5/8 0.0]
                mu, covariance = oracle(hist.edges, masses)
                @test [mean(dist)...] ≈ mu
                @test selection == 1 ? var(dist) ≈ covariance[1, 1] : cov(dist) ≈ covariance
            end
            @test source == before
            shape = NamedTupleShape(x=ScalarShape{Real}(), y=ScalarShape{Real}())
            shaped = shape.(source)
            marginal = BAT.MarginalDist(shaped, (:y, :x); bins=(0.0:1.0:2.0, 0.0:1.0:2.0), closed)
            @test varshape(marginal.dist) == varshape(BAT.bat_marginalize(shaped, (:y, :x)).result)
            @test mean(binned(marginal)) ≈ [7/8, 9/8]
            @test shaped == shape.(before)
        end
        original = BAT.MarginalDist(samples(([0.25, 0.75, 1.25],), [1., 2., 1.]), 1; bins=0.:1.:2.)
        @test BAT.MarginalDist(original, 1; bins=8, nsamples=128) isa BAT.MarginalDist
        prior = NamedTupleDist(x=Uniform(0, 2), y=Uniform(0, 2))
        @test BAT.MarginalDist(prior, :x; bins=8, nsamples=128) isa BAT.MarginalDist
    end

    @testset "Representation, sampling and ownership preservation" begin
        for T in (Float32, Float64, BigFloat), closed in (:left, :right), n in (1, 2)
            edges = ntuple(_ -> T.([-2, 0, 2]), n)
            weights = reshape(n == 1 ? [0.25, 0.75] : [0.125, 0.25, 0.125, 0.5], ntuple(_ -> 2, n))
            T === BigFloat && (weights = BigFloat.(weights))
            hist = Histogram(edges, weights, closed, false)
            before = deepcopy(hist)
            actual = BAT._plot_binned_dist(hist, n == 1)
            original = n == 1 ? UvBinnedDist(hist) : MvBinnedDist(hist)
            @test typeof(actual) == typeof(original)
            @test convert(Histogram, actual) == convert(Histogram, original)
            @test mean(actual) == mean(original)
            @test var(actual) == var(original)
            @test mode(actual) == mode(original)
            @test draws(actual, 73) == draws(original, 73)
            if n == 1
                @test cdf.(Ref(actual), [-2., -1., 0., 1., 2.]) == cdf.(Ref(original), [-2., -1., 0., 1., 2.])
                @test quantile.(Ref(actual), [0., 0.25, 0.5, 1.]) == quantile.(Ref(original), [0., 0.25, 0.5, 1.])
            else
                @test cov(actual) == cov(original)
            end
            copied = convert(Histogram, actual)
            copied.weights[1] = 0
            copied.edges[1][1] -= 1
            @test hist == before
            @test convert(Histogram, actual) == convert(Histogram, original)
        end
        for edge in (0:1:4, range(0//1, 4//1; length=5), range(0., 4.; length=5))
            hist = Histogram((edge,), [0.125, 0.25, 0.375, 0.25], :left, false)
            dist = BAT._plot_binned_dist(hist, true)
            @test convert(Histogram, dist).edges == (collect(float.(edge)),)
            @test mean(dist) == mean(UvBinnedDist(hist))
            @test var(dist) == var(UvBinnedDist(hist))
        end
        for edges in ((0:1:2, 0:1:2), (Float32[0, 1, 2], [0., 1., 2.]))
            hist = Histogram(edges, [0.125 0.25; 0.375 0.25], :left, false)
            dist = BAT._plot_binned_dist(hist, false)
            reference = MvBinnedDist(hist)
            @test dist isa MvBinnedDist
            @test convert(Histogram, dist) == convert(Histogram, reference)
            @test mean(dist) == mean(reference)
            @test cov(dist) == cov(reference)
        end
        # Indexed edge arithmetic makes this representable public range work,
        # even though broadcasting on the range itself used to produce NaN.
        for closed in (:left, :right)
            source = samples(([2.0^1022],), [1.])
            dist = BAT.MarginalDist(source, 1; bins=range(0., 2.0^1023; length=2), closed).dist
            @test mean(dist) == 2.0^1022
            @test mode(dist) == 2.0^1022
            @test var(dist) == 0.0
        end
        hist = Histogram(([2^62, 2^62 + 2^20],), [1.], :left, false)
        @test mean(BAT._plot_binned_dist(hist, true)) == Float64(2^62 + 2^19)
        moment_error(() -> BAT._plot_binned_dist(Histogram(([2^62, 2^62 + 1],), [1.], :left, false), true))
    end

    @testset "Range, zero and actual-result admission" begin
        for (edges, weights) in (
                (([floatmax(Float64)/1.5, floatmax(Float64)],), [1.]),
                (([-2.0^-540, 0., 2.0^-540],), [0.5, 0.5]),
                (([-2.0^700, 0., 1.],), [0., 1.]),
                ((Float32[0, 2.0^-100, 2.0^-99],), [0.5, 0.5]))
            moment_error(() -> BAT._plot_binned_dist(Histogram(edges, weights, :left, false), true))
        end
        for exponent in (-500, 0, 500), n in (1, 2)
            hist = Histogram(ntuple(_ -> [0., 2.0^exponent], n), ones(ntuple(_ -> 1, n)), :left, false)
            @test all(iszero, var(BAT._plot_binned_dist(hist, n == 1)))
        end
        hist = Histogram(([-2., 0., 2.], [-2., 0., 2.]), [0. 0.5; 0.5 0.], :left, false)
        dist = BAT._plot_binned_dist(hist, false)
        @test mean(dist) == [0., 0.]
        @test cov(dist) == [1. -1.; -1. 1.]
        checked = BAT._plot_binned_moments(StatsBase.normalize(hist), fill(4.0, size(hist.weights)), false)
        moment_error(() -> BAT._plot_check_binned_moments(dist, merge(checked, (mean=(1., 0.),)), false))
        moment_error(() -> BAT._plot_check_binned_moments(dist, merge(checked, (cov=zeros(2, 2),)), false))
        @test BAT._plot_moment_mul(0., 1e300) == 0
        for f in (() -> BAT._plot_moment_add(floatmax(Float64), floatmax(Float64)),
                () -> BAT._plot_moment_sub(floatmax(Float64), -floatmax(Float64)),
                () -> BAT._plot_moment_mul(2.0^-600, 2.0^-600),
                () -> BAT._plot_moment_mul(2.0^-600, 2.0^-450),
                () -> BAT._plot_moment_div(nextfloat(0.), 2),
                () -> BAT._plot_moment_convert(Float32, 2.0^-140),
                () -> BAT._plot_moment_convert(Float32, 2.0^-160))
            moment_error(f)
        end
        before_rng = copy(Random.default_rng())
        before_precision = precision(BigFloat)
        BAT._plot_binned_dist(hist, false)
        @test copy(Random.default_rng()) == before_rng
        @test precision(BigFloat) == before_precision
        previous = get_zero_subnormals()
        try
            set_zero_subnormals(true)
            moment_error(() -> BAT._plot_binned_dist(hist, false))
        finally
            set_zero_subnormals(previous)
        end
        setrounding(BigFloat, RoundDown) do
            big_hist = Histogram((BigFloat[0, 1, 2],), BigFloat[0.5, 0.5], :left, false)
            moment_error(() -> BAT._plot_binned_dist(big_hist, true))
            @test rounding(BigFloat) == RoundDown
        end
    end
end
end # module
