# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Core regressions: this file does not load Makie or another plotting backend.
module TestHistogramWeights

using BAT, Test, StatsBase, LogarithmicNumbers

samples(x, w) = DensitySampleVector([collect(v) for v in eachcol(x)],
    zeros(length(w)); weight=copy(w))
binned(marg) = marg.dist isa BAT.ReshapedDist ? marg.dist.dist : marg.dist
histogram(marg) = convert(Histogram, binned(marg))
oracle_weight(w) = BigFloat(w)
oracle_weight(w::ULogarithmic) = exp(BigFloat(log(w)))

# Count represented input mass independently, without StatsBase.fit/normalize
# or any BAT weight preparation. BigFloat prevents the oracle's own overflow.
function bin_mass(x, w, edges; closed=:left)
    mass = zeros(BigFloat, map(e -> length(e) - 1, edges))
    for index in CartesianIndices(mass), row in eachindex(w)
        inside = all(eachindex(edges)) do d
            lo, hi, value = edges[d][index[d]], edges[d][index[d] + 1], x[d, row]
            closed === :left ? lo <= value < hi : lo < value <= hi
        end
        inside && (mass[index] += oracle_weight(w[row]))
    end
    return mass
end

function expected_pdf(x, w, edges; closed=:left)
    mass = bin_mass(x, w, edges; closed)
    probability = mass ./ sum(mass)
    density = similar(probability)
    for index in CartesianIndices(density)
        volume = prod(BigFloat(edges[d][index[d] + 1]) - BigFloat(edges[d][index[d]])
            for d in eachindex(edges))
        density[index] = probability[index] / volume
    end
    return density, probability
end

function check_pdf(h, x, w, edges; closed=:left)
    expected, probability = expected_pdf(x, w, edges; closed)
    @test h.edges == edges
    @test h.isdensity
    @test all(v -> isfinite(v) && v >= 0, h.weights)
    @test all(h.weights[probability .> 0] .> 0)
    @test h.weights ≈ Float64.(expected) rtol=4e-14 atol=0
    integrated = sum(CartesianIndices(h.weights)) do index
        BigFloat(h.weights[index]) * prod(
            BigFloat(edges[d][index[d] + 1]) - BigFloat(edges[d][index[d]])
            for d in eachindex(edges))
    end
    @test integrated ≈ big(1) rtol=4e-14
    return h
end

function check_distribution(source, selection, edges; closed=:left)
    before = deepcopy(source)
    bins = length(edges) == 1 ? only(edges) : edges
    h = histogram(BAT.MarginalDist(source, selection; bins, closed))
    check_pdf(h, reduce(hcat, source.v), source.weight, edges; closed)
    @test source == before
    return h
end

@testset "Checked histogram weights without a plotting backend" begin
    @testset "Actual probability distributions survive common finite scales" begin
        x = reshape([0.25, 0.75, 2.0], 1, :)
        edges = (0.0:1.0:3.0,)
        populations = (
            [1.0, 2.0, 3.0],
            5e307 .* [1.0, 2.0, 3.0],
            1e-308 .* [1.0, 2.0, 3.0],
            nextfloat(0.0) .* [1.0, 2.0, 3.0],
            Float16[10_000, 20_000, 30_000],
            Float32[1e38, 2e38, 3e38],
            big(10)^400 .* BigInt[1, 2, 3],
            big"1e400" .* BigFloat[1, 2, 3],
        )
        for w in populations, closed in (:left, :right)
            check_distribution(samples(x, w), 1, edges; closed)
        end

        # Logarithmic values need not be representable before normalization.
        wlog = exp.(ULogarithmic, 1000.0 .+ log.([1.0, 2.0, 3.0]))
        check_distribution(samples(x, wlog), 1, edges)

        xy = [0.25 0.75 2.0; 0.5 2.0 2.0]
        edges2 = (0.0:1.0:3.0, 0.0:1.0:4.0)
        for scale in (1.0, 5e307, 1e-308), closed in (:left, :right)
            check_distribution(samples(xy, scale .* [1.0, 2.0, 3.0]),
                (1, 2), edges2; closed)
        end
    end

    @testset "Unequal widths cross the actual dependency boundary" begin
        # MarginalDist's existing bins API takes ranges. Unequal explicit
        # vectors belong at its fit/constructor boundary, not a new public API.
        fixtures = (
            (reshape([0.25, 0.75, 2.0], 1, :), ([0.0, 1.0, 3.0],)),
            ([0.25 0.75 2.0; 0.5 2.0 2.0],
                ([0.0, 1.0, 3.0], [0.0, 1.0, 4.0])),
        )
        for (x, edges) in fixtures, scale in (1.0, 5e307, 1e-308), closed in (:left, :right)
            w = scale .* [1.0, 2.0, 3.0]
            before = (copy(x), copy(w), deepcopy(edges))
            probability = BAT._plot_fit_histogram(Tuple(eachrow(x)), w, edges,
                :probability; closed)
            dist = BAT._plot_binned_dist(probability, size(x, 1) == 1)
            check_pdf(convert(Histogram, dist), x, w, edges; closed)
            @test (x, w, edges) == before
        end
    end

    @testset "Integer overflow cannot masquerade as a normalized PDF" begin
        x = reshape([0.5, 0.5, 1.5], 1, :)
        w = [typemax(Int), typemax(Int), 1]
        h = check_distribution(samples(x, w), 1, (0.0:1.0:2.0,))
        # The old wrapped histogram had heights [2,-1] and integral one.
        @test 0 < h.weights[2] < 1e-18
        @test h.weights[1] <= 1
        @test_throws OverflowError BAT._plot_checked_total(w)
        @test_throws OverflowError BAT._plot_checked_total([floatmax(Float64), floatmax(Float64)])
        # Small integer cached sums widen; they must not be checked in UInt8.
        @test BAT._plot_checked_total(UInt8[200, 200, 200]) == 600
        @test BAT._plot_checked_total(BigInt[typemax(Int), typemax(Int), 1]) ==
            2big(typemax(Int)) + 1
    end

    @testset "Absolute counts and density retain their units" begin
        raw = Histogram(([0.0, 1.0, 3.0],), [2, 6], :left, false)
        before = deepcopy(raw)
        expected = (none=[2.0, 6.0], density=[2.0, 3.0],
            probability=[0.25, 0.75], pdf=[0.25, 0.375])
        for mode in (:none, :density, :probability, :pdf)
            result = BAT._plot_normalize_histogram(raw, mode)
            @test result.weights == getproperty(expected, mode)
            @test raw.weights == before.weights && raw.edges == before.edges
            @test result.weights !== raw.weights
        end
        huge = Histogram(([0.0, 1.0, 3.0],), [1e308, 1e308], :left, false)
        @test BAT._plot_normalize_histogram(huge, :probability).weights ≈ [0.5, 0.5]
        @test BAT._plot_normalize_histogram(huge, :pdf).weights ≈ [0.5, 0.25]
        for mode in (:none, :density)
            @test_throws OverflowError BAT._plot_normalize_histogram(huge, mode)
        end
        # Density heights are not counts: their irrelevant sum may overflow.
        density = Histogram(([0.0, 1e-309, 2e-309],), [1e308, 1e308], :left, true)
        @test BAT._plot_normalize_histogram(density, :none).weights == density.weights
        @test BAT._plot_normalize_histogram(density, :density).weights == density.weights
        tiny_density = Histogram(([0.0, 1e-100],), [1e-300], :left, true)
        @test only(BAT._plot_normalize_histogram(tiny_density, :pdf).weights) ≈ 1e100
    end

    @testset "Final quotient representability governs histogram normalization" begin
        # Multiplying the first two widths overflows, but the final PDF does not.
        for widths in ((1e200, 1e200, 1e-200), (1e-200, 1e-200, 1e200))
            edges = map(width -> [0.0, width], widths)
            h = Histogram(edges, ones(1, 1, 1), :left, false)
            expected = Float64(inv(prod(BigFloat.(widths))))
            @test only(BAT._plot_normalize_histogram(h, :pdf).weights) ≈ expected rtol=4e-14
        end

        # A probability need not be materialized if division by a tiny width
        # makes the final density representable. The probability mode must fail.
        tiny = nextfloat(0.0)
        h = Histogram(([0.0, tiny, 1.0, 2.0],), [tiny, 1.0, 1.0], :left, false)
        @test_throws ArgumentError BAT._plot_normalize_histogram(h, :probability)
        density = BAT._plot_normalize_histogram(h, :pdf)
        @test density.weights ≈ fill(0.5, 3) rtol=4e-14
        # Two unit rows accumulate in one bin. A second maximum rescaling of
        # the fitted [tiny, 2] heights would incorrectly erase the other bin.
        fitted = BAT._plot_fit_histogram(([0.0, 1.0, 1.0],), [tiny, 1.0, 1.0],
            ([0.0, tiny, 2.0],), :pdf)
        @test fitted.weights ≈ [0.5, 0.5] rtol=4e-14
        impossible = Histogram(([0.0, tiny],), [1.0], :left, false)
        @test_throws ArgumentError BAT._plot_normalize_histogram(impossible, :pdf)
        @test_throws ArgumentError BAT._plot_normalize_histogram(impossible, :density)
    end

    @testset "Actual constructor limits fail explicitly" begin
        for widths in ((1e200, 1e200, 1e-200), (1e-200, 1e-200, 1e200))
            edges = map(width -> range(0.0, width; length=2), widths)
            source = samples(reshape(collect(widths) ./ 2, 3, 1), [1.0])
            before = deepcopy(source)
            # EmpiricalDistributions materializes the volume in its own order.
            @test_throws ArgumentError BAT.MarginalDist(source, (1, 2, 3); bins=edges)
            @test source == before
        end
        x = reshape([0.5, 1.5], 1, :)
        for w in ([0.0, 0.0], [-1.0, 2.0], [Inf, 1.0], [NaN, 1.0], [1e308, 1e-308]),
                filter in (false, true)
            source = samples(x, w)
            before = deepcopy(source)
            @test_throws ArgumentError BAT.MarginalDist(source, 1;
                bins=0.0:1.0:2.0, filter)
            for field in (:v, :logd, :weight, :info, :aux)
                @test isequal(getproperty(source, field), getproperty(before, field))
            end
        end
        @test_throws ArgumentError BAT.MarginalDist(samples(x, ones(2)), 1; bins=3.0:1.0:4.0)
        @test_throws ArgumentError BAT._plot_normalize_histogram(
            Histogram(([0.0, 1.0],), [0.0], :left, false), :unsupported)
    end

    @testset "Report fields preserve checked absolute totals" begin
        # A zero-weight upper endpoint preserves the existing left-closed
        # range convention while all 600 units of positive mass lie inside.
        x = reshape([0.0, 0.25, 0.5, 1.0], 1, :)
        w = UInt8[200, 200, 200, 0]
        source = samples(x, w)
        before = deepcopy(source)
        h = only(BAT._marginal_histograms(source))
        @test sum(h.weights) == 600
        @test BigFloat.(h.weights) == bin_mass(x, w, h.edges)
        @test source == before

        huge = samples(reshape([0.0, 1.0, 2.0], 1, :), [1e308, 1e308, 0.0])
        before_huge = deepcopy(huge)
        @test_throws OverflowError BAT._marginal_histograms(huge)
        report = BAT.LazyReports.lazyreport()
        @test_throws OverflowError BAT.LazyReports.lazyreport!(report, huge)
        @test report == BAT.LazyReports.lazyreport()
        @test huge == before_huge
    end

    @testset "Low-weight filtering preserves mass units, rows and cutoff ties" begin
        values = reshape(collect(1.0:5.0), 1, :)
        # Sorted cumulative mass crosses the default fraction at 5e-6; the
        # >= cutoff policy retains both tied 5e-6 rows and removes only 1e-6.
        base = [1.0, 1e-6, 5e-6, 5e-6, 1.0]
        for scale in (1.0, 1e308, 1e-300)
            source = samples(values, scale .* base)
            before = deepcopy(source)
            filtered = BAT.drop_low_weight_samples(source)
            @test first.(filtered.v) == [1.0, 3.0, 4.0, 5.0]
            @test filtered.weight == source.weight[[1, 3, 4, 5]]
            @test source == before
        end
        zero_cutoff = samples(reshape([1.0, 2.0], 1, :), [0.0, 1.0])
        @test BAT.drop_low_weight_samples(zero_cutoff) == zero_cutoff
        t = nextfloat(0.0)
        @test all(BAT._plot_low_weight_mask([t, 2t, ones(8)...], 0.0))
        @test BAT._plot_low_weight_mask([t, 2t, ones(8)...], t) ==
            [false, true, trues(8)...]
    end

    @testset "Rational plotting weights widen before division" begin
        weights = [Int64(4000000001)//4000000007, Int64(4000000003)//4000000009]
        exact = Rational{BigInt}.(weights)
        @test minimum(exact) / maximum(exact) > 1//100
        source = samples(reshape([0.0, 1.0], 1, :), weights)
        before = deepcopy(source)
        @test BAT.drop_low_weight_samples(source) === source
        @test source == before

        for I in (Int32, Int64, Int128, UInt64)
            q = typemax(I)
            w = [(q - I(1))//q, (q - I(2))//(q - I(1)), zero(I)//one(I)]
            wide = Rational{BigInt}.(w)
            expected = Float64.(wide ./ maximum(wide))
            @test BAT._plot_relative_weights(w) ≈ expected rtol=2e-15
            @test BAT._plot_relative_weights(view(w, :); unit_sum=true) ≈
                Float64.(wide ./ sum(wide)) rtol=2e-15
            @test BAT._plot_relative_weights(w; scale=one(w[1])) ≈ Float64.(wide) rtol=2e-15
            @test BAT._plot_relative_weights(w; scale=big(2)) ≈ BigFloat.(wide ./ 2) rtol=2e-15
            @test_throws ArgumentError BAT._plot_relative_weights(w; scale=one(w[1])/2)
            check_distribution(samples(reshape([0.5, 1.5, 2.5], 1, :), w),
                1, (0.0:1.0:3.0,))
        end

        # Equal working values must not erase ordering or ties in stored weights.
        @test weights[1] < weights[2]
        @test Float64(weights[1]) == Float64(weights[2])
        tied = samples(reshape(collect(1.0:4.0), 1, :),
            [weights[2], weights[1], weights[2], 0//1])
        before_tied = deepcopy(tied)
        filtered = BAT.drop_low_weight_samples(tied, 1)
        @test first.(filtered.v) == [1.0, 3.0]
        @test filtered.weight == tied.weight[[1, 3]]
        @test tied == before_tied

        setprecision(BigFloat, 2) do
            @test BAT._plot_relative_weights([1//3, 1//1]) == [1/3, 1.0]
            @test BAT._plot_relative_weights([1//3, 1//1]; unit_sum=true) == [0.25, 0.75]
            @test BAT.drop_low_weight_samples(source) === source
            @test BAT.drop_low_weight_samples(tied, 1).weight == tied.weight[[1, 3]]
        end

        for w in (Rational{Int64}[], [0//1, 0//1])
            @test BAT._plot_relative_weights(w) == zeros(length(w))
            @test BAT.drop_low_weight_samples(samples(zeros(1, length(w)), w)).weight == w
        end
        for w in ([-1//2, 1//1], [1//0, 1//1])
            @test_throws ArgumentError BAT._plot_relative_weights(w)
            @test_throws ArgumentError BAT.drop_low_weight_samples(samples(zeros(1, 2), w))
        end
        for scale in (0//1, -1//1, 1//0)
            @test_throws ArgumentError BAT._plot_relative_weights(weights; scale)
        end
        for scale in (big(10)^400, big(1)//big(10)^400)
            wide = Rational{BigInt}[1//3, 1//1] .* scale
            @test BAT._plot_relative_weights(wide) ≈ BigFloat[1//3, 1] rtol=2e-15
        end
    end
end

end # module
