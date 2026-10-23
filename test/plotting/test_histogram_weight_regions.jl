# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestHistogramWeightRegions

using BAT, StatsBase, Test

hist1d(weights; edges=collect(0:length(weights)), density=false) =
    Histogram((edges,), copy(weights), :left, density)

function samples1d(values, weights)
    DensitySampleVector([[x] for x in values], zeros(length(values)); weight=copy(weights))
end

interval_bounds(ivs) = [(minimum(iv), maximum(iv)) for iv in ivs]

@testset "Histogram region weight arithmetic" begin
    @testset "Independent height-mass and selection oracles" begin
        # Unit-width bins retain the independent integer-mass oracle.
        for scale in (1.0, 5e307, 1e-300, nextfloat(0.0))
            hist = hist1d(scale .* [3, 2, 1])
            original = deepcopy(hist)
            regions, content = BAT.get_smallest_intervals(hist, [0.25, 0.6])
            @test regions[1].weights == hist.weights .* [1, 1, 0]
            @test regions[2].weights == hist.weights .* [1, 0, 0]
            @test content ≈ [5/6, 1/2]
            @test hist.weights == original.weights
            @test hist.edges == original.edges
            @test regions[1].weights !== hist.weights
            @test regions[1].weights !== regions[2].weights

            central = hist1d(scale .* [1, 2, 3, 2, 1])
            central_regions, central_content = BAT.split_central(central, [0.5])
            @test only(central_regions).weights == central.weights .* [0, 1, 1, 1, 0]
            @test only(central_content) ≈ 7/9
        end

        # Independently normalizing each subset would incorrectly report 1.
        parent = hist1d([1e308, 5e307, 2.5e307])
        selected = StatsBase.Histogram[
            hist1d([0.0, 5e307, 0.0]), hist1d([0.0, 0.0, 2.5e307]),
        ]
        @test BAT.get_probability_content(parent, selected) ≈ [2/7, 1/7]

        # H16 includes the complete density plateau at the boundary.
        ties = hist1d([2, 2, 1])
        regions, content = BAT.get_smallest_intervals(ties, [0.3])
        @test only(regions).weights == [2, 2, 0]
        @test only(content) ≈ 4/5
    end

    @testset "Original typed heights, wider relative arithmetic" begin
        exact_height = Int64(2)^53 + 1
        hist = hist1d([exact_height, exact_height, Int64(1)])
        regions, content = BAT.get_smallest_intervals(hist, [0.4])
        @test eltype(only(regions).weights) == Int64
        @test only(regions).weights == [exact_height, exact_height, 0]
        @test only(content) ≈ 1.0

        # No absolute integer sum is needed to select a relative region.
        extreme = hist1d([typemax(Int), typemax(Int), 1])
        regions, content = BAT.get_smallest_intervals(extreme, [0.4])
        @test only(regions).weights == [typemax(Int), typemax(Int), 0]
        @test only(content) ≈ 1.0

        for weights in (Float16[45000, 30000, 15000],
                Float32[3e38, 2e38, 1e38],
                big(10)^400 .* [3, 2, 1],
                BigFloat[big"3e1000", big"2e1000", big"1e1000"])
            hist = hist1d(weights)
            regions, content = BAT.get_smallest_intervals(hist, [0.6])
            @test eltype(only(regions).weights) == eltype(weights)
            @test only(regions).weights == weights .* [1, 1, 0]
            # Float16(45000) rounds before the function sees it. The oracle
            # uses exact represented values in wider arithmetic.
            expected = sum(BigFloat.(weights[1:2])) / sum(BigFloat.(weights))
            @test isapprox(only(content), expected; rtol=1e-13)
        end

        weights2d = reshape(Int64[exact_height, 2, 3, 4], 2, 2)
        hist2d = Histogram(([0, 1, 2], [0, 1, 2]), weights2d, :left, false)
        regions, _ = BAT.get_smallest_intervals(hist2d, [0.5])
        @test size(only(regions).weights) == (2, 2)
        @test only(regions).weights[1, 1] == exact_height
        @test count(!iszero, only(regions).weights) == 1
        @test hist2d.weights == weights2d
    end

    @testset "Unequal bins use integrated mass" begin
        hist = hist1d([0.0, 0.15, 0.2]; edges=[0.0, 2.0, 6.0, 8.0], density=true)
        regions, content = BAT.get_smallest_intervals(hist, [0.5])
        @test only(regions).weights == [0.0, 0.15, 0.2]
        @test only(content) == 1
        @test sum(only(regions).weights .* diff(only(hist.edges))) ≈ 1
    end

    @testset "Contour cutoffs share the displayed field and initialize crossings" begin
        h2(values) = Histogram(([0., 1., 2.], [0., 1., 2.]), reshape(values, 2, 2), :left, false)
        result = BAT._histogram_contour_data(h2([9., 1., 0., 0.]), [0.1, 0.2, 0.8])
        @test result.cutoffs == [1, 1, 1]
        @test result.contents ≈ [0.9, 0.9, 0.9]
        @test maximum(result.density) == 1
        exact = BAT._histogram_contour_data(h2([6., 3., 1., 0.]), [0.6])
        @test exact.cutoffs == [1]
        @test isempty(BAT._histogram_contour_data(h2([6., 3., 1., 0.]), Float64[]).cutoffs)
        # Contour density cutoffs need no absolute count sum or 1.1*sum cap.
        for weights in ([typemax(Int), typemax(Int), 0, 0], [1e308, 1e308, 0, 0])
            @test BAT._histogram_contour_data(h2(weights), [0.5]).contents == [1]
        end
    end

    @testset "Zero mass, invalid input and positive-loss boundaries" begin
        hist = hist1d(zeros(3))
        for region_function in (BAT.get_smallest_intervals, BAT.split_central)
            regions, content = region_function(hist, [0.5, 0.9])
            @test all(h -> all(iszero, h.weights), regions)
            @test content == [0.0, 0.0]
            regions, content = region_function(hist, Float64[])
            @test isempty(regions)
            @test isempty(content)
        end
        @test BAT._histogram_contour_data(Histogram(([0., 1., 2.], [0., 1., 2.]), zeros(2, 2), :left, false), [0.5, 0.9]).cutoffs == [nothing, nothing]
        @test BAT.get_probability_content(hist, StatsBase.Histogram[deepcopy(hist)]) == [0]
        @test_throws ArgumentError BAT.get_probability_content(hist,
            StatsBase.Histogram[hist1d([1.0, 0, 0])])

        for bad in ([1.0, -1], [1.0, NaN], [1.0, Inf], [1e308, 1e-308])
            for region_function in (BAT.get_smallest_intervals, BAT.split_central)
                @test_throws ArgumentError region_function(hist1d(bad), [0.5])
            end
            @test_throws ArgumentError BAT.get_probability_content(hist1d(bad), StatsBase.Histogram[])
        end
        for intervals in ([-0.1], [1.1], [NaN], [Inf])
            @test_throws ArgumentError BAT.get_smallest_intervals(hist, intervals)
            @test_throws ArgumentError BAT.split_central(hist, intervals)
            @test_throws ArgumentError BAT._histogram_contour_data(Histogram(([0., 1., 2.], [0., 1., 2.]), zeros(2, 2), :left, false), intervals)
        end
    end
end

@testset "Sample credible-interval weight preparation" begin
    values = [0.0, 1.0, 3.0, 7.0]
    weights = [1.0, 2.0, 4.0, 2.0]
    # Sample containers use the upstream empirical interval semantics while
    # preserving their stored weights and ignoring zero-mass coordinates.
    reference = BAT.smallest_credible_intervals(values, Weights(weights))
    for scale in (1.0, 4e307, 1e-300, nextfloat(0.0))
        samples = samples1d(values, scale .* weights)
        before = deepcopy(samples)
        result = only(BAT.smallest_credible_intervals(samples))
        @test interval_bounds(result) == interval_bounds(reference)
        @test samples.weight == before.weight
        @test samples.v == before.v
    end

    # Empirical mass depends on values and counts, not quantile interpolation
    # conventions of the weight wrapper.
    ordinary = BAT.smallest_credible_intervals([0.0, 2.0], Weights([1, 1]))
    @test all(iv -> minimum(iv) in (0, 2) && maximum(iv) in (0, 2), ordinary)
    @test interval_bounds(BAT.smallest_credible_intervals([0.0, 2.0], Weights([2, 2]))) ==
        interval_bounds(ordinary)
    frequency = BAT.smallest_credible_intervals([0.0, 2.0], FrequencyWeights([2, 2]))
    @test interval_bounds(frequency) == interval_bounds(ordinary)

    with_zero = samples1d([NaN, 0.0, 1.0, 3.0, 7.0, Inf], [0.0; weights; 0.0])
    before = deepcopy(with_zero)
    @test interval_bounds(only(BAT.smallest_credible_intervals(with_zero))) == interval_bounds(reference)
    @test isequal(with_zero.v, before.v)
    @test with_zero.weight == before.weight
    constant = samples1d([3.0, NaN], [1e308, 0.0])
    @test interval_bounds(only(BAT.smallest_credible_intervals(constant))) == [(3.0, 3.0)]

    for bad in ([0.0, 0.0], [1.0, -1.0], [1.0, Inf], [1.0, NaN])
        @test_throws ArgumentError BAT.smallest_credible_intervals(samples1d([0.0, 1.0], bad))
    end
    @test_throws ArgumentError BAT.smallest_credible_intervals(with_zero[1:0])
    @test_throws ArgumentError BAT.smallest_credible_intervals(samples1d([0.0, NaN], [1.0, 1.0]))
    # Constant-only samples have no coordinate rows; validate their weights
    # before the per-coordinate interval loop can be skipped.
    zero_dim_samples(w) = DensitySampleVector([Float64[], Float64[]], zeros(2); weight=w)
    @test isempty(BAT.smallest_credible_intervals(zero_dim_samples([1.0, 2.0])))
    for bad in ([0.0, 0.0], [-1.0, 2.0], [NaN, 1.0], [Inf, 1.0])
        @test_throws ArgumentError BAT.smallest_credible_intervals(zero_dim_samples(bad))
    end
    # Exact mass and width comparisons admit these cases without Float64
    # normalization or subtraction; full coverage must retain both endpoints.
    for (values, weights) in (([0.0, 1.0], [1e308, 1e-308]),
                              ([-1e308, 1e308], [1.0, 1.0]))
        samples = samples1d(values, weights)
        before = deepcopy(samples)
        for mode in (:connected, :disjoint)
            intervals = only(BAT.smallest_credible_intervals(samples; p=1, mode))
            @test all(x -> any(iv -> x in iv, intervals), values)
            @test intervals == BAT.smallest_credible_intervals(values, Weights(weights); p=1, mode)
        end
        @test samples == before
    end
end

end # module
