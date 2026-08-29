# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using IntervalSets
using StatsBase
using Test

function _exact_weight(w::AbstractFloat)
    significand, exponent, sign = Base.decompose(w)
    numerator = BigInt(sign) * significand
    exponent < 0 ? numerator // (big(1) << -exponent) : (numerator << exponent) // big(1)
end
_exact_weight(w::Real) = Rational{BigInt}(w)

function _union_mass(values, weights, intervals)
    exact_weights = _exact_weight.(weights)
    included = sum(
        (exact_weights[i] for i in eachindex(values)
         if any(interval -> values[i] in interval, intervals));
        init = 0 // big(1),
    )
    included / sum(exact_weights)
end

function _brute_connected(values, weights, target)
    atoms = sort(unique(values[findall(!iszero, weights)]))
    candidates = (
        ClosedInterval(atoms[i], atoms[j]) for i in eachindex(atoms) for j in i:lastindex(atoms)
        if _union_mass(values, weights, [ClosedInterval(atoms[i], atoms[j])]) >= target
    )
    first(sort!(collect(candidates); by = interval -> (
        _exact_weight(maximum(interval)) - _exact_weight(minimum(interval)),
        minimum(interval),
        maximum(interval),
    )))
end

function _check_modes(values, weights, nsigma, target)
    results = map((:disjoint, :connected)) do mode
        intervals = BAT.smallest_credible_intervals(
            values, Weights(weights); nsigma_equivalent = nsigma, mode
        )
        @test _union_mass(values, weights, intervals) >= target
        intervals
    end
    @test only(results[2]) == _brute_connected(values, weights, target)
    results
end

function _prefix_union(atoms, boundaries, ranking, r)
    intervals = sort([
        ClosedInterval(first(atoms[boundaries[i]]), first(atoms[boundaries[i + 1]]))
        for i in ranking[1:r]
    ]; by = minimum)
    merged = eltype(intervals)[]
    for interval in intervals
        if isempty(merged) || minimum(interval) > maximum(last(merged))
            push!(merged, interval)
        else
            merged[end] = ClosedInterval(minimum(last(merged)), maximum(interval))
        end
    end
    merged
end

function _disjoint_oracle(values, weights, m, n)
    exact_weights = _exact_weight.(weights)
    common_denominator = foldl(lcm, denominator.(exact_weights); init = big(1))
    rows = sort!([
        (values[i], numerator(exact_weights[i]) * div(
            common_denominator, denominator(exact_weights[i])
        )) for i in eachindex(values)
    ]; by = first)
    atoms = eltype(rows)[]
    for row in rows
        iszero(last(row)) && continue
        if !isempty(atoms) && first(row) == first(last(atoms))
            atoms[end] = (first(row), last(last(atoms)) + last(row))
        else
            push!(atoms, row)
        end
    end

    total = sum(last, atoms)
    boundaries = [findfirst(k -> n * sum(last, @view atoms[1:k]) >= j * total,
        eachindex(atoms)) for j in 0:n]
    ranking = sortperm(1:n; by = i -> (
        _exact_weight(first(atoms[boundaries[i + 1]])) -
            _exact_weight(first(atoms[boundaries[i]])), i
    ))
    for r in m:n
        intervals = _prefix_union(atoms, boundaries, ranking, r)
        _union_mass(values, weights, intervals) >= m // n && return intervals
    end
    error("no qualifying disjoint prefix")
end

@testset "empirical credible intervals" begin
    targets = ((1, 28, 41), (1.64, 9, 10), (2, 42, 44), (3, 369, 370))

    @testset "modes and objectives" begin
        values = [-10, -9, 9, 10]
        weights = [20, 1, 1, 20]
        disjoint, connected = _check_modes(values, weights, 1, 28 // 41)
        @test length(disjoint) > 1
        @test connected == [ClosedInterval(-10, 10)]

        regression_values = [-3, -1, 2, 8]
        regression_weights = [0, 1, 2, 5]
        disjoint = first(_check_modes(
            regression_values, regression_weights, 1, 28 // 41
        ))
        @test disjoint == _disjoint_oracle(
            regression_values, regression_weights, 28, 41
        )
        for (oracle_values, oracle_weights, nsigma, m, n) in (
                ([-10, -9, 9, 10], [20, 1, 1, 20], 1, 28, 41),
                ([0, 0, 1, 5, 9], [1, 2, 1, 3, 1], 1, 28, 41),
                ([0, 1, 2, 10], [1, 3, 1, 5], 1.64, 9, 10),
            )
            @test BAT.smallest_credible_intervals(
                oracle_values, Weights(oracle_weights); nsigma_equivalent = nsigma
            ) == _disjoint_oracle(oracle_values, oracle_weights, m, n)
        end
    end

    @testset "exact inverse ECDF" begin
        atoms = BAT._credible_atoms([0, 1, 2], UInt128[1, 1, 2])
        @test BAT._credible_grid(atoms, 4) == [1, 1, 2, 3, 3]

        left = BAT._credible_atoms([0, 1], UInt128[1, 3])
        right = BAT._credible_atoms([0, 1], UInt128[1, 3 + 1])
        above = BAT._credible_atoms([0, 1], UInt128[2, 3])
        @test BAT._credible_grid(left, 4)[2] == 1
        @test BAT._credible_grid(right, 4)[2] == 2
        @test BAT._credible_grid(above, 4)[2] == 1

        repeated = BAT._credible_grid(BAT._credible_atoms([0, 10], UInt128[3, 1]), 41)
        @test count(==(1), repeated) > 1
        @test count(==(2), repeated) > 1
    end

    @testset "exact targets" begin
        for (nsigma, m, n) in targets
            target = m // n
            exact = _check_modes([0, 10], [m, n - m], nsigma, target)
            @test only(exact[2]) == ClosedInterval(0, 0)

            below = [prevfloat(Float64(m)), Float64(n - m)]
            below_results = _check_modes([0, 10], below, nsigma, target)
            @test only(below_results[2]) == ClosedInterval(0, 10)
        end
    end

    @testset "measure invariance" begin
        values = [-3.0, -1.0, 2.0, 8.0]
        weights = [0, 1, 2, 5]
        for mode in (:disjoint, :connected)
            reference = BAT.smallest_credible_intervals(values, Weights(weights); mode)
            @test BAT.smallest_credible_intervals(
                reverse(values), Weights(reverse(weights)); mode
            ) == reference
            @test BAT.smallest_credible_intervals(
                values, Weights(7 .* weights); mode
            ) == reference
            @test BAT.smallest_credible_intervals(
                repeat(values, inner = 2),
                Weights(reduce(vcat, ([w, w] for w in weights))); mode
            ) == reference

            for compressed_weights in (
                    [3, 1], Float32[3, 1], [3 // 4, 1 // 4], BigFloat[3, 1]
                )
                @test BAT.smallest_credible_intervals(
                    [0, 10], Weights(compressed_weights); mode
                ) == BAT.smallest_credible_intervals([0, 0, 0, 10]; mode)
            end
        end

        duplicate_values = [-0.0, 0.0, 1.0, 10.0, 100.0]
        duplicate_weights = [1, 2, 0, 1, 0]
        for mode in (:disjoint, :connected)
            intervals = BAT.smallest_credible_intervals(
                duplicate_values, Weights(duplicate_weights); mode
            )
            @test all(interval -> minimum(interval) == 0, intervals)
            @test _union_mass(duplicate_values, duplicate_weights, intervals) >= 28 // 41
        end
    end

    @testset "numeric routes" begin
        for weights in (
                [1.0, 1e-20],
                Float32[floatmax(Float32), 1],
                [floatmax(Float64), nextfloat(0.0)],
                [big(1) << 4096, big(1)],
                [28 // 41, 13 // 41],
                Real[prevfloat(28 / 41), 13 // 41],
            )
            for orientation in (identity, reverse)
                oriented_weights = orientation(weights)
                values = orientation([0, 10])
                _check_modes(values, oriented_weights, 1, 28 // 41)
            end
        end

        for weights in (
                [typemax(UInt128) ÷ 370 + 1, one(UInt128)],
                fill(typemax(UInt128) ÷ 2 + 1, 2),
                [68292682926829267 // 10^17, 31707317073170733 // 10^17],
            )
            _check_modes([0, 10], weights, 1, 28 // 41)
        end

        for T in (Float16, Float32, Float64, BigFloat)
            values = T.([-3, -1, 2, 8])
            for intervals in _check_modes(values, T.([0, 1, 2, 5]), 1, 28 // 41)
                @test all(interval -> minimum(interval) isa T, intervals)
                @test all(interval -> maximum(interval) isa T, intervals)
            end
        end
    end

    @testset "stored precision and width" begin
        stored_weights = setprecision(512) do
            scale = BigFloat(2)^400
            [28scale - 1, 13scale + 1]
        end
        low = setprecision(128) do
            [BAT.smallest_credible_intervals(
                [0, 10], Weights(stored_weights); mode
            ) for mode in (:disjoint, :connected)]
        end
        high = setprecision(512) do
            [BAT.smallest_credible_intervals(
                [0, 10], Weights(stored_weights); mode
            ) for mode in (:disjoint, :connected)]
        end
        @test low == high
        @test all(result -> _union_mass([0, 10], stored_weights, result) >= 28 // 41, low)
        low_grid = setprecision(128) do
            atoms = BAT._credible_atoms(
                [0, 10], BAT._credible_big_coefficients(Weights(stored_weights))
            )
            BAT._credible_grid(atoms, 41)
        end
        high_grid = setprecision(512) do
            atoms = BAT._credible_atoms(
                [0, 10], BAT._credible_big_coefficients(Weights(stored_weights))
            )
            BAT._credible_grid(atoms, 41)
        end
        @test low_grid == high_grid

        stored_values = setprecision(512) do
            epsilon = BigFloat(2)^-200
            [BigFloat(0), BigFloat(1) + epsilon, BigFloat(2) + epsilon]
        end
        connected = setprecision(128) do
            BAT.smallest_credible_intervals(
                stored_values, Weights([5, 20, 5]); mode = :connected
            )
        end
        @test connected == [ClosedInterval(stored_values[2], stored_values[3])]

        extreme_values = [-floatmax(Float64), 0.0, floatmax(Float64)]
        @test BAT.smallest_credible_intervals(
            extreme_values, Weights([5, 20, 5]); mode = :connected
        ) == [ClosedInterval(-floatmax(Float64), 0.0)]
        int_values = [typemin(Int128), Int128(0), typemax(Int128)]
        _check_modes(int_values, [5, 20, 5], 1, 28 // 41)
    end

    @testset "logarithmic weights" begin
        values = [0.0, 1.0, 10.0]
        weights = exp.(BAT.ULogarithmic, [-2.0, -1.0, 0.0])
        ordinary = BAT._canonical_rel_weights(weights)
        for mode in (:disjoint, :connected)
            expected = BAT.smallest_credible_intervals(values, Weights(ordinary); mode)
            @test BAT.smallest_credible_intervals(values, Weights(weights); mode) == expected
        end

        underflowed = exp.(BAT.ULogarithmic, [-1000.0, 0.0])
        for mode in (:disjoint, :connected)
            @test BAT.smallest_credible_intervals([0, 1], Weights(underflowed); mode) ==
                [ClosedInterval(1, 1)]
        end
        @test_throws ArgumentError BAT.smallest_credible_intervals(
            [0, 1], Weights(Real[weights[1], 1.0])
        )
    end

    @testset "sample and report wrappers" begin
        values = [-10.0, -9.0, 9.0, 10.0]
        weights = [20, 1, 1, 20]
        scalar_samples = DensitySampleVector(v = values, logd = zeros(4), weight = weights)
        @test length(BAT.smallest_credible_intervals(scalar_samples)) > 1
        @test length(BAT.smallest_credible_intervals(
            scalar_samples; mode = :connected
        )) == 1

        vector_values = [[x, 2x] for x in values]
        vector_samples = DensitySampleVector(
            v = vector_values, logd = zeros(4), weight = weights
        )
        for mode in (:disjoint, :connected)
            intervals = BAT.smallest_credible_intervals(vector_samples; mode)
            @test length(intervals) == 2
            @test all(parameter -> parameter isa Vector{<:ClosedInterval}, intervals)
        end
        @test length(only(BAT._marginal_table(scalar_samples).credible_intervals)) > 1
    end

    @testset "validation" begin
        for mode in (:disjoint, :connected)
            @test BAT.smallest_credible_intervals([1.0]; mode) ==
                [ClosedInterval(1.0, 1.0)]
        end
        @test_throws ArgumentError BAT.smallest_credible_intervals([0, 1]; mode = :other)
        @test_throws ArgumentError BAT.smallest_credible_intervals(Float64[])
        @test_throws ArgumentError BAT.smallest_credible_intervals(
            [0.0, 1.0], Weights([1.0])
        )
        for values in ([0.0, NaN], [0.0, Inf], [0.0, -Inf])
            @test_throws ArgumentError BAT.smallest_credible_intervals(values)
        end
        for weights in ([0.0, 0.0], [1.0, -1.0], [1.0, Inf])
            @test_throws ArgumentError BAT.smallest_credible_intervals(
                [0.0, 1.0], Weights(weights)
            )
        end
        @test_throws ArgumentError BAT.smallest_credible_intervals(
            [0.0, 1.0]; nsigma_equivalent = nextfloat(0.0)
        )
    end
end
