# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieUnboundedKDE
using BAT, Makie, Random, Statistics, Test

const EXT = Base.get_extension(BAT, :BATMakieExt)
const KD = EXT.KernelDensity
const TAU = 2.0^-20

# Differential reference for the inherited selection and dependency call. The
# independent moment and Gaussian references below supply separate correctness
# checks; this reference detects changes to admitted legacy grids/densities.
function legacy_kde(coords, weights)
    weights = weights ./ maximum(weights)
    hs = map(x -> allequal(weights) ? nothing : EXT._weighted_kde_bandwidth(x, weights), coords)
    data = length(coords) == 1 ? only(coords) : coords
    any(isnothing, hs) && return KD.kde(data; weights)
    return KD.kde(data; weights, bandwidth=length(coords) == 1 ? only(hs) : hs)
end

function centered_moments(values, weights; corrected=false)
    z = BigFloat.(values) .- BigFloat(first(values))
    w = BigFloat.(weights)
    total = sum(w)
    center = sum(w .* z) / total
    denominator = corrected ? total - total / length(w) : total
    variance = sum(w .* (z .- center).^2) / denominator
    return (mean=Float64(BigFloat(first(values)) + center), variance=Float64(variance))
end

function check_result(k)
    grids = hasproperty(k, :y) ? (k.x, k.y) : (k.x,)
    @test all(g -> all(isfinite, collect(Float64, g)), grids)
    @test all(g -> all(>(0), diff(collect(Float64, g))), grids)
    @test all(d -> isfinite(d) && d >= 0, k.density)
    @test maximum(k.density) > 0
    # BigFloat is an independent oracle, not production's peak/scale product.
    mass = sum(BigFloat(d) for d in k.density) * prod(BigFloat(step(g)) for g in grids)
    @test abs(mass - 1) <= 8TAU
    if hasproperty(k, :xedges)
        @test length(k.xedges) == length(k.x) + 1
        @test length(k.yedges) == length(k.y) + 1
        @test all(>(0), diff(collect(Float64, k.xedges)))
        @test all(>(0), diff(collect(Float64, k.yedges)))
    end
end

function check_legacy(k, old)
    @test k.x == old.x
    @test k.density == old.density
    hasproperty(k, :y) && (@test k.y == old.y)
end

# Direct Gaussian mixtures, conditional on the selected bandwidth. The periodic
# reference includes the retained FFT wrap tails. It has no tabulation, FFT,
# production interpolation, or normalization. Three image periods suffice for
# these four-bandwidth-padded fixtures; more distant images are negligible.
function check_gaussian_mixture(k, coords, weights)
    relative = weights ./ maximum(weights)
    hs = map(x -> allequal(relative) ? KD.default_bandwidth(x) :
        EXT._weighted_kde_bandwidth(x, relative), coords)
    grids = length(coords) == 1 ? (k.x,) : (k.x, k.y)
    strides = length(coords) == 1 ? (79,) : (29, 31)
    indices = map((g, stride) -> unique(vcat(collect(1:stride:length(g)), length(g))), grids, strides)
    xb, wb, hb = map(x -> BigFloat.(x), coords), BigFloat.(relative), BigFloat.(hs)
    periods = map(g -> BigFloat(step(g)) * length(g), grids)
    root_two_pi, total = sqrt(2big(pi)), sum(wb)
    direct_error, periodic_error = 0.0, 0.0
    peak = maximum(k.density)
    for index in Iterators.product(indices...)
        direct, periodic = big"0", big"0"
        for t in eachindex(wb)
            direct_term, periodic_term = wb[t], wb[t]
            for axis in eachindex(coords)
                delta = BigFloat(grids[axis][index[axis]]) - xb[axis][t]
                gaussian(offset) = exp(-((delta + offset) / hb[axis])^2 / 2) /
                    (root_two_pi * hb[axis])
                central = gaussian(0)
                direct_term *= central
                periodic_term *= central + gaussian(-periods[axis]) + gaussian(periods[axis])
            end
            direct += direct_term
            periodic += periodic_term
        end
        actual = k.density[index...]
        direct_error = max(direct_error, abs(Float64(direct / total) - actual) / peak)
        periodic_error = max(periodic_error, abs(Float64(periodic / total) - actual) / peak)
    end
    # Linear deposition's Gaussian interpolation bound, rather than the much
    # tighter arithmetic-admission budget, controls approximation to the mixture.
    interpolation_bound = sum((step(g) / h)^2 for (g, h) in zip(grids, hs)) /
        (8 * (2pi)^(length(coords) / 2) * prod(hs) * peak)
    @test periodic_error <= 1.01interpolation_bound + 8TAU
    @test direct_error < 3e-3
end

@testset "Unbounded KDE numerical admission" begin
    x, y = [-2.0,-1.0,0.0,1.0,2.0], [1.0,-2.0,2.0,0.0,-1.0]

    @testset "Exact legacy estimator, support spelling and Gaussian shape" begin
        for weights in (ones(5), [1.0,2.0,3.0,4.0,2.0], Float32[1,2,3,4,2])
            before = (copy(x), copy(y), copy(weights))
            curve = EXT._weighted_kde1d(x, weights)
            pair = EXT._weighted_kde2d(x, y, weights)
            @test length(curve.x) == 2048
            @test size(pair.density) == (256, 256)
            check_legacy(curve, legacy_kde((x,), weights))
            check_legacy(pair, legacy_kde((x,y), weights))
            check_result(curve)
            check_result(pair)
            for support in (nothing, (-Inf, Inf))
                same = EXT._weighted_kde1d(x, weights, support)
                @test same == curve
            end
            for support in (nothing, ((-Inf,Inf), (-Inf,Inf)))
                same = EXT._weighted_kde2d(x, y, weights, support)
                @test same == pair
            end
            if eltype(weights) == Float64
                check_gaussian_mixture(curve, (x,), weights)
                check_gaussian_mixture(pair, (x,y), weights)
            end
            @test isequal((x,y,weights), before)
        end
        # Coordinates need not provide mutable broadcast storage. Admission's
        # centered witness must work for a range just as for its materialization.
        range_x, range_y = range(-2.0, 2.0; length=5), range(3.0, -1.0; length=5)
        @test EXT._weighted_kde1d(range_x, ones(5)) == EXT._weighted_kde1d(collect(range_x), ones(5))
        @test EXT._weighted_kde2d(range_x, range_y, ones(5)) ==
            EXT._weighted_kde2d(collect(range_x), collect(range_y), ones(5))
    end

    @testset "Independent bandwidth moments and convolved moments" begin
        # Here IQR/1.34 is larger than either standard deviation. Thus the
        # selected bandwidth is determined independently from centered moments;
        # no production bandwidth or weighted-quantile helper is the oracle.
        values = [-2.0,-2.0,0.0,2.0,2.0]
        for weights in (ones(5), [1.0,2.0,1.0,2.0,1.0])
            population = centered_moments(values, weights)
            bandwidth_variance = centered_moments(values, weights; corrected=allequal(weights)).variance
            effective_count = Float64(sum(BigFloat.(weights))^2 / sum(abs2, BigFloat.(weights)))
            expected_h = 0.9sqrt(bandwidth_variance) * effective_count^(-0.2)
            curve = EXT._weighted_kde1d(values, weights)
            actual_h = (minimum(values) - first(curve.x)) / 4
            @test actual_h ≈ expected_h rtol=3e-15
            expected_variance = population.variance + expected_h^2
            actual_variance = sum(curve.density .* (collect(curve.x) .- population.mean).^2) * step(curve.x)
            @test actual_variance ≈ expected_variance rtol=0.002
            pair = EXT._weighted_kde2d(values, -values, weights)
            covariance = sum(pair.density .* ((collect(pair.x) .- population.mean) *
                (collect(pair.y) .+ population.mean)')) * step(pair.x) * step(pair.y)
            @test covariance ≈ -population.variance rtol=0.002
        end
        # A normal, finite Float32 variance can still be wrong. The same
        # represented coordinates in Float64 retain a valid legacy estimate.
        c = Float32(2^24)
        bad = Float32[c,c,c,c,c+4]
        reference_sd = sqrt(centered_moments(bad, ones(5); corrected=true).variance)
        @test isfinite(std(bad)) && std(bad) > 0
        @test abs(std(bad) / reference_sd - 1) > 0.1
        @test_throws ArgumentError EXT._weighted_kde1d(bad, ones(5))
        @test_throws ArgumentError EXT._weighted_kde2d(bad, y, ones(5))
        valid = EXT._weighted_kde1d(Float64.(bad), ones(5))
        check_result(valid)
        check_legacy(valid, legacy_kde((Float64.(bad),), ones(5)))
    end

    @testset "Singletons, constants and original positive population" begin
        for (values, other, weights) in (([0.0], [3.0], [1.0]),
            (fill(2.0,5), fill(3.0,5), ones(5)),
            (fill(2.0,5), y, [1.0,2.0,3.0,4.0,2.0]))
            curve = EXT._weighted_kde1d(values, weights)
            pair = EXT._weighted_kde2d(values, other, weights)
            check_result(curve)
            check_result(pair)
            check_legacy(curve, legacy_kde((values,), weights))
            check_legacy(pair, legacy_kde((values,other), weights))
        end
        for weights in (ones(5), [1.0,2.0,3.0,4.0,2.0])
            curve = EXT._weighted_kde1d(x, weights)
            pair = EXT._weighted_kde2d(x, y, weights)
            ignored_x = vcat(x, NaN, Inf, -Inf, 1e300)
            ignored_y = vcat(y, Inf, NaN, -1e300, -Inf)
            ignored_w = vcat(weights, zeros(4))
            @test EXT._weighted_kde1d(ignored_x, ignored_w) == curve
            @test EXT._weighted_kde2d(ignored_x, ignored_y, ignored_w) == pair
        end
        tiny_weights = [1e308, nextfloat(0.0)]
        for invalid in (NaN, Inf, -Inf)
            @test_throws ArgumentError EXT._weighted_kde1d([0.0,invalid], tiny_weights)
            @test_throws ArgumentError EXT._weighted_kde2d([0.0,1.0], [0.0,invalid], tiny_weights)
        end
        wider_weights = [big"1e400", big"1e-400", big"1e-401"]
        # Preserve the retained weight's arithmetic type in this exact oracle.
        @test EXT._weighted_kde1d([0.1,0.4,0.8], wider_weights) == EXT._weighted_kde1d([0.1], [big"1"])
        @test EXT._weighted_kde2d([0.1,0.4,0.8], [0.2,0.7,0.9], wider_weights) ==
            EXT._weighted_kde2d([0.1], [0.2], [big"1"])
        @test_throws ArgumentError EXT._weighted_kde1d([big"0.1",big"1e400",big"0.8"], wider_weights)
        # Several retained positive rows with an effective count rounded to one
        # must not silently acquire the true-singleton/default fallback.
        concentrated = [1.0,1e-100,1e-100,1e-100,1e-100]
        @test_throws ArgumentError EXT._weighted_kde1d(x, concentrated)
        @test_throws ArgumentError EXT._weighted_kde2d(x, y, concentrated)
    end

    @testset "Avoidable source-precision weight loss" begin
        for T in (Float16, Float32)
            smallest = nextfloat(zero(T))
            for weights in (T[smallest,3], Real[smallest,T(3)], Real[smallest,3])
                @test weights[1] > 0 && Float64(weights[1]) / Float64(weights[2]) > 0
                @test weights[1] / weights[2] == 0
                @test_throws ArgumentError EXT._weighted_kde1d([0.0,1.0], weights)
                @test_throws ArgumentError EXT._weighted_kde2d([0.0,1.0], [1.0,0.0], weights)
                # The shared population boundary also protects finite support.
                @test_throws ArgumentError EXT._weighted_kde1d([0.0,1.0], weights, (0.0,1.0))
                @test_throws ArgumentError EXT._weighted_kde2d([0.0,1.0], [1.0,0.0], weights,
                    ((0.0,1.0),(0.0,1.0)))
            end
            # The check must not become a blanket prohibition on these types.
            check_legacy(EXT._weighted_kde1d(x, fill(T(2),5)), legacy_kde((x,), fill(T(2),5)))
        end
    end

    @testset "Variance, physical grids and extreme dependency arithmetic" begin
        for values in (1e-170 .* x, 1e170 .* x, 1e12 .+ 1e-3 .* x, 1e14 .+ x)
            @test_throws ArgumentError EXT._weighted_kde1d(values, ones(5))
            @test_throws ArgumentError EXT._weighted_kde2d(values, y, ones(5))
        end
        # Common scales expose squared cell area; reciprocal axis scales can
        # preserve area while breaking Gaussian variance/frequency arithmetic.
        for (sx,sy) in ((1e-100,1e-100), (1e100,1e100), (1e-155,1e155), (1e155,1e-155))
            @test_throws ArgumentError EXT._weighted_kde2d(sx .* x, sy .* y, ones(5))
        end
        # Admission is not a new adaptive-grid policy: this ordinary fixture
        # keeps KernelDensity's fixed 256 x 256 estimate.
        rng = MersenneTwister(31)
        a, b = randn(rng,1000), randn(rng,1000)
        ordinary = EXT._weighted_kde2d(a, b, ones(1000))
        check_result(ordinary)
        check_legacy(ordinary, legacy_kde((a,b), ones(1000)))
    end

    @testset "Binary translations, units, weights and axis order" begin
        weights = [1.0,2.0,3.0,4.0,2.0]
        base = EXT._weighted_kde1d(x, weights)
        pair = EXT._weighted_kde2d(x, y, weights)
        for offset in (2.0^10, 2.0^20)
            shifted = EXT._weighted_kde1d(offset .+ x, weights)
            shifted2 = EXT._weighted_kde2d(offset .+ x, offset .+ y, weights)
            @test maximum(abs.(collect(shifted.x) .- offset .- collect(base.x))) <= 2eps(offset)
            @test maximum(abs.(shifted.density .- base.density)) / maximum(base.density) < 8TAU
            @test maximum(abs.(shifted2.density .- pair.density)) / maximum(pair.density) < 8TAU
        end
        for exponent in (-100, -20, 20, 100)
            scale = 2.0^exponent
            scaled = EXT._weighted_kde1d(scale .* x, weights)
            @test scaled.x ./ scale == base.x
            @test scaled.density .* scale ≈ base.density rtol=2e-12
            anisotropic = EXT._weighted_kde2d(scale .* x, y ./ scale, weights)
            @test anisotropic.x ./ scale == pair.x
            @test anisotropic.y .* scale == pair.y
            @test anisotropic.density ≈ pair.density rtol=2e-12
            check_result(anisotropic)
        end
        for scale in (2.0^-40, 2.0^40)
            common = EXT._weighted_kde2d(scale .* x, scale .* y, weights)
            @test common.density .* scale^2 ≈ pair.density rtol=2e-12
            check_result(common)
        end
        # These physical densities are numerically valid. Numeric Heatmap color
        # representability is a distinct adapter contract tested elsewhere.
        check_result(EXT._weighted_kde2d(1e20 .* x, 1e20 .* y, weights))
        for factor in (2.0^-500, 2.0^500)
            @test EXT._weighted_kde1d(x, weights .* factor) == base
            @test EXT._weighted_kde2d(x, y, weights .* factor) == pair
        end
        single_weights = Float32.(weights)
        @test EXT._weighted_kde2d(x, y, single_weights) == EXT._weighted_kde2d(x, y, 8f0 .* single_weights)
        swapped = EXT._weighted_kde2d(y, x, weights)
        @test swapped.xedges == pair.yedges && swapped.yedges == pair.xedges
        @test swapped.density ≈ permutedims(pair.density) rtol=2e-12
    end

    @testset "BigFloat arithmetic still deposits into Float64 FFT buffers" begin
        # The unguarded estimator returns unit lattice mass here, but rounds
        # the remote population's probability from 1/19 to 1/10. Checking only
        # BigFloat factors and final total mass cannot detect that distortion.
        delta = BigFloat(nextfloat(0.0))
        area = inv(10delta)
        remote = 255sqrt(area)
        values = BigFloat[-4,-3,-2,-1,0,1,2,3,4,remote]
        weights = BigFloat[1,1,1,1,1,1,1,1,1,0.5]
        unguarded = KD.kde((values, values); weights,
            bandwidth=(EXT._weighted_kde_bandwidth(values, weights),
                EXT._weighted_kde_bandwidth(values, weights)))
        cell_area = step(unguarded.x) * step(unguarded.y)
        @test sum(BigFloat.(unguarded.density)) * cell_area ≈ 1 rtol=TAU
        actual_remote = sum(BigFloat.(unguarded.density[end-2:end,end-2:end])) * cell_area
        @test abs(actual_remote - weights[end]/sum(weights)) > 0.01
        @test_throws ArgumentError EXT._weighted_kde2d(values, values, weights)
        # Wider intermediate arithmetic is useful when the native buffer scale
        # remains representable; do not impose a blanket Float64 factor limit.
        scale = big"1e150"
        large = EXT._weighted_kde2d(scale .* x, scale .* y, ones(5))
        ordinary = EXT._weighted_kde2d(x, y, ones(5))
        check_result(large)
        restored = Float64.(BigFloat.(large.density) .* scale^2)
        @test maximum(abs.(restored .- ordinary.density)) / maximum(ordinary.density) < 8TAU
    end

    @testset "Synthetic dependency-result boundary faults" begin
        # Arbitrary faulty dependency outputs are not constructible through the
        # fixed public estimator. Test these contracts directly without mocking
        # KernelDensity or changing global methods.
        grid = range(-2.0, 2.0; length=5)
        @test EXT._kde_check_unbounded_axis([nextfloat(-2.0), -1.0, prevfloat(2.0), 2.0], grid, 0.5) == step(grid)
        for value in (prevfloat(-2.0), -2.0, nextfloat(2.0))
            @test_throws ArgumentError EXT._kde_check_unbounded_axis([value], grid, 0.5)
        end
        repeated = range(1e16, 1e16+4; length=2048)
        @test length(unique(collect(repeated))) < length(repeated)
        @test_throws ArgumentError EXT._kde_check_unbounded_axis([1e16+2], repeated, 1.0)
        irregular = range(1e12, 1e12+1; length=2048)
        @test all(>(0), diff(collect(irregular)))
        @test_throws ArgumentError EXT._kde_check_unbounded_axis([1e12+0.5], irregular, 0.1)

        @test isnothing(EXT._kde_check_unbounded_density([0.5,0.5], (1.0,)))
        @test isnothing(EXT._kde_check_unbounded_density(fill(2.5e-301,2,2), (1e200,1e100)))
        for density in ([NaN,1.0], [Inf,1.0], [-eps(),1.0], zeros(2), [1.0,1.0])
            @test_throws ArgumentError EXT._kde_check_unbounded_density(density, (1.0,))
        end
        @test isnothing(EXT._kde_check_unbounded_edges([0.0,1.0,2.0+TAU,3.0], 1.0))
        for edges in ([0.0,1.0,1.0,2.0], [0.0,1.0,2.0+8TAU,3.0], [0.0,1.0,Inf])
            @test_throws ArgumentError EXT._kde_check_unbounded_edges(edges, 1.0)
        end
        # Positive, regular native edges can collapse only when emitted as
        # Float64. Checking the native vector alone misses this fault.
        precise_edges = [BigFloat(2)^80 + i for i in 0:3]
        @test all(==(1), diff(precise_edges))
        @test length(unique(Float64.(precise_edges))) == 1
        @test_throws ArgumentError EXT._kde_check_unbounded_edges(precise_edges, big"1")
        for scale in (big"1e160", big"1e-160")
            @test_throws ArgumentError EXT._kde_check_unbounded_tabulation([1.0], (scale, scale))
        end
    end
end

end # module
