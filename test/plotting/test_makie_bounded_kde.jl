# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieBoundedKDE
using BAT, Makie, Distributions, Test

const EXT = Base.get_extension(BAT, :BATMakieExt)
const STANDARD_NORMAL = Normal()
const ROOT_TWO_PI = sqrt(2pi)

# Independent Gaussian-mixture oracle. The selected bandwidth is part of the
# estimator's contract; this reference does not use FFT tabulation, production
# interpolation, or production normalization/region helpers.
function gaussian_interval(a, b)
    a == b && return 0.0
    if isfinite(a) && isfinite(b) && b - a < 1e-4
        # Subtracting two Float64 CDF values loses the narrow-support oracle.
        # Composite Simpson in 256-bit arithmetic integrates a smooth Gaussian
        # over this tiny interval; its discretization error is negligible here.
        return setprecision(BigFloat, 256) do
            lo, hi = BigFloat(a), BigFloat(b)
            step = (hi - lo) / 16
            f(t) = exp(-t * t / 2) / sqrt(2big(pi))
            result = f(lo) + f(hi)
            for i in 1:15
                result += (isodd(i) ? 4 : 2) * f(lo + i * step)
            end
            Float64(result * step / 3)
        end
    end
    return a >= 0 ? ccdf(STANDARD_NORMAL, a) - ccdf(STANDARD_NORMAL, b) :
        cdf(STANDARD_NORMAL, b) - cdf(STANDARD_NORMAL, a)
end

function image_locations(value, h, support)
    lo, hi = support
    result = [value]
    isfinite(lo) && value - lo < 4h && push!(result, 2lo - value)
    isfinite(hi) && hi - value < 4h && push!(result, 2hi - value)
    return result
end

function reference1d(grid, values, weights, h, support)
    weights = weights ./ maximum(weights)
    result = zeros(length(grid))
    mass = 0.0
    lo, hi = extrema(grid)
    for (value, weight) in zip(values, weights)
        weight == 0 && continue
        for center in image_locations(value, h, support)
            mass += weight * gaussian_interval((lo - center) / h, (hi - center) / h)
            for j in eachindex(grid)
                z = (grid[j] - center) / h
                result[j] += weight * exp(-z * z / 2) / (ROOT_TWO_PI * h)
            end
        end
    end
    return result ./ mass
end

function reference2d(result, x, y, weights, hx, hy, support)
    weights = weights ./ maximum(weights)
    gridx = [a + (b - a) / 2 for (a, b) in zip(result.xedges[1:end-1], result.xedges[2:end])]
    gridy = [a + (b - a) / 2 for (a, b) in zip(result.yedges[1:end-1], result.yedges[2:end])]
    density = zeros(length(gridx), length(gridy))
    mass = 0.0
    for i in eachindex(x, y, weights)
        weights[i] == 0 && continue
        for cx in image_locations(x[i], hx, support[1]), cy in image_locations(y[i], hy, support[2])
            px = gaussian_interval((first(result.xedges) - cx) / hx, (last(result.xedges) - cx) / hx)
            py = gaussian_interval((first(result.yedges) - cy) / hy, (last(result.yedges) - cy) / hy)
            mass += weights[i] * px * py
            for iy in eachindex(gridy), ix in eachindex(gridx)
                zx, zy = (gridx[ix] - cx) / hx, (gridy[iy] - cy) / hy
                # Divide successively: hx*hy can be unusable in other fixtures.
                density[ix, iy] += (weights[i] * exp(-(zx*zx + zy*zy)/2) / (2pi) / hx) / hy
            end
        end
    end
    return density ./ mass
end

curve_area(x, y) = sum((b - a) * (u / 2 + v / 2) for (a, b, u, v) in
    zip(x[1:end-1], x[2:end], y[1:end-1], y[2:end]))

function curve_second_moment(x, y, center)
    # Simpson's rule is exact for x^2 times a linear segment.
    return sum((b-a) / 6 * ((a-center)^2*u +
        4 * (a + (b-a)/2 - center)^2 * (u/2+v/2) + (b-center)^2*v)
        for (a,b,u,v) in zip(x[1:end-1],x[2:end],y[1:end-1],y[2:end]))
end

cell_masses(result) = result.density .* (diff(result.xedges) * diff(result.yedges)')

function polygon_area(points)
    # Shoelace area of the actual primitive, translated before multiplying to
    # avoid subtracting huge products for a narrow interval at a large offset.
    origin = first(points)
    area = 0.0
    for i in eachindex(points)
        j = i == lastindex(points) ? firstindex(points) : i+1
        area += (points[i][1]-origin[1]) * (points[j][2]-origin[2]) -
            (points[j][1]-origin[1]) * (points[i][2]-origin[2])
    end
    return abs(area)/2
end

function check_curve(result, support)
    @test length(result.x) == length(result.density) >= 2
    @test all(isfinite, result.x)
    @test all(>(0), diff(result.x))
    @test all(d -> isfinite(d) && d >= 0, result.density)
    @test first(result.x) >= support[1]
    @test last(result.x) <= support[2]
    @test curve_area(result.x, result.density) ≈ 1 atol=2e-12
end

function check_cells(result, support)
    @test size(result.density) == (length(result.xedges)-1, length(result.yedges)-1)
    @test length(result.x) == size(result.density, 1)
    @test length(result.y) == size(result.density, 2)
    @test all(isfinite, result.xedges) && all(isfinite, result.yedges)
    @test all(>(0), diff(result.xedges)) && all(>(0), diff(result.yedges))
    @test all(d -> isfinite(d) && d >= 0, result.density)
    @test first(result.xedges) >= support[1][1] && last(result.xedges) <= support[1][2]
    @test first(result.yedges) >= support[2][1] && last(result.yedges) <= support[2][2]
    @test sum(cell_masses(result)) ≈ 1 atol=2e-12
end

function captured_error(f)
    try
        f()
        return nothing
    catch err
        return err
    end
end

@testset "Bounded KDE estimator and probability geometry" begin
    @testset "Concentrated data retains selected Gaussian shape and moments" begin
        x = 0.5 .+ 1e-6 .* collect(range(-2.0, 2.0; length=101))
        y = reverse(x)
        weights = ones(length(x))
        h = EXT._kde_bandwidth(x, weights)
        expected_variance = sum(abs2, x .- 0.5) / length(x) + h^2
        result = EXT._weighted_kde1d(x, weights, (0.0, 1.0))
        check_curve(result, (0.0, 1.0))
        reference = reference1d(result.x, x, weights, h, (0.0, 1.0))
        @test maximum(abs.(result.density .- reference)) / maximum(reference) < 0.005
        @test curve_second_moment(result.x, result.density, 0.5) ≈ expected_variance rtol=0.005
        # Distant hard bounds must not consume resolution or move the estimate.
        wider = EXT._weighted_kde1d(x, weights, (-1e8, 1e8))
        @test wider.x == result.x
        @test wider.density ≈ result.density rtol=2e-12

        pair = EXT._weighted_kde2d(x, y, weights, ((0.0, 1.0), (0.0, 1.0)))
        check_cells(pair, ((0.0, 1.0), (0.0, 1.0)))
        hx, hy = EXT._kde_bandwidth(x, weights), EXT._kde_bandwidth(y, weights)
        reference = reference2d(pair, x, y, weights, hx, hy, ((0.0, 1.0), (0.0, 1.0)))
        @test maximum(abs.(pair.density .- reference)) / maximum(reference) < 0.01
        masses = cell_masses(pair)
        centers = [a + (b-a)/2 for (a,b) in zip(pair.xedges[1:end-1], pair.xedges[2:end])]
        # Constant cells contribute their within-cell variance as well.
        variance = sum(masses .* (((centers .- 0.5).^2 .+ diff(pair.xedges).^2 ./ 12)))
        @test variance ≈ expected_variance rtol=0.005
        ycenters = [a + (b-a)/2 for (a,b) in zip(pair.yedges[1:end-1], pair.yedges[2:end])]
        # Independent axis kernels add no cross-covariance to the population.
        expected_covariance = sum((x .- 0.5) .* (y .- 0.5)) / length(x)
        covariance = sum(masses .* ((centers .- 0.5) * (ycenters .- 0.5)'))
        @test covariance ≈ expected_covariance rtol=0.005

        # Move an interior observation by a non-integral fraction of h. This
        # changes its phase relative to tabulation nodes without a new tolerance
        # or a reference derived from the implementation's tabulated density.
        perturbed_x, perturbed_y = copy(x), copy(y)
        perturbed_x[51] += 0.37hx
        perturbed_y[51] -= 0.23hy
        hx2, hy2 = EXT._kde_bandwidth(perturbed_x,weights), EXT._kde_bandwidth(perturbed_y,weights)
        changed = EXT._weighted_kde1d(perturbed_x,weights,(0.0,1.0))
        expected = reference1d(changed.x,perturbed_x,weights,hx2,(0.0,1.0))
        @test maximum(abs.(changed.density .- expected)) / maximum(expected) < 0.005
        changed2 = EXT._weighted_kde2d(perturbed_x,perturbed_y,weights,((0.0,1.0),(0.0,1.0)))
        expected2 = reference2d(changed2,perturbed_x,perturbed_y,weights,hx2,hy2,((0.0,1.0),(0.0,1.0)))
        @test maximum(abs.(changed2.density .- expected2)) / maximum(expected2) < 0.01
    end

    @testset "Finite images at boundaries and corners" begin
        weights = [1.0, 2.0, 3.0, 1.0]
        for (x, support) in (([0.0, 0.015, 0.06, 0.12], (0.0, Inf)),
            ([0.88, 0.94, 0.985, 1.0], (-Inf, 1.0)),
            ([0.0, 0.02, 0.08, 0.12], (0.0, 0.12)),
            ([0.1, 0.12, 0.75, 0.82], (0.0, 1.0)))
            result = EXT._weighted_kde1d(x, weights, support)
            check_curve(result, support)
            expected = reference1d(result.x, x, weights, EXT._kde_bandwidth(x, weights), support)
            @test maximum(abs.(result.density .- expected)) / maximum(expected) < 0.005
        end
        x, y = [0.0, 0.02, 0.08, 0.12], [0.0, 0.03, 0.1, 0.15]
        for support in (((0.0, Inf), (0.0, Inf)), ((0.0, 0.12), (0.0, 0.15)))
            result = EXT._weighted_kde2d(x, y, weights, support)
            check_cells(result, support)
            expected = reference2d(result, x, y, weights,
                EXT._kde_bandwidth(x, weights), EXT._kde_bandwidth(y, weights), support)
            @test maximum(abs.(result.density .- expected)) / maximum(expected) < 0.01
        end
        # Two separated clusters with unequal weights exercise a genuinely
        # multimodal joint density, including the selected finite reflections.
        x, y = [0.12,0.16,0.22,0.76,0.82,0.88], [0.15,0.22,0.17,0.75,0.8,0.86]
        weights = [1.0,2.0,1.0,3.0,2.0,1.0]
        support = ((0.0,1.0),(0.0,1.0))
        result = EXT._weighted_kde2d(x,y,weights,support)
        check_cells(result,support)
        expected = reference2d(result,x,y,weights,
            EXT._kde_bandwidth(x,weights),EXT._kde_bandwidth(y,weights),support)
        @test maximum(abs.(result.density .- expected)) / maximum(expected) < 0.01
    end

    @testset "Support narrower than an FFT interval" begin
        lo, hi = 0.5, 0.5 + 1e-10
        x, weights = [lo], [1.0]
        result = EXT._weighted_kde1d(x, weights, (lo, hi))
        check_curve(result, (lo, hi))
        @test first(result.x) == lo && last(result.x) == hi
        reference = reference1d(result.x, x, weights, EXT._kde_bandwidth(x, weights), (lo, hi))
        @test maximum(abs.(result.density .- reference)) / maximum(reference) < 0.005
        region = EXT._kde_curve_region(result.x, result.density, 0.68)
        @test !isempty(region.components)
        @test sum(curve_area(c.x, c.y) for c in region.components) > 0

        pair = EXT._weighted_kde2d(x, x, weights, ((lo, hi), (lo, hi)))
        check_cells(pair, ((lo, hi), (lo, hi)))
        @test pair.xedges == [lo, hi] && pair.yedges == [lo, hi]
        @test size(pair.density) == (1, 1)
        @test only(pair.density) ≈ inv(hi-lo)^2 rtol=2e-12
        # Adjacent representable endpoints still define positive area.
        adjacent = EXT._weighted_kde1d(x, weights, (lo, nextfloat(lo)))
        check_curve(adjacent, (lo, nextfloat(lo)))
        @test adjacent.x == [lo, nextfloat(lo)]
    end

    @testset "Cell probabilities use actual areas in safe relative units" begin
        # Heights 2 and 1 on cells of areas 2 and 6 give masses 4 and 6.
        # Ranking density and accumulating probability are different operations.
        density = reshape([2.0,1.0],2,1)
        result = EXT._kde_cell_mass(density,[0.0,1.0,4.0],[0.0,2.0])
        @test vec(result.probabilities) ≈ [0.4,0.6] atol=2e-15
        # Nominally uniform edges can acquire unequal *actual* widths near a
        # large offset; these two represented widths are exactly 2 and 4.
        rounded = EXT._kde_cell_mass(density,[1e16,1e16+2,1e16+6],[0.0,2.0])
        @test vec(rounded.probabilities) ≈ [0.5,0.5] atol=2e-15
        extreme = EXT._kde_cell_mass(reshape([1.7e308,0.85e308],2,1),
            [0.0,1e-300,4e-300],[0.0,2e300])
        @test vec(extreme.probabilities) ≈ [0.4,0.6] atol=2e-15
        @test all(isfinite,extreme.probabilities)
    end

    @testset "Weights, zero-mass rows, units and output ownership" begin
        x, y, w = [0.1,0.2,0.4,0.8], [0.15,0.3,0.7,0.8], [1.0,2.0,3.0,4.0]
        support = (0.0, 1.0)
        before = (copy(x), copy(y), copy(w))
        base = EXT._weighted_kde1d(x, w, support)
        pair = EXT._weighted_kde2d(x, y, w, (support, support))
        for factor in (1e-250, 1e250)
            scaled = EXT._weighted_kde1d(x, w .* factor, support)
            @test scaled.x ≈ base.x rtol=1e-12
            @test scaled.density ≈ base.density rtol=1e-11
            scaled2 = EXT._weighted_kde2d(x, y, w .* factor, (support, support))
            @test scaled2.xedges ≈ pair.xedges rtol=1e-12
            @test scaled2.yedges ≈ pair.yedges rtol=1e-12
            @test scaled2.density ≈ pair.density rtol=1e-11
        end
        for weights in (w, ones(4))
            positive = EXT._weighted_kde1d(x, weights, support)
            zero_outlier = EXT._weighted_kde1d(vcat(x, 1e300), vcat(weights, 0.0), support)
            @test zero_outlier.x == positive.x
            @test zero_outlier.density == positive.density
            positive2 = EXT._weighted_kde2d(x, y, weights, (support, support))
            zero_outlier2 = EXT._weighted_kde2d(vcat(x, 1e300), vcat(y, -1e300), vcat(weights, 0.0), (support, support))
            @test zero_outlier2.xedges == positive2.xedges
            @test zero_outlier2.yedges == positive2.yedges
            @test zero_outlier2.density == positive2.density
        end
        # Wider weight types can retain a ratio that the Float64 convolution
        # cannot represent. Such rows must not survive solely in bandwidth
        # selection after their numerical convolution mass has become zero.
        wider_weights = [big"1e400",big"1e-400",big"1e-401"]
        converted = EXT._weighted_kde1d([0.1,0.2,0.3],wider_weights,support)
        retained = EXT._weighted_kde1d([0.1],[1.0],support)
        @test converted.x == retained.x
        @test converted.density == retained.density
        # Nonconstant samples have scale-covariant selected bandwidths. The
        # inherited unit-dependent constant-data fallback is not tested as such.
        for scale in (1e-90, 1e90)
            scaled = EXT._weighted_kde1d(x .* scale, w, (0.0, scale))
            @test scaled.x ./ scale ≈ base.x rtol=2e-12
            @test scaled.density .* scale ≈ base.density rtol=2e-10
        end
        anisotropic = EXT._weighted_kde2d(x .* 1e-90, y .* 1e90, w, ((0.0,1e-90),(0.0,1e90)))
        @test anisotropic.xedges ./ 1e-90 ≈ pair.xedges rtol=2e-12
        @test anisotropic.yedges ./ 1e90 ≈ pair.yedges rtol=2e-12
        @test anisotropic.density ≈ pair.density rtol=2e-10
        # An individually unnormalizable microscopic axis can still have a
        # representable joint density when paired with a broad second axis.
        subnormal = EXT._weighted_kde2d(zeros(4),1e100.*x,ones(4),
            ((0.0,1e-310),(0.0,1e100)))
        check_cells(subnormal,((0.0,1e-310),(0.0,1e100)))
        @test subnormal.xedges == [0.0,1e-310]
        shifted_x = 1e100 .+ 1e90 .* x
        shifted_support = (1e100,1e100+1e90)
        shifted = EXT._weighted_kde1d(shifted_x,ones(4),shifted_support)
        check_curve(shifted,shifted_support)
        reference = reference1d(shifted.x,shifted_x,ones(4),
            EXT._kde_bandwidth(shifted_x,ones(4)),shifted_support)
        @test maximum(abs.(shifted.density .- reference)) / maximum(reference) < 0.005
        swapped = EXT._weighted_kde2d(y, x, w, (support, support))
        @test swapped.xedges == pair.yedges && swapped.yedges == pair.xedges
        @test swapped.density ≈ permutedims(pair.density) rtol=2e-12
        @test isequal((x,y,w), before)
        again = EXT._weighted_kde1d(x, w, support)
        @test base.x !== again.x && base.density !== again.density
        pair_again = EXT._weighted_kde2d(x, y, w, (support, support))
        @test pair.xedges !== pair_again.xedges && pair.density !== pair_again.density
    end

    @testset "Invalid input and excessive resolution fail explicitly" begin
        x, w = [0.1, 0.2, 0.4], ones(3)
        for support in ((NaN,1.0),(0.0,NaN),(1.0,0.0),(0.0,0.0),(Inf,Inf),(-Inf,-Inf))
            @test_throws ArgumentError EXT._weighted_kde1d(x, w, support)
            @test_throws ArgumentError EXT._weighted_kde2d(x, x, w, (support,(0.0,1.0)))
        end
        for bad_weights in ([1.0,-1.0,1.0],[1.0,NaN,1.0],[1.0,Inf,1.0])
            @test_throws ArgumentError EXT._weighted_kde1d(x, bad_weights, (0.0,1.0))
            @test_throws ArgumentError EXT._weighted_kde2d(x, x, bad_weights, ((0.0,1.0),(0.0,1.0)))
        end
        for bad_x in ([0.1,NaN,0.4],[0.1,Inf,0.4],[-0.1,0.2,0.4],[0.1,0.2,1.1])
            @test_throws ArgumentError EXT._weighted_kde1d(bad_x, w, (0.0,1.0))
            @test_throws ArgumentError EXT._weighted_kde2d(x, bad_x, w, ((0.0,1.0),(0.0,1.0)))
        end
        # Validation precedes dropping numerically underflowed relative weights.
        @test_throws ArgumentError EXT._weighted_kde1d([0.2,2.0], [1e308,nextfloat(0.0)], (0.0,1.0))
        @test_throws ArgumentError EXT._weighted_kde1d(x, w[1:2], (0.0,1.0))
        @test_throws ArgumentError EXT._weighted_kde2d(x, x[1:2], w, ((0.0,1.0),(0.0,1.0)))
        @test_throws ArgumentError EXT._weighted_kde1d(x, w, (BigFloat(-1e300), big"1e400"))
        @test_throws ArgumentError EXT._weighted_kde1d(x,w,("a","z"))
        @test_throws ArgumentError EXT._weighted_kde2d(x,x,w,((0.0,1.0),))
        @test_throws ArgumentError EXT._weighted_kde2d(x,x,w,((0.0,1.0),(0.0,1.0),(0.0,1.0)))

        # The planner's declared resolution limits can be checked without
        # allocating their FFT arrays. Counts that would overflow Int must
        # receive the same deliberate budget failure, not a conversion error.
        @test EXT._kde_grid_size((65536-1)/16,16,2048,65536) == 65536
        @test EXT._kde_grid_size((32768-1)/16,16,2048,65536) == 32768
        @test_throws ArgumentError EXT._kde_grid_size(65536/16,16,2048,65536)
        @test EXT._kde_grid_size((8192-1)/8,8,256,8192) == 8192
        @test_throws ArgumentError EXT._kde_grid_size(8192/8,8,256,8192)
        @test_throws ArgumentError EXT._kde_grid_size(floatmax(Float64),16,2048,65536)
        @test_throws ArgumentError EXT._kde_grid_size(Inf,8,256,8192)

        # A tiny but positive satellite is real mass: it must not be discarded
        # or silently drawn on a grid too coarse for the selected bandwidth.
        satellite = [-100.0,-100.0+1e-5,100.0,100.0+1e-5]
        weights = [0.499995,0.499995,5e-6,5e-6]
        err = captured_error(() -> EXT._weighted_kde1d(satellite,weights,(-200.0,200.0)))
        @test err isa Exception
        @test occursin(r"(?i)grid|resolution|budget|limit|cap", sprint(showerror,err))
        # Each axis is feasible below the axis cap, but the Cartesian grid
        # exceeds the total budget. This is independent of process memory.
        satellite2 = [0.0,1.0,100.0,101.0]
        err = captured_error(() -> EXT._weighted_kde2d(satellite2,satellite2,weights,((-1.0,102.0),(-1.0,102.0))))
        @test err isa Exception
        @test occursin(r"(?i)grid|resolution|budget|limit|cap", sprint(showerror,err))
        # Centering does not make an h/16 physical grid below one ULP possible.
        err = captured_error(() -> EXT._weighted_kde1d([1e16,1e16+2,1e16+4],ones(3),(1e16-10,1e16+10)))
        @test err isa Exception
        err = captured_error(() -> EXT._weighted_kde1d([0.0],[1.0],(0.0,1e-310)))
        @test err isa Exception
        @test occursin("density", sprint(showerror,err))
    end

    @testset "Continuous credible regions integrate their rendered geometry" begin
        x, density = [0.0,1.0,2.0], [0.0,1.0,0.0]
        for p in (0.1,0.5,0.9)
            result = EXT._kde_curve_region(x, density, p)
            threshold = sqrt(1-p)
            @test result.threshold ≈ threshold atol=2e-13
            @test result.achieved ≈ p atol=2e-13
            @test length(result.components) == 1
            component = only(result.components)
            @test first(component.x) ≈ threshold atol=2e-13
            @test last(component.x) ≈ 2-threshold atol=2e-13
            @test curve_area(component.x,component.y) ≈ p atol=2e-13
        end
        narrow = only(EXT._kde_curve_region(x,density,0.1).components)
        broad = only(EXT._kde_curve_region(x,density,0.9).components)
        @test first(broad.x) < first(narrow.x) < last(narrow.x) < last(broad.x)
        # No sampled vertex lies at the crossing in this increasing two-node
        # curve. A node mask alone would produce a zero-area filled polygon.
        result = EXT._kde_curve_region([0.0,1.0],[0.0,2.0],0.5)
        @test result.threshold ≈ sqrt(2.0) atol=2e-13
        @test first(only(result.components).x) ≈ inv(sqrt(2.0)) atol=2e-13
        @test curve_area(only(result.components).x,only(result.components).y) ≈ 0.5 atol=2e-13

        plateau = EXT._kde_curve_region([0.0,1.0,2.0,3.0],[0.0,0.5,0.5,0.0],0.25)
        @test plateau.threshold == 0.5
        @test plateau.achieved ≈ 0.5 atol=2e-13
        @test sum(curve_area(c.x,c.y) for c in plateau.components) ≈ 0.5 atol=2e-13
        uniform = EXT._kde_curve_region([0.0,1e-100],[1e100,1e100],0.1)
        @test uniform.achieved ≈ 1 atol=2e-13
        @test only(uniform.components).x == [0.0,1e-100]

        x, density = [0.0,1.0,2.0,3.0,4.0,5.0], [0.0,0.5,0.0,0.0,0.5,0.0]
        result = EXT._kde_curve_region(x,density,0.68)
        @test length(result.components) == 2
        @test last(first(result.components).x) < 2
        @test first(last(result.components).x) > 3
        @test sum(curve_area(c.x,c.y) for c in result.components) ≈ 0.68 atol=2e-13
        # Content is relative to the actual curve area, including preserved
        # unbounded results whose rectangular and trapezoid integrals differ.
        scaled = EXT._kde_curve_region(x,density .* 17,0.68)
        @test scaled.threshold ≈ 17result.threshold rtol=2e-13
        @test scaled.achieved ≈ result.achieved atol=2e-13
        @test all(a.x ≈ b.x for (a,b) in zip(scaled.components,result.components))
        # Relative-height arithmetic prevents a finite physical curve from
        # overflowing merely because its coordinate units are microscopic.
        extreme = EXT._kde_curve_region([0.0,1e-308],[0.0,1.7e308],0.5)
        @test extreme.achieved ≈ 0.5 atol=2e-13
        @test all(c -> all(isfinite,c.y),extreme.components)
    end

    @testset "Credible-region primitive polygons carry their declared mass" begin
        data = reshape([0.05,0.1,0.15,0.8,0.85,0.9],1,:)
        weights = [1.0,2.0,1.0,1.0,2.0,1.0]
        for p in (0.5,0.9)
            config = merge(BAT._default_makie_diagonal_config(),(support=(0.0,1.0),levels=[p]))
            primitive = EXT.compute_plotting_primitives(view(data,:,:),view(weights,:),
                QuantileKDE1D(),EXT.LiveRecipe(),EXT.LiveCell(),config)
            @test !isempty(primitive.polys)
            @test sum(polygon_area,primitive.polys) ≈ p atol=2e-12
        end
        lo, hi = 0.5, 0.5+1e-10
        narrow = reshape([lo],1,:)
        config = merge(BAT._default_makie_diagonal_config(),(support=(lo,hi),levels=[0.68]))
        primitive = EXT.compute_plotting_primitives(view(narrow,:,:),view([1.0],:),
            QuantileKDE1D(),EXT.LiveRecipe(),EXT.LiveCell(),config)
        area = sum(polygon_area,primitive.polys)
        # A flat segment may include its whole tied plateau; a vertex mask must
        # never turn this supported narrow interval into a zero-area polygon.
        @test 0.68-2e-12 <= area <= 1+2e-12
    end

    @testset "Ordinary positive unbounded density compatibility" begin
        x, y, w = [-1.0,-0.3,0.2,1.3], [-0.2,0.7,-0.8,0.9], ones(4)
        result = EXT._weighted_kde1d(x,w)
        original = EXT.kde(x;weights=w)
        @test result.x == original.x && result.density == original.density
        pair = EXT._weighted_kde2d(x,y,w)
        original2 = EXT.kde((x,y);weights=w)
        @test pair.x == original2.x && pair.y == original2.y
        @test pair.density == original2.density
    end
end

end # module
