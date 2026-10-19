# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestHistogramRegions

using BAT, StatsBase, Test

hist1d(weights; edges=collect(0:length(weights)), density=false, closed=:left) =
    Histogram((copy(edges),), copy(weights), closed, density)

exact(x::Real) = Rational{BigInt}(x)

# This oracle integrates the represented source endpoints and heights directly.
# It neither factors products nor normalizes heights before multiplying volumes.
function exact_cells(hist)
    volumes = map(CartesianIndices(hist.weights)) do index
        prod(exact(hist.edges[d][index[d]+1]) - exact(hist.edges[d][index[d]])
            for d in 1:ndims(hist.weights))
    end
    heights = exact.(vec(hist.weights))
    volumes = vec(volumes)
    masses = hist.isdensity ? heights .* volumes : heights
    densities = hist.isdensity ? heights : heights ./ volumes
    return (; masses, densities, total=sum(masses))
end

function exact_hdr(hist, p)
    cells = exact_cells(hist)
    mask = falses(length(cells.masses))
    (iszero(p) || iszero(cells.total)) && return mask, exact(0)
    for cutoff in sort(unique(cells.densities); rev=true)
        cutoff > 0 || continue
        mask .= cells.densities .>= cutoff
        sum(cells.masses[mask]) >= exact(p) * cells.total && break
    end
    return mask, sum(cells.masses[mask]) / cells.total
end

# Independent acceptance bound, evaluated with wider arithmetic only in tests.
# rho covers conversion, widths and products; g covers positive summation.
# The production error also includes outward rounding of bound evaluation.
function reference_error(hist, data)
    isempty(data.masses) && return big(0)
    working_precision = precision(data.total)
    working_eta = nextfloat(zero(eltype(data.masses)))
    setprecision(BigFloat, max(256, working_precision + 64)) do
        n, d = length(data.masses), ndims(hist.weights)
        u = ldexp(BigFloat(1), -working_precision)
        eta = BigFloat(working_eta)
        gamma(k) = k*u / (1-k*u)
        rho, g = gamma(2d+1), gamma(max(n-1, 0))
        delta = (rho/(1-rho) + g)/(1-g) + n*eta/(1-rho)
        (4delta + 4u + 4eta)/(1-delta)
    end
end

function check_hdr(hist, requests; exact_masks=false)
    original = deepcopy(hist)
    data = BAT._histogram_region_data(hist)
    cells = exact_cells(hist)
    regions, contents = BAT.get_smallest_intervals(hist, Float64.(requests))
    @test length(regions) == length(contents) == length(requests)
    @test data.error >= 0
    @test data.error < 1
    if !iszero(cells.total)
        @test 1 <= maximum(data.masses) < 2
        @test data.total >= 1
        @test all(i -> iszero(hist.weights[i]) || data.masses[i] > 0, eachindex(hist.weights))
        @test BigFloat(data.error) >= reference_error(hist, data)
    end
    error = exact(data.error)
    previous = trues(length(cells.masses))
    for (region, content, p) in zip(regions, contents, sort(Float64.(requests); rev=true))
        mask = vec(region.weights .> 0)
        @test all(i -> iszero(region.weights[i]) || region.weights[i] == hist.weights[i], eachindex(hist.weights))
        @test eltype(region.weights) == eltype(hist.weights)
        @test region.edges == hist.edges
        @test region.closed == hist.closed
        @test region.isdensity == hist.isdensity
        @test region.weights !== hist.weights
        @test all(mask .<= previous)
        previous = mask
        actual = iszero(cells.total) ? exact(0) : sum(cells.masses[mask]) / cells.total
        @test abs(exact(content) - actual) <= error
        @test actual >= exact(p) - error || iszero(cells.total)
        if p == 0 || iszero(cells.total)
            @test !any(mask)
            @test content == 0
        elseif p == 1
            @test mask == (vec(hist.weights) .> 0)
            @test content == 1
        else
            @test any(mask)
            cutoff = minimum(data.density_keys[mask])
            @test mask == ((data.density_keys .>= Ref(cutoff)) .& (vec(hist.weights) .> 0))
            above = (data.density_keys .> Ref(cutoff)) .& (vec(hist.weights) .> 0)
            @test sum(cells.masses[above]) / cells.total < exact(p) + error
        end
        if exact_masks
            expected, _ = exact_hdr(hist, p)
            @test mask == expected
        end
    end
    @test hist.weights == original.weights
    @test hist.edges == original.edges
    if length(regions) > 1
        @test regions[1].weights !== regions[2].weights
    end
    return regions, contents
end

@testset "Integrated histogram region probability" begin
    @testset "Unequal cells, density ranking and representations" begin
        motivating = hist1d([0.0, 0.15, 0.2]; edges=[0.0, 2.0, 6.0, 8.0], density=true)
        regions, contents = check_hdr(motivating, [0.5]; exact_masks=true)
        @test only(regions).weights == motivating.weights
        @test only(contents) == 1

        counts = hist1d([4, 6]; edges=[0, 1, 10])
        regions, contents = check_hdr(counts, [0.35, 0.5]; exact_masks=true)
        @test regions[2].weights == [4, 0]
        @test contents ≈ [1, 0.4]

        edges = ([0, 1, 4], [0, 2, 3])
        density = [4.0 1.0; 2.0 3.0]
        volume = [2 1; 6 3]
        hist = Histogram(edges, copy(density), :right, true)
        regions, contents = check_hdr(hist, [0.6]; exact_masks=true)
        @test only(regions).weights == [4 0; 2 3]
        @test only(contents) ≈ 29/30
        masks = [vec(r.weights .> 0) for r in check_hdr(hist, [0.2, 0.55, 0.6, 0.98]; exact_masks=true)[1]]
        for equivalent in (
                Histogram(edges, density .* volume, :right, false),
                Histogram(edges, (density .* volume) ./ 30, :right, false),
                Histogram(edges, density ./ 30, :right, true))
            got, _ = check_hdr(equivalent, [0.2, 0.55, 0.6, 0.98]; exact_masks=true)
            @test [vec(r.weights .> 0) for r in got] == masks
        end
        transposed = Histogram(reverse(edges), permutedims(density), :right, true)
        transposed_regions, transposed_contents = check_hdr(transposed, [0.6]; exact_masks=true)
        @test only(transposed_regions).weights == permutedims(only(regions).weights)
        @test transposed_contents == contents
    end

    @testset "Complete plateaus, endpoints, duplicate requests and ownership" begin
        for density in (false, true)
            hist = hist1d(density ? [2.0, 2.0, 1.0] : [2, 6, 2]; edges=[0, 1, 4, 6], density)
            requests = [0.3, 0.0, 1.0, 0.3, 0.7]
            regions, contents = check_hdr(hist, requests; exact_masks=true)
            @test contents ≈ [1, 0.8, 0.8, 0.8, 0]
            @test regions[2].weights == hist.weights .* [1, 1, 0]
            regions[2].weights[1] = 0
            @test regions[3].weights[1] == hist.weights[1]
            @test hist.weights[1] > 0
        end
        flat = hist1d([2.0, 2.0, 2.0]; edges=[0, 1, 3, 7], density=true)
        regions, contents = check_hdr(flat, [nextfloat(0.0), 0.5, 1.0]; exact_masks=true)
        @test all(==(1), contents)
        @test all(r -> r.weights == flat.weights, regions)
        for closed in (:left, :right)
            check_hdr(hist1d([0, 1, 0, 3, 0]; closed), [0, 0.2, 0.8, 1]; exact_masks=true)
        end
        zero_hist = hist1d(zeros(3); edges=[0, 1, 4, 9], density=true)
        check_hdr(zero_hist, [0, 0.5, 1])
        check_hdr(zero_hist, Float64[])
        check_hdr(hist1d([3, 2, 1]), Float64[])
    end

    @testset "Original heights and scale invariance" begin
        exact_height = Int64(2)^53 + 1
        count_hist = hist1d([exact_height, Int64(1), Int64(0)])
        count_regions, _ = check_hdr(count_hist, [0.2]; exact_masks=true)
        @test only(count_regions).weights == [exact_height, 0, 0]
        # StatsBase's constructor requires floating density heights. Its mutable
        # flag can nevertheless produce integer-density storage, which the
        # internal helper admits; preserve its source ordering and typed copies.
        hist = hist1d([exact_height, exact_height-1, Int64(1)])
        hist.isdensity = true
        regions, _ = check_hdr(hist, [0.2]; exact_masks=true)
        @test only(regions).weights == [exact_height, 0, 0]
        for values in (Float16[45000, 30000, 15000], Float32[3e38, 2e38, 1e38],
                BigInt[big(10)^400*3, big(10)^400*2, big(10)^400],
                BigFloat[big"3e1000", big"2e1000", big"1e1000"])
            hist = hist1d(values)
            regions, _ = check_hdr(hist, [0.6]; exact_masks=true)
            @test only(regions).weights == values .* [1, 1, 0]
        end
        for scale in (1.0, 5e307, 1e-300, nextfloat(0.0))
            check_hdr(hist1d(scale .* [3, 2, 1]), [0.25, 0.6]; exact_masks=true)
        end
        base = hist1d([4.0, 3.0, 1.0]; edges=[0, 1, 3, 7], density=true)
        base_regions, base_contents = check_hdr(base, [0.2, 0.6, 0.9]; exact_masks=true)
        for scale in (2.0^-400, 2.0^400)
            transformed = hist1d(Float64.(base.weights) ./ scale;
                edges=Float64.(only(base.edges)) .* scale, density=true)
            regions, contents = check_hdr(transformed, [0.2, 0.6, 0.9]; exact_masks=true)
            @test [r.weights .> 0 for r in regions] == [r.weights .> 0 for r in base_regions]
            @test contents ≈ base_contents
        end
    end

    @testset "Compensating scales and admitted numerical boundaries" begin
        compensated = hist1d([1e308, 1e-308]; edges=[0.0, 1e-308, 1e308], density=true)
        regions, contents = check_hdr(compensated, [0.4]; exact_masks=true)
        @test only(regions).weights == [1e308, 0.0]
        @test only(contents) ≈ 0.5
        enormous = hist1d([1e308]; edges=[0.0, 1e308], density=true)
        check_hdr(enormous, [0, 0.5, 1]; exact_masks=true)
        hist3d = Histogram(([0.0, 1e200], [0.0, 1e200], [0.0, 1e-200]),
            fill(1.0, 1, 1, 1), :left, false)
        check_hdr(hist3d, [0.5, 1]; exact_masks=true)

        # Float32 subtraction overflows; promoted endpoint subtraction does not.
        check_hdr(hist1d(Float32[1]; edges=Float32[-3e38, 3e38]), [0.5]; exact_masks=true)
        tiny = nextfloat(0.0)
        parent = hist1d([tiny, 1.0])
        subset = hist1d([tiny, 0.0])
        @test only(BAT.get_probability_content(parent, StatsBase.Histogram[subset])) == tiny
        for hist in (hist1d([tiny, 2.0]), hist1d([1e308, 1e-308]),
                hist1d([1.0]; edges=[-1e308, 1e308]),
                hist1d([1, 1]; edges=Int64[2^53+1, 2^53+3, 2^53+5]))
            @test_throws ArgumentError BAT.get_smallest_intervals(hist, [0.5])
        end
        setrounding(BigFloat, RoundUp) do
            @test_throws ArgumentError BAT.get_smallest_intervals(hist1d(BigFloat[3, 2, 1]), [0.5])
        end
        # Test the real machine mode, including BigFloat working arithmetic
        # whose original Float64 heights still undergo hardware comparisons.
        subnormal_parents = (hist1d([tiny]), hist1d([tiny]; edges=BigFloat[0, 1]))
        previous_mode = get_zero_subnormals()
        try
            if set_zero_subnormals(true)
                for hist in subnormal_parents
                    @test_throws ArgumentError BAT.get_smallest_intervals(hist, [0.5])
                    @test_throws ArgumentError BAT.split_central(hist, [0.5])
                    @test_throws ArgumentError BAT.get_probability_content(hist, [deepcopy(hist)])
                end
                @test get_zero_subnormals()
            else
                @test !get_zero_subnormals()
            end
        finally
            set_zero_subnormals(previous_mode)
        end
        @test get_zero_subnormals() == previous_mode
        # MPFR's minimum normal value has no gradual underflow margin.
        floor = floatmin(BigFloat)
        @test_throws ArgumentError BAT.get_smallest_intervals(hist1d([floor]), [0.5])
        @test_throws ArgumentError BAT.get_smallest_intervals(
            hist1d(BigFloat[1]; edges=[zero(floor), floor]), [0.5])
        @test_throws ArgumentError BAT.get_smallest_intervals(hist1d(BigFloat[1]), [floor])
    end

    @testset "Independent exact oracle over small unequal grids" begin
        # Deterministic integer fixtures are identical across Julia versions.
        # Non-power-of-two widths also exercise computed count-density keys.
        for n in 2:6, rep in 1:6, density in (false, true)
            widths = [1 + mod(3i + rep, 5) for i in 1:n]
            edges = [0; cumsum(widths)]
            values = [mod(5i + 3rep, 13) for i in 1:n]
            hist = hist1d(density ? float.(values) : values; edges, density)
            check_hdr(hist, [0, 0.125, 0.3, 0.5, 0.75, 0.9, 1])
        end
        # This exact crossing failed under max-mantissa normalization. Common
        # power-of-two scaling preserves its dyadic count arithmetic.
        boundary = hist1d([1, 3, 6, 4]; edges=[0, 8, 10, 18, 22])
        regions, contents = check_hdr(boundary, [0.5]; exact_masks=true)
        @test only(regions).weights == [0, 3, 0, 4]
        @test only(contents) == 0.5
    end

    @testset "Central intervals integrate mass and retain strict tails" begin
        hist = hist1d([0.2, 0.3, 0.2]; edges=[0, 1, 3, 4], density=true)
        regions, contents = BAT.split_central(hist, [0.5, 0.0, 1.0])
        @test regions[1].weights == hist.weights
        @test regions[2].weights == [0.0, 0.3, 0.0]
        @test all(iszero, regions[3].weights)
        @test contents ≈ [1, 0.6, 0]
        equality = hist1d([1, 2, 1])
        regions, contents = BAT.split_central(equality, [0.5])
        @test only(regions).weights == equality.weights
        @test only(contents) == 1
        for density in (false, true), rep in 1:8
            values = [1+mod(i+2rep, 7) for i in 1:5]
            hist = hist1d(density ? float.(values) : values; edges=[0, 1, 3, 6, 10, 15], density)
            cells = exact_cells(hist)
            data = BAT._histogram_region_data(hist)
            regions, contents = BAT.split_central(hist, [0.2, 0.6, 0.9])
            for (region, content, p) in zip(regions, contents, [0.9, 0.6, 0.2])
                selected = vec(region.weights .> 0)
                actual = sum(cells.masses[selected]) / cells.total
                @test abs(exact(content) - actual) <= exact(data.error)
                @test actual >= exact(p) - exact(data.error)
                @test all(selected[findfirst(selected):findlast(selected)])
                left = sum(cells.masses[1:findfirst(selected)-1]) / cells.total
                right = sum(cells.masses[findlast(selected)+1:end]) / cells.total
                @test left < (1-exact(p))/2 + exact(data.error)
                @test right < (1-exact(p))/2 + exact(data.error)
            end
        end
        @test_throws ArgumentError BAT.split_central(Histogram(([0, 1, 2], [0, 1, 2]), ones(2, 2), :left, false), [0.5])
    end

    @testset "Masked-parent content, complementarity and validation" begin
        parent = hist1d([4.0, 3.0, 1.0]; edges=[0, 1, 3, 7], density=true, closed=:right)
        first = hist1d([4.0, 0.0, 0.0]; edges=[0, 1, 3, 7], density=true, closed=:right)
        rest = hist1d([0.0, 3.0, 1.0]; edges=[0, 1, 3, 7], density=true, closed=:right)
        contents = BAT.get_probability_content(parent, StatsBase.Histogram[first, rest, deepcopy(parent)])
        @test contents ≈ [2/7, 5/7, 1]
        @test sum(contents[1:2]) ≈ 1
        for bad in (
                hist1d([2.0, 0.0, 0.0]; edges=[0, 1, 3, 7], density=true, closed=:right),
                hist1d([4.0, 0.0, 0.0]; edges=[0, 1, 4, 7], density=true, closed=:right),
                hist1d([4, 0, 0]; edges=[0, 1, 3, 7], density=false, closed=:right),
                hist1d([4.0, 0.0, 0.0]; edges=[0, 1, 3, 7], density=true, closed=:left),
                hist1d([4.0, -1.0, 0.0]; edges=[0, 1, 3, 7], density=true, closed=:right),
                hist1d([4, NaN, 0]; edges=[0, 1, 3, 7], density=true, closed=:right))
            @test_throws ArgumentError BAT.get_probability_content(parent, StatsBase.Histogram[bad])
        end
        zero_parent = hist1d(zeros(2))
        @test BAT.get_probability_content(zero_parent, StatsBase.Histogram[deepcopy(zero_parent)]) == [0]
        @test_throws ArgumentError BAT.get_probability_content(zero_parent, StatsBase.Histogram[hist1d([1, 0])])
        for invalid in ([1.0, -1], [1.0, NaN], [1.0, Inf])
            for f in (BAT.get_smallest_intervals, BAT.split_central)
                @test_throws ArgumentError f(hist1d(invalid), Float64[])
            end
            @test_throws ArgumentError BAT.get_probability_content(hist1d(invalid), StatsBase.Histogram[])
        end
        for edges in ([0.0, 0, 1], [0.0, 2, 1], [0.0, NaN, 2], [0.0, Inf, 2])
            for f in (BAT.get_smallest_intervals, BAT.split_central)
                @test_throws ArgumentError f(hist1d(zeros(2); edges), Float64[])
            end
        end
        malformed = hist1d(zeros(2); edges=[0, 1, 2])
        push!(only(malformed.edges), 3) # Corrupt a valid parent after construction.
        @test_throws DimensionMismatch BAT.get_smallest_intervals(malformed, Float64[])
        for probabilities in ([-0.1], [1.1], [NaN], [Inf])
            for f in (BAT.get_smallest_intervals, BAT.split_central)
                @test_throws ArgumentError f(zero_parent, probabilities)
            end
        end
        @test_throws ArgumentError BAT.get_smallest_intervals(hist1d([1//2, 1//3]), [0.5])
    end

    @testset "Contour field and cutoff are the same distribution" begin
        edges = ([0.0, 1.0, 4.0], [0.0, 2.0, 3.0])
        hist = Histogram(edges, [4.0 0.0; 0.0 1.0], :left, false)
        original = deepcopy(hist)
        requests = [0.3, 1.0, 0.0, 0.8, 0.3]
        for smoothing in (0.0, 0.6)
            result = BAT._histogram_contour_data(hist, requests; smoothing)
            @test result.requests == sort(requests; rev=true)
            @test requests[result.order] == result.requests
            @test size(result.density) == size(hist.weights)
            @test eltype(result.density) == Float64
            @test all(x -> isfinite(x) && x >= 0, result.density)
            @test maximum(result.density) == 1
            displayed = Histogram(edges, result.density, :left, true)
            cells = exact_cells(displayed)
            error = exact(BAT._histogram_region_data(displayed).error)
            for (p, cutoff, content) in zip(result.requests, result.cutoffs, result.contents)
                if p == 0
                    @test isnothing(cutoff)
                    @test content == 0
                else
                    @test cutoff isa Float64
                    @test 0 < cutoff <= 1
                    mask = vec(result.density .>= cutoff)
                    above = vec(result.density .> cutoff)
                    actual = sum(cells.masses[mask]) / cells.total
                    @test abs(exact(content) - actual) <= error
                    @test actual >= exact(p) - error
                    @test sum(cells.masses[above]) / cells.total < exact(p) + error
                end
            end
        end
        unsmoothed = BAT._histogram_contour_data(hist, requests)
        smoothed = BAT._histogram_contour_data(hist, requests; smoothing=0.6)
        @test unsmoothed.density != smoothed.density
        @test all(>(0), smoothed.density)
        @test hist.weights == original.weights
        @test hist.edges == original.edges
        compensated = Histogram(([0.0, 1e-308, 1e308], [0.0, 1.0, 2.0]), ones(2, 2), :left, false)
        check_hdr(compensated, [0.3, 0.8])
        @test_throws ArgumentError BAT._histogram_contour_data(compensated, [0.5])
        for smoothing in (-1.0, Inf, NaN)
            @test_throws ArgumentError BAT._histogram_contour_data(hist, Float64[]; smoothing)
        end
        @test_throws ArgumentError BAT._histogram_contour_data(
            Histogram(([0, 1], [0, 1, 2]), ones(1, 2), :left, false), [0.5])
        zero_result = BAT._histogram_contour_data(Histogram(edges, zeros(2, 2), :left, false), [0, 0.5, 1])
        @test all(isnothing, zero_result.cutoffs)
        @test all(iszero, zero_result.contents)
        @test all(iszero, zero_result.density)
    end
end

end # module
