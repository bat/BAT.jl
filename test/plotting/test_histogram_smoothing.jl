# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestHistogramSmoothing
using BAT, Test, StatsBase, Random

# Mathematical resource oracle: unbounded integer arithmetic, with no use of
# the production planner's division guards or its order of admission checks.
function resource_oracle(dims, sigmas)
    all(>(0), dims) && all(s -> isfinite(s) && s >= 0, sigmas) || return nothing
    sides = map(s -> 4ceil(BigInt, BigFloat(s)) + 1, sigmas)
    n, k = prod(big.(dims)), prod(sides)
    padded = prod(big.(dims) .+ sides .- 1)
    buffer, work = 2k + 3n + padded + 2sum(sides), n*k
    return (admit=k <= 2^20 && buffer <= 2^23 && work <= 2^26 &&
        all(l -> ((l-1)÷2)^2 <= typemax(Int), sides),
        sides=sides, kernel=k, buffer=buffer, work=work)
end

function check_plan(dims, sigmas)
    expected = resource_oracle(dims, sigmas)
    actual = try
        BAT._histogram_smoothing_plan(dims, sigmas)
    catch err
        err
    end
    if expected === nothing || !expected.admit
        @test actual isa ArgumentError
    else
        @test !(actual isa Exception)
        if !(actual isa Exception)
            @test actual.sides == expected.sides
            @test actual.kernel_cells == expected.kernel
            @test actual.buffer_cells == expected.buffer
            @test actual.work == expected.work
        end
    end
end

# Direct finite-support Gaussian sums in high precision. Neither BAT's
# Gaussian/convolution nor its histogram region helpers construct this field.
function gaussian_oracle(hist, sigmas)
    setprecision(BigFloat, 256) do
        dims = size(hist.weights)
        widths = map(e -> diff(BigFloat.(e)), hist.edges)
        field = [BigFloat(hist.weights[i,j]) /
            (hist.isdensity ? 1 : widths[1][i]*widths[2][j])
            for i in 1:dims[1], j in 1:dims[2]]
        radius = map(s -> 2ceil(Int, s), sigmas)
        factor(d, s) = iszero(s) ? BigFloat(d == 0) : exp(-BigFloat(d)^2/(2BigFloat(s)^2))
        out = [sum(field[a,b]*factor(i-a, sigmas[1])*factor(j-b, sigmas[2])
            for a in 1:dims[1], b in 1:dims[2]
            if abs(i-a) <= radius[1] && abs(j-b) <= radius[2])
            for i in 1:dims[1], j in 1:dims[2]]
        Float64.(out ./ maximum(out))
    end
end

@testset "Bounded histogram contour smoothing" begin
    @testset "Independent resource admission and overflow" begin
        for dims in ((2,2), (200,200), (1000,1000), (2,1000000),
                     (2,typemax(Int)), (typemax(Int),typemax(Int)), (1,1), (0,2), (-1,2)),
            s in (0., nextfloat(0.), 0.1, 1., 9., nextfloat(9.), 255., 256.,
                  262143., nextfloat(262143.), 1e6, ldexp(1.,62), floatmax(Float64), NaN, Inf, -1.)
            for sigmas in ((s,s), (s,0.), (0.,s))
                check_plan(dims, sigmas)
            end
        end
        rng = Xoshiro(975)
        for _ in 1:1000
            dims = (rand(rng,1:10000), rand(rng,1:10000))
            sigmas = (rand(rng,Bool) ? 0. : 2.0^rand(rng,-10:22), 2.0^rand(rng,-10:22))
            check_plan(dims, sigmas)
            check_plan(reverse(dims), reverse(sigmas))
        end
        # Cross each independent budget with its other two inequalities met.
        for (dims, accepted, rejected, budget) in (
            ((2,2), (255.,255.), (nextfloat(255.),nextfloat(255.)), "kernel"),
            ((200,200), (9.,9.), (10.,10.), "work"),
            ((2,4), (262142.,0.), (262143.,0.), "buffer"))
            for (d, a, r) in ((dims, accepted, rejected),
                             (reverse(dims), reverse(accepted), reverse(rejected)))
                check_plan(d, a)
                check_plan(d, r)
                err = try BAT._histogram_smoothing_plan(d,r) catch e; e end
                @test err isa ArgumentError
                reason = Sys.WORD_SIZE == 32 && budget == "buffer" ? "working integer range" : "$budget budget"
                @test err isa ArgumentError && occursin(reason, sprint(showerror,err))
            end
        end
        c = Int(big(2)^26 ÷ 41^2 ÷ 2)
        check_plan((2,c), (10.,10.))
        check_plan((2,c+1), (10.,10.))
        # Neighbours of the native-offset square boundary, tested symbolically.
        for s in (23170.,23171.)
            check_plan((2,2), (s,0.))
            check_plan((2,2), (0.,s))
        end
    end

    @testset "Gaussian field, displayed mass and scalar compatibility" begin
        for closed in (:left,:right), density in (false,true),
            sigmas in ((0.,0.), (0.,0.8), (0.8,0.), (0.25,1.3), (2.,0.4), (1.,1.))
            edges = ([0.,1.,3.,4.], [0.,2.,3.,6.,7.])
            hist = Histogram(edges, [0. 2. 1. 4.; 3. 0. 6. 1.; 2. 5. 0. 7.], closed, density)
            before = deepcopy(hist)
            result = BAT._histogram_contour_data(hist, [0.27,0.65,0.93]; smoothing=sigmas)
            @test result.density ≈ gaussian_oracle(hist,sigmas) rtol=5e-14 atol=1e-15
            widths = map(e -> diff(BigFloat.(e)), hist.edges)
            mass = BigFloat.(result.density) .* (widths[1]*widths[2]')
            for (p,c,content) in zip(result.requests, result.cutoffs, result.contents)
                expected = first(t for t in sort(unique(vec(result.density)); rev=true)
                    if sum(mass[result.density .>= t]) >= BigFloat(p)*sum(mass))
                @test c == expected
                @test content ≈ Float64(sum(mass[result.density .>= c])/sum(mass)) rtol=1e-14
            end
            @test hist == before
            if sigmas[1] == sigmas[2]
                @test isequal(result, BAT._histogram_contour_data(hist, [0.27,0.65,0.93]; smoothing=sigmas[1]))
            end
        end
        # Preserve the inherited scalar accumulation exactly for admitted input.
        hist = Histogram(([0.,1.,2.,3.], [0.,1.,2.,3.,4.]),
            [1. 2. 3. 4.; 5. 6. 7. 8.; 9. 10. 11. 12.], :left, true)
        field = hist.weights ./ maximum(hist.weights)
        for sigma in (0.3,0.6,1.,2.,10.)
            legacy = BAT.convolution(field, BAT.gaussian_kernel(sigma); padding=:same)
            legacy ./= maximum(legacy)
            result = BAT._histogram_contour_data(hist, [0.3,0.9]; smoothing=sigma)
            @test isequal(result.density, legacy)
        end
    end

    @testset "Input contracts, demand and ownership" begin
        hist = Histogram(([0.,1.,2.], [0.,1.,2.]), [1. 2.; 3. 4.], :left, true)
        before = deepcopy(hist)
        zero = Histogram(hist.edges, zeros(2,2), :left, true)
        default_rng = copy(Random.default_rng())
        for sigma in (1e6, ldexp(1.,62), floatmax(Float64), (1e6,0.), (0.,1e6))
            @test_throws ArgumentError BAT._histogram_contour_data(hist, [0.6]; smoothing=sigma)
            @test isempty(BAT._histogram_contour_data(hist, Float64[]; smoothing=sigma).cutoffs)
            @test all(isnothing, BAT._histogram_contour_data(hist, [0.]; smoothing=sigma).cutoffs)
            @test all(isnothing, BAT._histogram_contour_data(zero, [0.6]; smoothing=sigma).cutoffs)
        end
        for sigma in (-1., NaN, Inf, (-1.,0.), (0.,NaN), (Inf,0.))
            for intervals in (Float64[], [0.6])
                @test_throws ArgumentError BAT._histogram_contour_data(hist, intervals; smoothing=sigma)
            end
        end
        for sigma in ((1.,), (1.,2.,3.), ("bad",1.), [1.,2.], 1+im)
            @test_throws TypeError BAT._histogram_contour_data(hist, [0.6]; smoothing=sigma)
        end
        for sigma in (0, 1, Float32(0.5), 0.5, big"0.5", 1//2,
                      (Float32(0.5),1), (big"0.5",1//2), (0,0))
            canonical = sigma isa Real ? Float64(sigma) : map(Float64,sigma)
            @test isequal(BAT._histogram_contour_data(hist, [0.6]; smoothing=sigma),
                BAT._histogram_contour_data(hist, [0.6]; smoothing=canonical))
        end
        @test_throws ArgumentError BAT._histogram_contour_data(hist, [0.6]; smoothing=1e-200)
        @test_throws ArgumentError BAT._histogram_contour_data(hist, [0.6]; smoothing=(0.,1e-200))
        @test hist == before
        @test rand(copy(Random.default_rng()), UInt64, 8) == rand(default_rng, UInt64, 8)
    end
end
end
