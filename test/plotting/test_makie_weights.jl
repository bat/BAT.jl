# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieWeights

using BAT, Test, Makie, CairoMakie, LinearAlgebra

const EXT = Base.get_extension(BAT, :BATMakieExt)
const DIAG = merge(BAT._default_makie_diagonal_config(), (nbins=4,))
const PAIR = merge(BAT._default_makie_triagonal_config(), (nbins=(4, 3),))
const RECIPES = (upper=Hist2D, diagonal=Hist1D, lower=Hist2D)

primitives(x, w, recipe, config=DIAG) = EXT.compute_plotting_primitives(
    view(x, :, :), view(w, :), recipe, EXT.LiveRecipe(), EXT.LiveCell(), config)
samples(x, w) = DensitySampleVector([collect(v) for v in eachcol(x)],
    zeros(length(w)); weight=copy(w))
oracle_weight(w) = BigFloat(w)
oracle_weight(w::BAT.ULogarithmic) = exp(BigFloat(log(w)))

# Independent population oracle, including represented subnormal inputs.
function moments(x, w)
    raw = oracle_weight.(w)
    p = raw ./ sum(raw)
    means = [sum(BigFloat(x[d, i]) * p[i] for i in eachindex(w)) for d in axes(x, 1)]
    covariance = [sum((BigFloat(x[d, i]) - means[d]) *
        (BigFloat(x[e, i]) - means[e]) * p[i] for i in eachindex(w))
        for d in axes(x, 1), e in axes(x, 1)]
    return Float64.(means), Float64.(covariance)
end

function bin_mass(x, w, edges; closed=:left)
    mass = zeros(BigFloat, map(e -> length(e) - 1, edges))
    for index in CartesianIndices(mass), row in eachindex(w)
        inside = all(eachindex(edges)) do d
            lo, hi, value = edges[d][index[d]], edges[d][index[d] + 1], x[d, row]
            closed === :left ? lo <= value < hi : lo < value <= hi
        end
        inside && (mass[index] += BigFloat(w[row]))
    end
    return mass
end

function expected_bins(x, w, edges, mode; closed=:left)
    result = bin_mass(x, w, edges; closed)
    mode in (:probability, :pdf) && (result ./= sum(result))
    if mode in (:density, :pdf)
        for index in CartesianIndices(result)
            result[index] /= prod(BigFloat(edges[d][index[d] + 1]) - BigFloat(edges[d][index[d]])
                for d in eachindex(edges))
        end
    end
    return Float64.(result)
end

function check_histogram(x, w, mode; closed=:left)
    before = (copy(x), copy(w))
    config = merge(DIAG, (normalization=mode, closed))
    p = primitives(x[1:1, :], w, Hist1D(), config)
    expected = expected_bins(x[1:1, :], w, (p.edges,), mode; closed)
    @test p.weights ≈ expected rtol=5e-14 atol=0
    @test all(v -> isfinite(v) && v >= 0, p.weights)
    @test p.centers ≈ p.edges[1:end-1] .+ diff(p.edges) ./ 2
    @test x == before[1] && w == before[2]

    pairconfig = merge(PAIR, (normalization=mode, closed))
    edges = EXT._hist_edges(nothing, Tuple(eachrow(x)), PAIR.nbins, closed)
    p2 = primitives(x, w, Hist2D(), pairconfig)
    expected2 = expected_bins(x, w, edges, mode; closed)
    @test isnan.(p2.weights) == iszero.(expected2)
    occupied = expected2 .> 0
    @test p2.weights[occupied] ≈ expected2[occupied] rtol=5e-14 atol=0
    @test x == before[1] && w == before[2]
    return p, p2
end

function descendants(axis, type)
    result = Any[]
    function visit(plot)
        plot isa type && push!(result, plot)
        foreach(visit, plot.plots)
    end
    foreach(visit, axis.scene.plots)
    return result
end

function check_public_figure(fig, x, w)
    axis = only(filter(item -> item isa Axis, fig.content))
    points = only(descendants(axis, Makie.BarPlot))[1][]
    edges = Float64[p[1] for p in only(descendants(axis, Makie.Stairs))[1][]]
    expected = expected_bins(reshape(x, 1, :), w, (edges,), :pdf)
    @test Float64[p[2] for p in points] ≈ expected rtol=3e-5 atol=1e-6
    @test all(p -> all(isfinite, p), points)
    return nothing
end

@testset "Makie operation-specific weight arithmetic" begin
    x = [-2.0 0.0 3.0; 1.0 4.0 -1.0]

    @testset "Histograms distinguish probability, PDF, count and count density" begin
        for scale in (1.0, 5e307, 1e-308, nextfloat(0.0)),
                mode in (:probability, :pdf), closed in (:left, :right)
            check_histogram(x, scale .* [1.0, 2.0, 3.0], mode; closed)
        end
        for w in ([1, 2, 3], UInt8[200, 200, 200], 1e-300 .* [1.0, 2.0, 3.0]),
                mode in (:none, :density), closed in (:left, :right)
            check_histogram(x, w, mode; closed)
        end
        for mode in (:none, :density),
                w in (5e307 .* [1.0, 2.0, 3.0], [typemax(Int), typemax(Int), 1])
            @test_throws OverflowError primitives(x[1:1, :], w, Hist1D(),
                merge(DIAG, (normalization=mode,)))
            @test_throws OverflowError primitives(x, w, Hist2D(),
                merge(PAIR, (normalization=mode,)))
        end
        check_histogram(x, [typemax(Int), typemax(Int), 1], :pdf)
    end

    @testset "Population moments and covariance geometry use relative mass" begin
        for w in ([1.0, 2.0, 3.0], 5e307 .* [1.0, 2.0, 3.0],
                1e-308 .* [1.0, 2.0, 3.0], nextfloat(0.0) .* [1.0, 2.0, 3.0],
                Float16[10_000, 20_000, 30_000], Float32[1e38, 2e38, 3e38],
                big"1e400" .* BigFloat[1, 2, 3], big"1e-400" .* BigFloat[1, 2, 3],
                exp.(BAT.ULogarithmic, 1000.0 .+ log.([1.0, 2.0, 3.0])))
            before = copy(w)
            mu, covariance = moments(x, w)
            sigma = sqrt.(diag(covariance))
            @test only(primitives(x[1:1, :], w, Mean1D()).x) ≈ mu[1] rtol=4e-14
            mean2 = primitives(x, w, Mean2D(), PAIR)
            @test [only(mean2.μ_x), only(mean2.μ_y)] ≈ mu rtol=4e-14
            @test primitives(x[1:1, :], w, Std1D()).positions ≈
                [mu[1] - sigma[1], mu[1] + sigma[1]] rtol=4e-14
            err = primitives(x, w, Errorbars2D(), PAIR)
            @test [only(err.μ_x), only(err.μ_y)] ≈ mu rtol=4e-14
            @test [only(err.err_x), only(err.err_y)] ≈ sigma rtol=4e-14

            # Reconstruct covariance from the two visible principal-axis
            # vectors, rather than comparing against BAT/StatsBase covariance.
            ellipse = primitives(x, w, Cov2D(), PAIR)
            directions = [(Vector(ellipse.axes_segments[2i]) -
                Vector(ellipse.axes_segments[2i - 1])) / 2 for i in 1:2]
            observed_covariance = sum(v * v' for v in directions)
            @test observed_covariance ≈ covariance rtol=2e-13 atol=1e-14
            @test all(i -> (Vector(ellipse.axes_segments[2i]) +
                Vector(ellipse.axes_segments[2i - 1])) / 2 ≈ mu, 1:2)

            pdf = primitives(x[1:1, :], w, PDF1D(), merge(DIAG, (npoints_pdf=31,)))
            expected_pdf = exp.(-((pdf.x .- mu[1]) ./ sigma[1]).^2 ./ 2) ./ (sqrt(2pi) * sigma[1])
            @test pdf.y ≈ expected_pdf rtol=4e-14 atol=0
            @test w == before
        end

        # Float32 division would erase the tiny mass. Its large coordinate
        # makes the contribution visible in the population variance.
        wide = reshape([1e40, 0.0, 0.0], 1, :)
        weights = Float32[floatmin(Float32), floatmax(Float32), floatmax(Float32)]
        mu, covariance = moments(wide, weights)
        @test only(primitives(wide, weights, Mean1D()).x) ≈ only(mu) rtol=4e-14
        @test only(primitives(wide, weights, Errorbars1D()).err) ≈ sqrt(only(covariance)) rtol=4e-14
    end

    @testset "Zero mass, invalid input and nonconstant numerical collapse" begin
        recipes = ((Mean1D(), 1, DIAG), (Std1D(), 1, DIAG),
            (Errorbars1D(), 1, DIAG), (PDF1D(), 1, DIAG),
            (Mean2D(), 2, PAIR), (Std2D(), 2, PAIR),
            (Errorbars2D(), 2, PAIR), (Cov2D(), 2, PAIR))
        for (recipe, ndims, config) in recipes
            empty = primitives(zeros(ndims, 0), Float64[], recipe, config)
            @test isequal(primitives(fill(NaN, ndims, 3), zeros(3), recipe, config), empty)
            for w in ([-1.0, 2.0, 3.0], [NaN, 1.0, 1.0], [Inf, 1.0, 1.0], [1e308, 1.0, 1e-308]),
                    filter in (false, true)
                @test_throws ArgumentError primitives(x[1:ndims, :], w,
                    recipe, merge(config, (filter=filter,)))
            end
        end
        zero_rows = reshape([NaN, Inf, 3.0, 5.0], 1, :)
        @test primitives(zero_rows, [0.0, 0.0, 1.0, 1.0], Mean1D()).x == [4.0]
        @test primitives(zero_rows, [0.0, 0.0, 1.0, 1.0], Std1D()).positions == [3.0, 5.0]
        @test_throws ArgumentError primitives(reshape([NaN, 0.0, 1.0], 1, :),
            [1e-6, 1.0, 1.0], Mean1D(), merge(DIAG, (filter=true,)))
        @test isempty(primitives(fill(3.0, 1, 3), ones(3), PDF1D()).x)
        singular = primitives([0.0 1.0 2.0; 0.0 1.0 2.0], ones(3), Cov2D(), PAIR)
        @test all(point -> all(isfinite, point), singular.ellipse_points)
        for recipe in (Std1D(), Errorbars1D(), PDF1D())
            @test_throws ArgumentError primitives(reshape([0.0, 1e-200], 1, :), ones(2), recipe)
        end
        for recipe in (Std1D(), Errorbars1D())
            @test_throws ArgumentError primitives(reshape([-2.0, 2.0], 1, :), ones(2),
                recipe, merge(DIAG, (nsigma=1e308,)))
        end
        @test_throws ArgumentError primitives([-2.0 2.0; -2.0 2.0], ones(2),
            Cov2D(), merge(PAIR, (nsigma=1e308,)))
        for filter in (false, true), w in ([-1.0, 2.0, 3.0], [Inf, 1.0, 1.0])
            @test_throws ArgumentError primitives(x[1:1, :], w, Hist1D(), merge(DIAG, (filter=filter,)))
        end
        @test_throws ArgumentError primitives(zeros(1, 0), Float64[], Hist1D(),
            merge(DIAG, (normalization=:unsupported,)))
        for (recipe, ndims, config) in recipes
            recipe isa Union{Std1D,Std2D,Errorbars1D,Errorbars2D,Cov2D} || continue
            for count in (0, 2), bad in (-1.0, NaN, Inf)
                @test_throws ArgumentError primitives(zeros(ndims, count), zeros(count),
                    recipe, merge(config, (nsigma=bad,)))
            end
        end
        @test_throws ArgumentError primitives(zeros(1, 0), Float64[], PDF1D(),
            merge(DIAG, (npoints_pdf=1,)))
    end

    @testset "Hexbin absolute admission and heatmap color conversion" begin
        for scale in (1.0, 1e-300)
            w = scale .* [1.0, 2.0, 3.0]
            p = primitives(x, w, Hexbin2D(), PAIR)
            @test p.weights == w
            @test p.thresh == minimum(w)
            explicit = primitives(x, w, Hexbin2D(), merge(PAIR, (threshold=0.25,)))
            @test explicit.thresh == 0.25
            spec = only(EXT.compose_plotspecs(p, Hexbin2D(), PAIR))
            @test spec.kwargs[:weights] == w
            @test spec.kwargs[:threshold] == minimum(w)
        end
        @test_throws OverflowError primitives(x, 5e307 .* [1.0, 2.0, 3.0], Hexbin2D(), PAIR)
        @test_throws ArgumentError primitives(x, BigFloat[1, 1, big"1e-400"], Hexbin2D(), PAIR)
        @test_throws ArgumentError primitives(x, [-1.0, 2.0, 3.0], Hexbin2D(), PAIR)
        @test_throws ArgumentError primitives([NaN 1.0; 0.0 1.0], [0.0, 1.0], Hexbin2D(), PAIR)
        for threshold in (-1.0, NaN, Inf)
            @test_throws ArgumentError primitives(zeros(2, 0), Float64[], Hexbin2D(),
                merge(PAIR, (threshold=threshold,)))
        end
        for height in (1e50, 1e-50), transposed in (false, true)
            p = (xedges=[-0.5, 0.5, 1.5], yedges=[-0.5, 0.5, 1.5], weights=[height NaN; NaN height])
            @test_throws ArgumentError EXT.compose_plotspecs(p, Hist2D(), PAIR; transposed)
        end
        valid = (xedges=[-0.5, 0.5, 1.5], yedges=[-0.5, 0.5, 1.5], weights=[1.0 NaN; NaN 2.0])
        @test length(EXT.compose_plotspecs(valid, Hist2D(), PAIR)) == 1
    end

    @testset "Actual graphs and public figures retain the relative population" begin
        w = 5e307 .* [1.0, 2.0, 3.0]
        source = samples(x, w)
        before = deepcopy(source)
        graph = first(EXT._setup_static_graph(source, RECIPES, [1, 2], 2, PAIR, DIAG))
        @test graph[:flat_weights][] == w
        for window in (nothing, (2, 3), (3, 3), nothing)
            Makie.update!(graph; window_steps=window)
            rows = isnothing(window) ? (1:3) : (window[1]:window[2])
            mu, _ = moments(x[:, rows], w[rows])
            actual = graph[EXT.primitive_symbol(Mean1D(), (1, 1))][]
            @test only(actual.x) ≈ mu[1] rtol=4e-14
            @test graph[:flat_weights][] == w[rows]
        end
        @test source == before

        values = [0.137, 0.473, 0.819]
        one_dimensional = samples(reshape(values, 1, :), w)
        for ordinary in (false, true)
            fig = ordinary ? Figure() : bat_makie_plot(one_dimensional,
                RECIPES, [1], 1; display=false)
            try
                ordinary && Makie.plot(fig[1, 1], one_dimensional;
                    recipes=RECIPES, vsel=[1], N_max=1)
                check_public_figure(fig, values, w)
                @test !isempty(Makie.colorbuffer(fig))
                if !ordinary
                    slider = only(filter(item -> item isa IntervalSlider, fig.content))
                    apply = only(filter(item -> item isa Button && item.label[] == "Apply", fig.content))
                    slider.interval[] = (2, 3)
                    Makie.notify(apply.clicks)
                    check_public_figure(fig, values[2:3], w[2:3])
                end
            finally
                empty!(fig)
            end
        end
        @test one_dimensional.weight == w
    end
end

end # module
