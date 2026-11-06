# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieSupportPrecision

using BAT, Test, Makie, CairoMakie, Distributions, ValueShapes

const EXT = Base.get_extension(BAT, :BATMakieExt)
const VALUES = parse.(BigFloat, ["0.1", "0.15", "0.2", "0.25", "0.3"])
const SUPPORT = (first(VALUES), last(VALUES))
const RECIPES = (upper=KDE2D, diagonal=KDE1D, lower=KDE2D)
const QUANTILES = (upper=QuantileKDE2D, diagonal=QuantileKDE1D, lower=QuantileKDE2D)

samples(x=VALUES) = DensitySampleVector([[a, 2b] for (a, b) in zip(x, reverse(x))], zeros(length(x)))
curve_mass(k) = sum(diff(k.x) .* (k.density[1:end-1] .+ k.density[2:end]) ./ 2)
cell_mass(k) = sum(k.density .* (diff(k.xedges) * diff(k.yedges)'))

@testset "KDE support preserves original membership" begin
    @testset "Boundary observations and consistent numerical conversion" begin
        @test Float64(SUPPORT[1]) > SUPPORT[1]
        @test Float64(SUPPORT[2]) < SUPPORT[2]
        weights = ones(length(VALUES))
        for bounds in (SUPPORT, (big"0.3", big"0.4"))
            x = collect(range(bounds...; length=5))
            @test all(v -> bounds[1] <= v <= bounds[2], x)
            result = EXT._weighted_kde1d(x, weights, bounds)
            control = EXT._weighted_kde1d(Float64.(x), weights, Float64.(bounds))
            @test (first(result.x), last(result.x)) == Float64.(bounds)
            @test result.x ≈ control.x rtol=2e-12
            @test result.density ≈ control.density rtol=2e-12
            @test curve_mass(result) ≈ 1 atol=2e-12

            for ybounds in (bounds, (-Inf, Inf))
                joint = EXT._weighted_kde2d(x, reverse(x), weights, (bounds, ybounds))
                reference = EXT._weighted_kde2d(Float64.(x), reverse(Float64.(x)), weights,
                    (Float64.(bounds), Float64.(ybounds)))
                @test (first(joint.xedges), last(joint.xedges)) == Float64.(bounds)
                @test joint.xedges ≈ reference.xedges rtol=2e-12
                if isfinite(first(ybounds))
                    @test joint.yedges ≈ reference.yedges rtol=2e-12
                    @test joint.density ≈ reference.density rtol=2e-12
                else
                    # Original-coordinate bandwidths can differ by roundoff,
                    # changing ceil-based cell counts on the unbounded axis.
                    @test collect(extrema(joint.yedges)) ≈ collect(extrema(reference.yedges)) rtol=2e-12
                    @test all(v -> isfinite(v) && v >= 0, joint.density)
                end
                @test cell_mass(joint) ≈ 1 atol=2e-12
            end

            # These outside observations round to the same Float64 as a valid
            # endpoint. Neither conversion nor outward rounding may admit them.
            for (index, outside) in ((1, prevfloat(bounds[1])), (5, nextfloat(bounds[2])))
                bad = copy(x)
                bad[index] = outside
                @test Float64(bad[index]) == Float64(x[index])
                @test_throws "inside hard support" EXT._weighted_kde1d(bad, weights, bounds)
                @test_throws "inside hard support" EXT._weighted_kde2d(x, bad, weights, (bounds, bounds))
                @test_throws "inside hard support" EXT._weighted_kde2d(bad, x, weights, (bounds, (-Inf, Inf)))
            end
            ignored = EXT._weighted_kde1d(vcat(x, prevfloat(bounds[1])), vcat(weights, 0.0), bounds)
            @test ignored == result
            tiny = vcat(BigFloat.(weights), big"1e-1000")
            @test_throws "inside hard support" EXT._weighted_kde1d(vcat(x, prevfloat(bounds[1])), tiny, bounds)
        end
        for bounds in ((SUPPORT[1], nextfloat(SUPPORT[1])), (-big"1e400", SUPPORT[2]),
                (SUPPORT[1], big"1e400"))
            @test_throws ArgumentError EXT._weighted_kde1d(VALUES, weights, bounds)
        end
    end

    @testset "Explicit and inferred bounds retain their values" begin
        support = [SUPPORT, 2 .* SUPPORT]
        prior = product_distribution([Uniform(s...) for s in support])
        for target in (support, prior, EvaluatedMeasure(prior; empirical=samples()))
            lo, hi = EXT._support_vectors(target, 2)
            @test lo == first.(support)
            @test hi == last.(support)
        end
        @test EXT._support_vectors(Uniform(SUPPORT...), 1) == ([SUPPORT[1]], [SUPPORT[2]])
        nested = NamedTupleDist(a=Uniform(SUPPORT...), sub=NamedTupleDist(b=Uniform(0.0, 1.0), c=4.2))
        for target in (nested, ValueShapes.unshaped(nested))
            lo, hi = EXT._support_vectors(target, 2)
            @test (lo[1], hi[1]) == SUPPORT
            @test (lo[2], hi[2]) === (0.0, 1.0)
        end
        # Mixed coordinate bounds must not promote exact endpoints to Float64.
        lo, hi = EXT._support_vectors([(1//10, 3//10), (0.0, 1.0)], 2)
        @test (lo[1], hi[1]) === (1//10, 3//10)
        @test (lo[2], hi[2]) === (0.0, 1.0)
    end

    @testset "Public figures and graph updates" begin
        source = samples()
        before = deepcopy(source)
        support = [SUPPORT, 2 .* SUPPORT]
        prior = product_distribution([Uniform(s...) for s in support])
        for recipes in (RECIPES, QUANTILES), ordinary in (false, true), target in (support, prior)
            fig = ordinary ? Figure() : bat_makie_plot(source, recipes, [1, 2], 2;
                support=target, display=false)
            try
                ordinary && Makie.plot(fig[1, 1], source; recipes, vsel=[1, 2], N_max=2, support=target)
                @test !isempty(Makie.colorbuffer(fig))
            finally
                empty!(fig)
            end
        end
        for recipes in (RECIPES, QUANTILES)
            graph, _ = EXT._setup_static_graph(source, recipes, [1, 2], 2,
                BAT._default_makie_triagonal_config(), BAT._default_makie_diagonal_config(); support)
            node = EXT.primitive_symbol(recipes.upper(), (2, 1))
            original = deepcopy(graph[node][])
            @test graph[:support_lo][] == first.(support)
            @test graph[:support_hi][] == last.(support)
            EXT._apply_vsel_to_graph!(graph, 2, 2, [2, 1])
            @test graph[node][].xedges == original.yedges
            @test graph[node][].yedges == original.xedges
            Makie.update!(graph; window_steps=(2, 4))
            @test graph[:flat_samples][] == source[2:4]
            Makie.update!(graph; window_steps=(100, 101))
            @test isempty(graph[node][].xedges)
            Makie.update!(graph; window_steps=nothing)
            EXT._apply_vsel_to_graph!(graph, 2, 2, [1, 2])
            @test isequal(graph[node][], original)
        end
        @test source == before

        for endpoint in (1, length(VALUES)), recipes in (RECIPES, QUANTILES)
            bad = copy(VALUES)
            bad[endpoint] = endpoint == 1 ? prevfloat(first(bad)) : nextfloat(last(bad))
            for target in (support, prior)
                @test_throws "inside hard support" bat_makie_plot(samples(bad), recipes, [1, 2], 2;
                    support=target, display=false)
                @test_throws "inside hard support" Makie.convert_arguments(Makie.Plot{Makie.plot}, samples(bad);
                    recipes, vsel=[1, 2], N_max=2, support=target)
            end
        end
    end
end

end
