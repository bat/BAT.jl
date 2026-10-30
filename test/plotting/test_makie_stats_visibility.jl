# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieStatsVisibility
using BAT, Test, Makie, CairoMakie, Statistics

const EXT = Base.get_extension(BAT, :BATMakieExt)
const RECIPES = (upper=Scatter2D, diagonal=QuantileHist1D, lower=Scatter2D)
const DIAG = BAT._default_makie_diagonal_config()
const PAIR = BAT._default_makie_triagonal_config()

function graph(samples; recipes=RECIPES, selection=[1, 2])
    first(EXT._setup_static_graph(samples, recipes, selection, 2, PAIR, DIAG))
end
primitive(g, recipe, cell=(1, 1)) = g[EXT.primitive_symbol(recipe, cell)][]
empty_primitive(g, recipe, cell=(1, 1)) = all(isempty, values(primitive(g, recipe, cell)))

@testset "Statistics are computed only when requested" begin
    tiny = DensitySampleVector([[-1e-200], [1e-200]], zeros(2))
    recipes = (upper=Hist2D, diagonal=QuantileHist1D, lower=Hist2D)

    @testset "Representable histogram with unrepresentable variance" begin
        before = deepcopy(tiny)
        g = graph(tiny; recipes, selection=[1])
        hist = primitive(g, QuantileHist1D())
        @test length(hist.polys) == 2
        @test all(p -> all(isfinite, p), Iterators.flatten(hist.polys))
        @test 2.4e201 < maximum(p[2] for p in Iterators.flatten(hist.polys)) < 2.6e201
        @test sum((poly[2][1] - poly[1][1]) * poly[3][2] for poly in hist.polys) ≈ 1.0
        @test empty_primitive(g, Mean1D())
        @test empty_primitive(g, Std1D())

        for ordinary in (false, true)
            fig = ordinary ? Figure() : bat_makie_plot(tiny, recipes, [1], 1; display=false)
            try
                ordinary && Makie.plot(fig[1, 1], tiny; recipes, vsel=[1], N_max=1)
                axis = only(filter(x -> x isa Axis, fig.content))
                @test any(p -> p isa Makie.Poly, axis.scene.plots)
                @test !any(p -> p isa Makie.VLines, axis.scene.plots)
                @test !isempty(Makie.colorbuffer(fig))
            finally
                empty!(fig)
            end
        end
        @test tiny == before
    end

    @testset "Hidden computations retain requested numerical checks" begin
        spread_error = "nonconstant weighted spread is outside the working numerical range"
        @test_throws spread_error bat_makie_plot(tiny,
            merge(recipes, (diagonal=Std1D,)), [1], 1; display=false)
        g = graph(tiny; selection=[1])
        grid = EXT._init_gridlayout!(g, 2)
        @test grid[] isa Makie.SpecApi.GridLayout
        Makie.update!(g; show_stats_diag=true)
        @test_throws spread_error grid[]

        tiny_pair = DensitySampleVector([[-1e-200, -2e-200], [1e-200, 2e-200]], zeros(2))
        for flag in (:show_stats_upper, :show_stats_lower)
            g = graph(tiny_pair)
            grid = EXT._init_gridlayout!(g, 2)
            @test grid[] isa Makie.SpecApi.GridLayout
            for recipe in (Mean2D(), Std2D(), Cov2D())
                @test empty_primitive(g, recipe, (2, 1))
            end
            Makie.update!(g; NamedTuple{(flag,)}((true,))...)
            @test_throws "outside the working numerical range" grid[]
        end
        for recipe in (Std2D, Cov2D), side in (:upper_recipe, :lower_recipe)
            g = graph(tiny_pair)
            Makie.update!(g; NamedTuple{(side,)}((recipe,))...)
            @test_throws "outside the working numerical range" primitive(g, recipe(), (2, 1))
        end
    end

    @testset "Overlay toggles, windows, and variable selection" begin
        source = DensitySampleVector([[1.0, 11.0], [3.0, 17.0], [8.0, 13.0]], zeros(3))
        before = deepcopy(source)
        g = graph(source)
        grid = EXT._init_gridlayout!(g, 2)
        for (diag, upper, lower) in ((false, false, false), (true, true, false),
            (false, false, true), (true, true, true), (false, false, false))
            Makie.update!(g; show_stats_diag=diag, show_stats_upper=upper, show_stats_lower=lower)
            @test grid[] isa Makie.SpecApi.GridLayout
            @test empty_primitive(g, Mean1D()) == !diag
            @test empty_primitive(g, Std1D()) == !diag
            @test empty_primitive(g, Cov2D(), (2, 1)) == !(upper || lower)
            @test empty_primitive(g, Std2D(), (2, 1)) == !(upper || lower)
            # The 2D overlay contains covariance and spread, not a mean marker.
            @test empty_primitive(g, Mean2D(), (2, 1))
        end

        Makie.update!(g; show_stats_diag=true, show_stats_lower=true)
        held = deepcopy(primitive(g, Std2D(), (2, 1)))
        for selection in ([2, 1], [1, 2]), window in ((2, 3), (100, 101), nothing)
            Makie.update!(g; idxs=selection, window_steps=window)
            @test grid[] isa Makie.SpecApi.GridLayout
            rows = window === nothing ? (1:3) : (window[1]:min(window[2], 3))
            @test empty_primitive(g, Mean1D()) == isempty(rows)
            @test empty_primitive(g, Std2D(), (2, 1)) == isempty(rows)
            if !isempty(rows)
                x = [source.v[i][selection[1]] for i in rows]
                y = [source.v[i][selection[2]] for i in rows]
                @test only(primitive(g, Mean1D()).x) ≈ mean(x)
                @test primitive(g, Std1D()).positions ≈ mean(x) .+ [-std(x; corrected=false), std(x; corrected=false)]
                @test primitive(g, Std2D(), (2, 1)).x_lines ≈ mean(y) .+ [-std(y; corrected=false), std(y; corrected=false)]
            end
        end
        @test primitive(g, Std2D(), (2, 1)) == held
        Makie.update!(g; idxs=[1])
        @test grid[] isa Makie.SpecApi.GridLayout
        @test empty_primitive(g, Std1D(), (2, 2))
        @test empty_primitive(g, Cov2D(), (2, 1))
        Makie.update!(g; idxs=[1, 2])
        @test primitive(g, Std2D(), (2, 1)) == held
        @test source == before
    end

    @testset "Direct selections remain active with overlays disabled" begin
        source = DensitySampleVector([[1.0, 11.0], [3.0, 17.0], [8.0, 13.0]], zeros(3))
        g = graph(source)
        for recipe in (Mean1D, Std1D)
            Makie.update!(g; diagonal_recipe=recipe)
            @test !empty_primitive(g, recipe())
            Makie.update!(g; show_stats_diag=true)
            Makie.update!(g; show_stats_diag=false)
            @test !empty_primitive(g, recipe())
            Makie.update!(g; diagonal_recipe=QuantileHist1D)
            @test empty_primitive(g, recipe())
        end
        for recipe in (Mean2D, Std2D, Cov2D), side in (:upper_recipe, :lower_recipe)
            Makie.update!(g; NamedTuple{(side,)}((recipe,))...)
            @test !empty_primitive(g, recipe(), (2, 1))
            Makie.update!(g; show_stats_upper=true, show_stats_lower=true)
            Makie.update!(g; show_stats_upper=false, show_stats_lower=false)
            @test !empty_primitive(g, recipe(), (2, 1))
            Makie.update!(g; upper_recipe=recipe, lower_recipe=recipe)
            @test !empty_primitive(g, recipe(), (2, 1))
            Makie.update!(g; upper_recipe=Scatter2D, lower_recipe=Scatter2D)
            @test empty_primitive(g, recipe(), (2, 1))
        end
    end
end
end
