# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieCovarianceAdmission

using BAT, Test, Makie, CairoMakie, LinearAlgebra

const EXT = Base.get_extension(BAT, :BATMakieExt)
const PAIR = BAT._default_makie_triagonal_config()
const RECIPES = (upper=QuantileHist2D, diagonal=QuantileHist1D, lower=QuantileHist2D)
const DECOMPOSITION_ERROR = "covariance eigendecomposition is inaccurate at the coordinate scales"

corners(a, b) = [a -a a -a; b b -b -b]
samples(x, w) = DensitySampleVector([collect(v) for v in eachcol(x)], zeros(length(w)); weight=copy(w))
primitives(x, w; nsigma=1.0) = EXT.compute_plotting_primitives(
    view(x, :, :), view(w, :), Cov2D(), EXT.LiveRecipe(), EXT.LiveCell(), merge(PAIR, (; nsigma)))

function check_covariance(p, scales, correlation; nsigma=1.0)
    directions = [(Vector(p.axes_segments[2i]) - Vector(p.axes_segments[2i - 1])) ./ (2 .* scales)
        for i in 1:2]
    observed = sum(v * v' for v in directions)
    @test observed ≈ nsigma^2 .* correlation rtol=2e-13 atol=2e-14
    for d in 1:2
        @test maximum(abs(point[d] / scales[d]) for point in p.ellipse_points) ≈ nsigma rtol=0.001
    end
    @test all(point -> all(isfinite, point), p.ellipse_points)
end

@testset "Covariance admission uses each coordinate's scale" begin
    a, b = 2.0^500, 2.0^-500
    weights = [3.0, 1.0, 1.0, 3.0]
    @testset "Lost positive variance is rejected" begin
        for x in (corners(a, b), corners(b, a), corners(a, -b))
            before = copy(x)
            @test_throws DECOMPOSITION_ERROR primitives(x, weights)
            @test x == before
        end
    end

    @testset "Admitted nonsingular and singular populations retain their geometry" begin
        for scales in ([1.0, 1.0], [2.0^20, 2.0^-20]), nsigma in (0.0, 1.0, 2.0)
            # Independent population covariance of the weighted corners.
            p = primitives(corners(scales...), weights; nsigma)
            check_covariance(p, scales, [1.0 0.5; 0.5 1.0]; nsigma)
        end
        for scales in ([1.0, 1.0], [a, b], [b, a]), sign in (-1, 1)
            x = corners(scales[1], sign * scales[2])
            p = primitives(x, [1.0, 0.0, 0.0, 1.0])
            check_covariance(p, scales, [1.0 sign; sign 1.0])
            @test count(i -> p.axes_segments[2i] == p.axes_segments[2i - 1], 1:2) == 1
        end
        for x in (zeros(2, 4), [0.0 0.0 0.0 0.0; -1.0 -1.0 1.0 1.0])
            p = primitives(x, ones(4))
            @test all(point -> iszero(point[1]), p.ellipse_points)
            @test all(point -> all(isfinite, point), p.axes_segments)
        end
        # An uncorrelated extreme covariance has representable eigenvectors.
        p = primitives(corners(a, b), ones(4))
        check_covariance(p, [a, b], Matrix{Float64}(I, 2, 2))
    end

    @testset "Public selection rejects loss while hidden overlays remain usable" begin
        source = samples(corners(a, b), weights)
        before = deepcopy(source)
        for ordinary in (false, true)
            fig = ordinary ? Figure() : bat_makie_plot(source, RECIPES, [1, 2], 2; display=false)
            try
                ordinary && Makie.plot(fig[1, 1], source; recipes=RECIPES, vsel=[1, 2], N_max=2)
                @test !isempty(Makie.colorbuffer(fig))
                if !ordinary
                    toggle = first(filter(x -> x isa Toggle, fig.content))
                    @test_throws DECOMPOSITION_ERROR (toggle.active[] = true)
                end
            finally
                empty!(fig)
            end
            recipes = merge(RECIPES, (upper=Cov2D,))
            if ordinary
                fig = Figure()
                try
                    @test_throws DECOMPOSITION_ERROR Makie.plot(fig[1, 1], source;
                        recipes, vsel=[1, 2], N_max=2)
                finally
                    empty!(fig)
                end
            else
                @test_throws DECOMPOSITION_ERROR bat_makie_plot(source, recipes, [1, 2], 2; display=false)
            end
        end
        @test source == before

        unit_source = samples(corners(1.0, 1.0), weights)
        fig = bat_makie_plot(unit_source, RECIPES, [1, 2], 2; display=false)
        try
            first(filter(x -> x isa Toggle, fig.content)).active[] = true
            axis = only(filter(x -> x isa Axis, contents(fig[1, 1][1, 2])))
            ellipse = only(filter(x -> x isa Makie.Lines, axis.scene.plots))[1][]
            @test maximum(abs, first.(ellipse)) ≈ 1.0 rtol=0.001
            @test maximum(abs, last.(ellipse)) ≈ 1.0 rtol=0.001
            @test !isempty(Makie.colorbuffer(fig))
        finally
            empty!(fig)
        end
    end
end

end # module
