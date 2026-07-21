# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function determine_recipe_status(subject::Mean1D, live_recipe::R1) where {R1<:BATMakieRecipe}
        return LiveRecipe()
end

function determine_recipe_status(subject::Std1D, live_recipe::R1) where {R1<:BATMakieRecipe}
        return LiveRecipe()
end

function determine_recipe_status(subject::Mean2D, live_recipe_1::R1, live_recipe_2::R2) where {R1<:BATMakieRecipe,R2<:BATMakieRecipe}
        return LiveRecipe()
end

function determine_recipe_status(subject::Cov2D, live_recipe_1::R1, live_recipe_2::R2) where {R1<:BATMakieRecipe,R2<:BATMakieRecipe}
        return LiveRecipe()
end

function determine_recipe_status(subject::Std2D, live_recipe_1::R1, live_recipe_2::R2) where {R1<:BATMakieRecipe,R2<:BATMakieRecipe}
        return LiveRecipe()
end


struct Makie1DStats <: BATMakieRecipe end

function get_stats_plotspecs(
        graph::ComputeGraph,
        vsel::Tuple{Integer,Integer},
        recipe::Makie1DStats,
        config::NamedTuple
)
        i = vsel[1]
        plotspecs = []

        mean_primitives = graph[primitive_symbol(Mean1D(), (i, i))][]
        append!(plotspecs, compose_plotspecs(mean_primitives, Mean1D(), config))

        std_primitives = graph[primitive_symbol(Std1D(), (i, i))][]
        append!(plotspecs, compose_plotspecs(std_primitives, Std1D(), config))

        return plotspecs
end

struct Makie2DStats <: BATMakieRecipe end

function get_stats_plotspecs(
        graph::ComputeGraph,
        vsel::Tuple{Integer,Integer},
        recipe::Makie2DStats,
        config::NamedTuple
)
        i, j = vsel
        plotspecs = []

        mean_primitives = graph[primitive_symbol(Mean2D(), (i, j))][]
        append!(plotspecs, compose_plotspecs(mean_primitives, Mean2D(), config))

        cov_primitives = graph[primitive_symbol(Cov2D(), (i, j))][]
        append!(plotspecs, compose_plotspecs(cov_primitives, Cov2D(), config))

        std_primitives = graph[primitive_symbol(Std2D(), (i, j))][]
        append!(plotspecs, compose_plotspecs(std_primitives, Std2D(), config))

        return plotspecs
end



function compute_plotting_primitives(
        ::SubArray,
        ::SubArray,
        ::Cov2D,
        ::RS,
        ::CS,
        ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
        return
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Cov2D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        (; nsigma) = config

        w_prob = ProbabilityWeights(weights)
        μ = vec(mean(marg_coords, w_prob, dims=2))
        Σ = cov(Matrix(marg_coords), w_prob, 2; corrected=true) # TODO: Is corrected = true the correct choice? 

        vals, vecs = eigen(Σ)
        stds = sqrt.(clamp.(vals, 0, Inf))

        θ = range(0, 2π, length=200)
        circle = [cos.(θ)'; sin.(θ)']

        ellipse_mat = vecs * nsigma * Diagonal(stds) * circle .+ μ
        ellipse_points = Point2f.(ellipse_mat[1, :], ellipse_mat[2, :])

        axes_segments = Point2f[]
        for i in 1:2
                direction = vecs[:, i] * stds[i] * 2 * nsigma
                p1 = Point2f(μ .- direction)
                p2 = Point2f(μ .+ direction)
                push!(axes_segments, p1, p2)
        end
        return (ellipse_points=ellipse_points, axes_segments=axes_segments)
end


function compose_plotspecs(
        primitives::NamedTuple,
        recipe::Cov2D,
        config::NamedTuple
)
        (; ellipse_points, axes_segments) = primitives

        lines = S.Lines(ellipse_points)
        line_segments = S.LineSegments(axes_segments)

        return [lines, line_segments]
end


function compute_plotting_primitives(
        ::SubArray,
        ::SubArray,
        ::Std1D,
        ::RS,
        ::CS,
        ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
        return
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Std1D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        (; nsigma) = config
        w_prob = ProbabilityWeights(weights)
        μ = mean(vec(marg_coords), w_prob)
        σ = std(vec(marg_coords), w_prob)
        return (positions=[μ - nsigma * σ, μ + nsigma * σ],)
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::Std1D,
        config::NamedTuple
)
        (; positions) = primitives
        lines = S.VLines(positions)
        return [lines]
end


function compute_plotting_primitives(
        ::SubArray,
        ::SubArray,
        ::Std2D,
        ::RS,
        ::CS,
        ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
        return
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Std2D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        (; nsigma) = config

        w_prob = ProbabilityWeights(weights)
        x = marg_coords[1, :]
        y = marg_coords[2, :]
        μ_x = mean(x, w_prob)
        μ_y = mean(y, w_prob)
        σ_x = std(x, w_prob)
        σ_y = std(y, w_prob)

        x_lines = [μ_x - nsigma * σ_x, μ_x + nsigma * σ_x]
        y_lines = [μ_y - nsigma * σ_y, μ_y + nsigma * σ_y]

        return (x_lines=x_lines, y_lines=y_lines)
end


function compose_plotspecs(
        primitives::NamedTuple,
        recipe::Std2D,
        config::NamedTuple
)
        (; x_lines, y_lines) = primitives

        vlines = S.VLines(x_lines)
        hlines = S.HLines(y_lines)
        return [vlines, hlines]
end



function compute_plotting_primitives(
        ::SubArray,
        ::SubArray,
        ::Mean1D,
        ::RS,
        ::CS,
        ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
        return
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Mean1D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        return (x=[mean(vec(marg_coords), ProbabilityWeights(weights))],)
end


function compose_plotspecs(
        primitives::NamedTuple,
        recipe::Mean1D,
        config::NamedTuple
)
        (; x) = primitives
        lines = S.VLines(x)
        return [lines]
end


function compute_plotting_primitives(
        ::SubArray,
        ::SubArray,
        ::Mean2D,
        ::RS,
        ::CS,
        ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
        return
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Mean2D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        w_prob = ProbabilityWeights(weights)
        μ_x = mean(marg_coords[1, :], w_prob)
        μ_y = mean(marg_coords[2, :], w_prob)
        return (μ_x=[μ_x], μ_y=[μ_y])
end


function compose_plotspecs(
        primitives::NamedTuple,
        recipe::Mean2D,
        config::NamedTuple
)
        (; μ_x, μ_y) = primitives
        vlines = S.VLines(μ_x)
        hlines = S.HLines(μ_y)
        return [vlines, hlines]
end


function compute_plotting_primitives(
        ::SubArray,
        ::SubArray,
        ::Errorbars1D,
        ::RS,
        ::CS,
        ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
        return
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Errorbars1D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        (; nsigma) = config
        w_prob = ProbabilityWeights(weights)
        μ = mean(marg_coords, w_prob)
        σ = std(marg_coords, w_prob)
        return (μ=[μ], err=[σ * nsigma])
end


function compose_plotspecs(
        primitives::NamedTuple,
        recipe::Errorbars1D,
        config::NamedTuple
)
        (; μ, err) = primitives
        (; y_ebars) = config

        ebars = S.Errorbars(μ, [y_ebars], err;
                direction=:x
        )
        points = S.Scatter(μ, [y_ebars])
        return [ebars, points]
end



function compute_plotting_primitives(
        ::SubArray,
        ::SubArray,
        ::RS,
        ::CS,
        ::Errorbars2D,
        ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
        return
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Errorbars2D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        (; nsigma) = config
        w_prob = ProbabilityWeights(weights)
        x = marg_coords[1, :]
        y = marg_coords[2, :]
        μ_x = mean(x, w_prob)
        μ_y = mean(y, w_prob)
        σ_x = std(x, w_prob)
        σ_y = std(y, w_prob)

        return (μ_x=[μ_x], μ_y=[μ_y], err_x=[σ_x * nsigma], err_y=[σ_y * nsigma])
end


function compose_plotspecs(
        primitives::NamedTuple,
        recipe::Errorbars2D,
        config::NamedTuple
)
        (; μ_x, μ_y, err_x, err_y) = primitives

        ebars_x = S.Errorbars(μ_x, μ_y, err_x,
                direction=:x
        )
        ebars_y = S.Errorbars(μ_x, μ_y, err_y,
                direction=:y
        )

        point = S.Scatter(μ_x, μ_y)

        return [ebars_x, ebars_y, point]
end


function compute_plotting_primitives(
        ::SubArray,
        ::SubArray,
        ::PDF1D,
        ::RS,
        ::CS,
        ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
        return
end

# TODO: Refactor
function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::PDF1D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        curve_data = lift(p[1], p.npoints) do dist, n
                μ = mean(dist)
                σ = std(dist)

                if σ == 0
                        x_min = μ - 1
                        x_max = μ + 1
                else
                        x_min = μ - 4σ
                        x_max = μ + 4σ
                end

                x_grid = LinRange(x_min, x_max, n)
                y_grid = pdf.(dist, x_grid)

                return (x_grid, y_grid)
        end

        x = lift(d -> d[1], curve_data)
        y = lift(d -> d[2], curve_data)

        poly_points = lift(x, y) do x_val, y_val
                vcat(
                        Point2f.(x_val, y_val),
                        Point2f.(reverse(x_val), 0.0)
                )
        end

        return (poly_points=poly_points, x=x, y=y)
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::PDF1D,
        config::NamedTuple
)
        (; poly_points, x, y) = primitives
        (; color, alpha, filled, strokecolor, strokewidht, edge) = config
        poly = S.Poly(poly_points;
                color=color,
                alpha=alpha,
                visible=filled
        )

        lines = S.Lines(x, y;
                color=strokecolor,
                linewidth=strokewidth,
                visible=edge
        )

        return [poly, lines]
end

