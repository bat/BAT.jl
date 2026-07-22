# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Dead-cell results are always identical and read-only (isempty-checked, then
# discarded), so these are shared const sentinels instead of fresh allocations
# on every recompute (which happens on every new sample batch for every
# non-selected recipe).
const _EMPTY_COV2D_PRIMITIVES = (ellipse_points=Vector{Point2f}(), axes_segments=Vector{Point2f}())
const _EMPTY_STD1D_PRIMITIVES = (positions=Vector{Float64}(),)
const _EMPTY_STD2D_PRIMITIVES = (x_lines=Vector{Float64}(), y_lines=Vector{Float64}())
const _EMPTY_MEAN1D_PRIMITIVES = (x=Vector{Float64}(),)
const _EMPTY_MEAN2D_PRIMITIVES = (μ_x=Vector{Float64}(), μ_y=Vector{Float64}())
const _EMPTY_ERRORBARS1D_PRIMITIVES = (μ=Vector{Float64}(), err=Vector{Float64}())
const _EMPTY_ERRORBARS2D_PRIMITIVES = (μ_x=Vector{Float64}(), μ_y=Vector{Float64}(), err_x=Vector{Float64}(), err_y=Vector{Float64}())
const _EMPTY_PDF1D_PRIMITIVES = (poly_points=Vector{Point2f}(), x=Float64[], y=Float64[])

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
        return _EMPTY_COV2D_PRIMITIVES
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Cov2D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        # A live cell can still have zero samples (e.g. right after vsel
        # activates, before the first batch flushes, or if buffered samples get
        # cleared later) -- mean/cov/eigen all misbehave or error on empty input,
        # so degrade to the same empty result as a dead cell instead.
        isempty(weights) && return _EMPTY_COV2D_PRIMITIVES
        (; nsigma) = config

        w_prob = ProbabilityWeights(weights)
        μ = vec(mean(marg_coords, w_prob, dims=2))
        # Matches Std1D/Std2D, which also don't pass `corrected` (StatsBase's
        # default for ProbabilityWeights is corrected=false) -- keeps the
        # covariance ellipse's implied scale consistent with the std error bars.
        Σ = cov(Matrix(marg_coords), w_prob, 2)

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

        if isempty(ellipse_points)
                return PlotSpec[]
        end

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
        return _EMPTY_STD1D_PRIMITIVES
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Std1D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        isempty(weights) && return _EMPTY_STD1D_PRIMITIVES
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

        if isempty(positions)
                return PlotSpec[]
        end

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
        return _EMPTY_STD2D_PRIMITIVES
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Std2D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        isempty(weights) && return _EMPTY_STD2D_PRIMITIVES
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

        if isempty(x_lines)
                return PlotSpec[]
        end

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
        return _EMPTY_MEAN1D_PRIMITIVES
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Mean1D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        isempty(weights) && return _EMPTY_MEAN1D_PRIMITIVES
        return (x=[mean(vec(marg_coords), ProbabilityWeights(weights))],)
end


function compose_plotspecs(
        primitives::NamedTuple,
        recipe::Mean1D,
        config::NamedTuple
)
        (; x) = primitives

        if isempty(x)
                return PlotSpec[]
        end

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
        return _EMPTY_MEAN2D_PRIMITIVES
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Mean2D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        isempty(weights) && return _EMPTY_MEAN2D_PRIMITIVES
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

        if isempty(μ_x)
                return PlotSpec[]
        end

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
        return _EMPTY_ERRORBARS1D_PRIMITIVES
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Errorbars1D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        isempty(weights) && return _EMPTY_ERRORBARS1D_PRIMITIVES
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

        if isempty(μ)
                return PlotSpec[]
        end

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
        ::Errorbars2D,
        ::RS,
        ::CS,
        ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
        return _EMPTY_ERRORBARS2D_PRIMITIVES
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Errorbars2D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        isempty(weights) && return _EMPTY_ERRORBARS2D_PRIMITIVES
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

        if isempty(μ_x)
                return PlotSpec[]
        end

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
        return _EMPTY_PDF1D_PRIMITIVES
end

# Overlays the Normal distribution fit (by weighted mean/std) to the 1D marginal.
function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::PDF1D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        isempty(weights) && return _EMPTY_PDF1D_PRIMITIVES
        (; npoints_pdf) = config

        w_prob = ProbabilityWeights(weights)
        μ = mean(vec(marg_coords), w_prob)
        σ = std(vec(marg_coords), w_prob)
        dist = Normal(μ, σ)

        x_min, x_max = σ == 0 ? (μ - 1, μ + 1) : (μ - 4σ, μ + 4σ)
        x = collect(LinRange(x_min, x_max, npoints_pdf))
        y = pdf.(dist, x)

        poly_points = vcat(
                Point2f.(x, y),
                Point2f.(reverse(x), 0.0)
        )

        return (poly_points=poly_points, x=x, y=y)
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::PDF1D,
        config::NamedTuple
)
        (; poly_points, x, y) = primitives

        if isempty(x)
                return PlotSpec[]
        end

        (; filled_pdf) = config

        poly = S.Poly(poly_points; visible=filled_pdf)
        lines = S.Lines(x, y)

        return [poly, lines]
end

