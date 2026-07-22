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

# Mean1D/Std1D/Mean2D/Cov2D/Std2D are updated incrementally from running
# (weighted) sufficient statistics instead of being refit from the full
# accumulated sample set on every batch. Reuses BAT's own online-statistics
# library (BAT.OnlineUvMean/OnlineUvVar/OnlineMvMean/OnlineMvCov, bundled here
# via BasicUvStatistics/BasicMvStatistics -- the same types BAT's own MCMC
# chain-statistics tracking uses in src/samplers/mcmc/mcmc_stats.jl) rather
# than a hand-rolled Welford implementation. Errorbars1D/Errorbars2D are
# excluded: nothing in this extension currently calls compose_plotspecs on
# them (dormant), so making them incremental too would just be unused code.
is_incremental(::BATMakieRecipe) = false
is_incremental(::Union{Mean1D,Std1D,Mean2D,Cov2D,Std2D}) = true

# `Weights` (StatsBase's no-bias-correction tag) makes BasicUvStatistics/
# BasicMvStatistics's variance/covariance divide by the plain weight sum with
# no correction term -- matching the direct computation this replaces, which
# relied on StatsBase's std/cov defaulting to corrected=false for
# ProbabilityWeights.
#
# The wrapper only adds what BAT's online-stats types don't track themselves:
# which real variable(s) (`vsel`) this grid cell's accumulator currently
# represents, and how many samples (`n`) have been folded in so far -- needed
# to detect "the picker changed which variable this cell shows" or "the
# buffered data shrank" and trigger a reset (via `empty!`) rather than an
# incorrect incremental merge.
mutable struct _IncrementalUvState
        stats::BasicUvStatistics{Float64,Weights}
        n::Int
        vsel::Int       # real variable index this state currently tracks (0 = none yet)
end
_IncrementalUvState() = _IncrementalUvState(BasicUvStatistics{Float64,Weights}(), 0, 0)

mutable struct _IncrementalMvState
        stats::BasicMvStatistics{Float64,Weights}
        n::Int
        vsel::Tuple{Int,Int}
end
_IncrementalMvState() = _IncrementalMvState(BasicMvStatistics{Float64,Weights}(2), 0, (0, 0))

# Folds only the newly-arrived samples (coords[state.n+1:end]) into `state`,
# resetting and reprocessing everything from scratch instead if `vsel` (the
# real variable this cell tracks) changed or the data shrank relative to what
# this state was last updated against.
function _update_stats!(state::_IncrementalUvState, coords::AbstractVector, weights::AbstractVector, vsel::Int)
        n_now = length(coords)
        if state.vsel != vsel || state.n > n_now
                empty!(state.stats)
                state.n = 0
                state.vsel = vsel
        end
        for k in (state.n+1):n_now
                push!(state.stats, coords[k], weights[k])
        end
        state.n = n_now
        return state
end

# `coords` is the 2xN marg_coords view (not pre-split into x/y): each column
# is one sample, matching BasicMvStatistics.push!'s expected per-sample vector.
function _update_stats!(state::_IncrementalMvState, coords::AbstractMatrix, weights::AbstractVector, vsel::Tuple{Int,Int})
        n_now = length(weights)
        if state.vsel != vsel || state.n > n_now
                empty!(state.stats)
                state.n = 0
                state.vsel = vsel
        end
        for k in (state.n+1):n_now
                push!(state.stats, view(coords, :, k), weights[k])
        end
        state.n = n_now
        return state
end

# Shared with the (non-incremental) Cov2D dead-cell fallback above so the
# ellipse-construction math isn't duplicated between the two code paths.
function _cov_ellipse_primitives(μ::AbstractVector, Σ::AbstractMatrix, nsigma::Real)
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

function compute_stats_primitives(::Mean1D, state::_IncrementalUvState, config::NamedTuple)
        state.n == 0 && return _EMPTY_MEAN1D_PRIMITIVES
        return (x=[state.stats.mean[]],)
end

function compute_stats_primitives(::Std1D, state::_IncrementalUvState, config::NamedTuple)
        state.n == 0 && return _EMPTY_STD1D_PRIMITIVES
        (; nsigma) = config
        μ = state.stats.mean[]
        σ = sqrt(state.stats.var[])
        return (positions=[μ - nsigma * σ, μ + nsigma * σ],)
end

function compute_stats_primitives(::Mean2D, state::_IncrementalMvState, config::NamedTuple)
        state.n == 0 && return _EMPTY_MEAN2D_PRIMITIVES
        return (μ_x=[state.stats.mean[1]], μ_y=[state.stats.mean[2]])
end

function compute_stats_primitives(::Std2D, state::_IncrementalMvState, config::NamedTuple)
        state.n == 0 && return _EMPTY_STD2D_PRIMITIVES
        (; nsigma) = config
        μ_x, μ_y = state.stats.mean[1], state.stats.mean[2]
        σ_x = sqrt(state.stats.cov[1, 1])
        σ_y = sqrt(state.stats.cov[2, 2])
        return (
                x_lines=[μ_x - nsigma * σ_x, μ_x + nsigma * σ_x],
                y_lines=[μ_y - nsigma * σ_y, μ_y + nsigma * σ_y]
        )
end

function compute_stats_primitives(::Cov2D, state::_IncrementalMvState, config::NamedTuple)
        state.n == 0 && return _EMPTY_COV2D_PRIMITIVES
        (; nsigma) = config
        μ = [state.stats.mean[1], state.stats.mean[2]]
        Σ = [state.stats.cov[1, 1] state.stats.cov[1, 2]; state.stats.cov[2, 1] state.stats.cov[2, 2]]
        return _cov_ellipse_primitives(μ, Σ, nsigma)
end

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

