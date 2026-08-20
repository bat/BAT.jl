# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Fresh values per call, deliberately not shared consts -- see the
# change-tracking rationale above _empty_scatter2d_primitives
# (makie_scatter.jl).
_empty_cov2d_primitives() = (ellipse_points=Vector{Point2f}(), axes_segments=Vector{Point2f}())
_empty_std1d_primitives() = (positions=Vector{Float64}(),)
_empty_std2d_primitives() = (x_lines=Vector{Float64}(), y_lines=Vector{Float64}())
_empty_mean1d_primitives() = (x=Vector{Float64}(),)
_empty_mean2d_primitives() = (μ_x=Vector{Float64}(), μ_y=Vector{Float64}())
_empty_errorbars1d_primitives() = (μ=Vector{Float64}(), err=Vector{Float64}())
_empty_errorbars2d_primitives() = (μ_x=Vector{Float64}(), μ_y=Vector{Float64}(), err_x=Vector{Float64}(), err_y=Vector{Float64}())
_empty_pdf1d_primitives() = (poly_points=Vector{Point2f}(), x=Float64[], y=Float64[])

# Precomputed once (not inline `linestyle=:dash` on every recompute) and
# wrapped in Makie.Linestyle -- see Mean1D's own compose_plotspecs comment
# for why the wrapper matters (avoids a deprecation warning that fires when
# a bare `:dash`-converted Vector gets re-resolved, as happens continuously
# under GLMakie's per-frame render loop for a live-updating plot).
const _MEAN1D_DASH = Makie.Linestyle(Makie.line_pattern(:dash, :normal))

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
# represents, how many samples (`n`) have been folded in so far, and the
# index-range slider's low end (`wstart`) -- needed to detect "the picker
# changed which variable this cell shows", "the buffered data shrank", or
# "the window panned to a different (same-width) range" and trigger a reset
# (via `empty!`) rather than an incorrect incremental merge. See
# _update_stats!'s comment below for why `wstart` specifically is needed
# (same underlying issue as _IncrementalHist1DState/2DState in makie_hist.jl).
mutable struct _IncrementalUvState
        stats::BasicUvStatistics{Float64,Weights}
        n::Int
        vsel::Int       # real variable index this state currently tracks (0 = none yet)
        wstart::Int     # window_start this state was last updated against
end
_IncrementalUvState() = _IncrementalUvState(BasicUvStatistics{Float64,Weights}(), 0, 0, 1)

mutable struct _IncrementalMvState
        stats::BasicMvStatistics{Float64,Weights}
        n::Int
        vsel::Tuple{Int,Int}
        wstart::Int
end
_IncrementalMvState() = _IncrementalMvState(BasicMvStatistics{Float64,Weights}(2), 0, (0, 0), 1)

# Folds only the newly-arrived samples (coords[state.n+1:end]) into `state`,
# resetting and reprocessing everything from scratch instead if `vsel` (the
# real variable this cell tracks) changed, the data shrank relative to what
# this state was last updated against, or `wstart` (the index-range slider's
# low handle) moved: this state's only signal for "how much of `coords` is
# genuinely new" is length, which silently breaks once `coords` can be a
# *panned* window instead of always growing from a fixed start -- shifting
# the whole interval (both slider handles at once) keeps the window's width,
# and therefore `n`, exactly constant while every sample in it changes.
# Without this check that shift is invisible to the `n_now > state.n` guard
# below (it's neither a genuine shrink nor a genuine growth in length), so
# the stats overlay would silently keep showing pre-shift values forever --
# the same bug confirmed via direct repro for the histogram recipes' own
# incremental state.
function _update_stats!(state::_IncrementalUvState, coords::AbstractVector, weights::AbstractVector, vsel::Int, wstart::Integer)
        n_now = length(coords)
        if state.vsel != vsel || state.wstart != wstart || state.n > n_now
                empty!(state.stats)
                state.n = 0
                state.vsel = vsel
                state.wstart = wstart
        end
        for k in (state.n+1):n_now
                push!(state.stats, coords[k], weights[k])
        end
        state.n = n_now
        return state
end

# `coords` is the 2xN marg_coords view (not pre-split into x/y): each column
# is one sample, matching BasicMvStatistics.push!'s expected per-sample vector.
# See the Uv method's comment above for why `wstart` must also force a reset.
function _update_stats!(state::_IncrementalMvState, coords::AbstractMatrix, weights::AbstractVector, vsel::Tuple{Int,Int}, wstart::Integer)
        n_now = length(weights)
        if state.vsel != vsel || state.wstart != wstart || state.n > n_now
                empty!(state.stats)
                state.n = 0
                state.vsel = vsel
                state.wstart = wstart
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
        state.n == 0 && return _empty_mean1d_primitives()
        μ = state.stats.mean[]
        # All-zero-weight samples can still be folded in (state.n > 0) while
        # contributing nothing to the weight sum -- BAT's OnlineUvMean/
        # OnlineUvVar/OnlineMvMean/OnlineMvCov all divide by that sum and
        # return NaN rather than erroring when it's zero. Degrading to the
        # empty sentinel (rather than rendering a NaN line, confirmed to
        # silently draw nothing) keeps this consistent with Cov2D below,
        # where the same NaN state is fatal (eigen() on a non-finite matrix
        # throws) rather than silently swallowed.
        isfinite(μ) || return _empty_mean1d_primitives()
        return (x=[μ],)
end

function compute_stats_primitives(::Std1D, state::_IncrementalUvState, config::NamedTuple)
        state.n == 0 && return _empty_std1d_primitives()
        (; nsigma) = config
        μ = state.stats.mean[]
        σ = sqrt(state.stats.var[])
        # See Mean1D's matching comment above -- sqrt(NaN) is NaN, not an
        # error, so this check safely runs after computing σ.
        (isfinite(μ) && isfinite(σ)) || return _empty_std1d_primitives()
        return (positions=[μ - nsigma * σ, μ + nsigma * σ],)
end

function compute_stats_primitives(::Mean2D, state::_IncrementalMvState, config::NamedTuple)
        state.n == 0 && return _empty_mean2d_primitives()
        μ_x, μ_y = state.stats.mean[1], state.stats.mean[2]
        (isfinite(μ_x) && isfinite(μ_y)) || return _empty_mean2d_primitives()
        return (μ_x=[μ_x], μ_y=[μ_y])
end

function compute_stats_primitives(::Std2D, state::_IncrementalMvState, config::NamedTuple)
        state.n == 0 && return _empty_std2d_primitives()
        (; nsigma) = config
        μ_x, μ_y = state.stats.mean[1], state.stats.mean[2]
        σ_x = sqrt(state.stats.cov[1, 1])
        σ_y = sqrt(state.stats.cov[2, 2])
        all(isfinite, (μ_x, μ_y, σ_x, σ_y)) || return _empty_std2d_primitives()
        return (
                x_lines=[μ_x - nsigma * σ_x, μ_x + nsigma * σ_x],
                y_lines=[μ_y - nsigma * σ_y, μ_y + nsigma * σ_y]
        )
end

function compute_stats_primitives(::Cov2D, state::_IncrementalMvState, config::NamedTuple)
        state.n == 0 && return _empty_cov2d_primitives()
        (; nsigma) = config
        μ = [state.stats.mean[1], state.stats.mean[2]]
        Σ = [state.stats.cov[1, 1] state.stats.cov[1, 2]; state.stats.cov[2, 1] state.stats.cov[2, 2]]
        # All-zero-weight samples make Σ all-NaN (see Mean1D's comment
        # above) -- eigen() on a non-finite matrix throws
        # ArgumentError("matrix contains Infs or NaNs"), confirmed directly.
        # Without this guard, a zero-weight dataset crashes the whole
        # visualizer regardless of which recipe is actually selected, since
        # Cov2D is always computed in the background.
        (all(isfinite, μ) && all(isfinite, Σ)) || return _empty_cov2d_primitives()
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
        inputs::NamedTuple,
        vsel::Tuple{Integer,Integer},
        recipe::Makie1DStats,
        config::NamedTuple
)
        i = vsel[1]
        plotspecs = []

        mean_primitives = getproperty(inputs, primitive_symbol(Mean1D(), (i, i)))
        append!(plotspecs, compose_plotspecs(mean_primitives, Mean1D(), config))

        std_primitives = getproperty(inputs, primitive_symbol(Std1D(), (i, i)))
        append!(plotspecs, compose_plotspecs(std_primitives, Std1D(), config))

        return plotspecs
end

struct Makie2DStats <: BATMakieRecipe end

function get_stats_plotspecs(
        inputs::NamedTuple,
        vsel::Tuple{Integer,Integer},
        recipe::Makie2DStats,
        config::NamedTuple;
        # Forwarded to the per-recipe compose_plotspecs calls below -- the 2D
        # stats overlay must follow the same orientation as the cell's main
        # recipe (see _init_gridlayout's invariant comment).
        transposed::Bool=false
)
        i, j = vsel
        plotspecs = []

        # Mean2D (a crosshair -- vlines/hlines exactly at the mean) is
        # deliberately not drawn here, per explicit request -- unlike
        # Makie1DStats above, which keeps its own Mean1D line. Mean2D's own
        # recipe machinery (compute_plotting_primitives/compute_stats_primitives)
        # is untouched and still runs in the background (it's is_incremental,
        # always computed regardless of what's actually drawn, and still
        # registered via BAT_MAKIE_RECIPES_2D in _init_compute_graph) -- this
        # only stops it from being rendered as part of the 2D stats overlay.

        cov_primitives = getproperty(inputs, primitive_symbol(Cov2D(), (i, j)))
        append!(plotspecs, compose_plotspecs(cov_primitives, Cov2D(), config; transposed=transposed))

        std_primitives = getproperty(inputs, primitive_symbol(Std2D(), (i, j)))
        append!(plotspecs, compose_plotspecs(std_primitives, Std2D(), config; transposed=transposed))

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
        return _empty_cov2d_primitives()
end

# Cov2D is_incremental (see is_incremental's definition above), so
# _init_compute_graph only reaches this method -- rather than the
# incremental running-state accumulator (compute_stats_primitives above) --
# when config.filter==true: the incremental accumulator can't apply the
# low-weight cutoff after the fact (it depends on the *global*,
# retroactively-recomputed weight distribution, not something folded in
# incrementally), so this is a full recompute over the filtered sample set
# every time, mirroring how Hist1D/Hist2D/QuantileHist1D/QuantileHist2D
# already handle their own filter=true case via _marginal_view_dist/
# _low_weight_mask. Before this was added, Cov2D (and Mean1D/Std1D/Mean2D/
# Std2D below) had no live+filtered method at all, so Julia's dispatch fell
# through silently to the empty dead-cell sentinel -- filter=true was a
# silent no-op for every stats overlay. Confirmed via direct dispatch test.
function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Cov2D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        isempty(weights) && return _empty_cov2d_primitives()
        (; nsigma) = config
        mask = _low_weight_mask(weights)
        w = view(weights, mask)
        isempty(w) && return _empty_cov2d_primitives()
        w_prob = ProbabilityWeights(w)
        # Plain indexing (not view): StatsBase.cov(::DenseMatrix, ...) doesn't
        # accept a SubArray -- confirmed directly (MethodError). The masked
        # subset is already small, so materializing it here is cheap.
        coords_m = marg_coords[:, mask]
        μ = [mean(view(coords_m, 1, :), w_prob), mean(view(coords_m, 2, :), w_prob)]
        Σ = StatsBase.cov(coords_m, w_prob, 2)
        # See Cov2D's compute_stats_primitives comment for why this guard is
        # needed even after masking -- a low-weight-filtered set can still be
        # entirely zero-weight if every remaining sample happens to carry
        # zero weight.
        (all(isfinite, μ) && all(isfinite, Σ)) || return _empty_cov2d_primitives()
        return _cov_ellipse_primitives(μ, Σ, nsigma)
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::Cov2D,
        config::NamedTuple;
        transposed::Bool=false
)
        (; ellipse_points, axes_segments) = primitives

        if isempty(ellipse_points)
                return PlotSpec[]
        end

        # Lower-triangle cells swap x/y at compose time (see _init_gridlayout's
        # invariant comment) -- for point-based primitives that means swapping
        # each point's own coordinates.
        if transposed
                ellipse_points = Point2f[Point2f(p[2], p[1]) for p in ellipse_points]
                axes_segments = Point2f[Point2f(p[2], p[1]) for p in axes_segments]
        end

        # color=:black set explicitly here (not via the shared Lines/
        # LineSegments theme) per explicit request -- Makie's automatic
        # color-*cycling* (enabled by default for both plot types) silently
        # overrides a theme-level color default, confirmed directly; an
        # explicit per-call color is the only way that reliably takes effect.
        lines = S.Lines(ellipse_points; color=:black)
        line_segments = S.LineSegments(axes_segments; color=:black)

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
        return _empty_std1d_primitives()
end

# See Cov2D's matching comment above -- reached only when config.filter==true.
function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Std1D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        isempty(weights) && return _empty_std1d_primitives()
        (; nsigma) = config
        mask = _low_weight_mask(weights)
        w = view(weights, mask)
        isempty(w) && return _empty_std1d_primitives()
        coords_m = view(vec(marg_coords), mask)
        w_prob = ProbabilityWeights(w)
        μ = mean(coords_m, w_prob)
        σ = std(coords_m, w_prob)
        (isfinite(μ) && isfinite(σ)) || return _empty_std1d_primitives()
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

        # color=:black set explicitly -- see Cov2D's matching comment above
        # (VLines' automatic color-cycling overrides a theme-level default).
        lines = S.VLines(positions; color=:black)
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
        return _empty_std2d_primitives()
end

# See Cov2D's matching comment above -- reached only when config.filter==true.
function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Std2D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        isempty(weights) && return _empty_std2d_primitives()
        (; nsigma) = config
        mask = _low_weight_mask(weights)
        w = view(weights, mask)
        isempty(w) && return _empty_std2d_primitives()
        w_prob = ProbabilityWeights(w)
        coords_m = view(marg_coords, :, mask)
        x, y = view(coords_m, 1, :), view(coords_m, 2, :)
        μ_x, μ_y = mean(x, w_prob), mean(y, w_prob)
        σ_x, σ_y = std(x, w_prob), std(y, w_prob)
        all(isfinite, (μ_x, μ_y, σ_x, σ_y)) || return _empty_std2d_primitives()
        return (
                x_lines=[μ_x - nsigma * σ_x, μ_x + nsigma * σ_x],
                y_lines=[μ_y - nsigma * σ_y, μ_y + nsigma * σ_y]
        )
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::Std2D,
        config::NamedTuple;
        transposed::Bool=false
)
        (; x_lines, y_lines) = primitives

        if isempty(x_lines)
                return PlotSpec[]
        end

        # Transposed cells swap which positions are vertical vs horizontal
        # lines -- see Cov2D's matching comment above.
        transposed && ((x_lines, y_lines) = (y_lines, x_lines))

        # color=:black set explicitly -- see Cov2D's matching comment above.
        vlines = S.VLines(x_lines; color=:black)
        hlines = S.HLines(y_lines; color=:black)
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
        return _empty_mean1d_primitives()
end

# See Cov2D's matching comment above -- reached only when config.filter==true.
function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Mean1D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        isempty(weights) && return _empty_mean1d_primitives()
        mask = _low_weight_mask(weights)
        w = view(weights, mask)
        isempty(w) && return _empty_mean1d_primitives()
        μ = mean(view(vec(marg_coords), mask), ProbabilityWeights(w))
        isfinite(μ) || return _empty_mean1d_primitives()
        return (x=[μ],)
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

        # linestyle=_MEAN1D_DASH (not the bare Symbol :dash) by explicit
        # request, distinguishing the 1D stats overlay's mean line from
        # Std1D's own (solid) VLines -- set directly on this PlotSpec, not
        # via the shared VLines theme, since Std1D/Std2D/Mean2D all use the
        # same VLines type and should stay solid. Wrapped in Makie.Linestyle
        # specifically: a bare `:dash` gets converted to a plain
        # Vector{Float32} pattern internally, and re-resolving that already-
        # converted vector a second time -- confirmed this happens on GLMakie's
        # per-frame render loop for a live-updating plot, though not under
        # CairoMakie's one-shot static render -- hits Makie's own deprecated
        # bare-Vector linestyle path and warns on every frame. A `Linestyle`
        # wrapper is genuinely idempotent under repeated resolution (Makie's
        # own recommended fix for this warning).
        # color=:black set explicitly too -- see Cov2D's matching comment
        # above (VLines' automatic color-cycling overrides a theme default).
        lines = S.VLines(x; linestyle=_MEAN1D_DASH, color=:black)
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
        return _empty_mean2d_primitives()
end

# See Cov2D's matching comment above -- reached only when config.filter==true.
function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Mean2D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        isempty(weights) && return _empty_mean2d_primitives()
        mask = _low_weight_mask(weights)
        w = view(weights, mask)
        isempty(w) && return _empty_mean2d_primitives()
        w_prob = ProbabilityWeights(w)
        coords_m = view(marg_coords, :, mask)
        μ_x = mean(view(coords_m, 1, :), w_prob)
        μ_y = mean(view(coords_m, 2, :), w_prob)
        (isfinite(μ_x) && isfinite(μ_y)) || return _empty_mean2d_primitives()
        return (μ_x=[μ_x], μ_y=[μ_y])
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::Mean2D,
        config::NamedTuple;
        transposed::Bool=false
)
        (; μ_x, μ_y) = primitives

        if isempty(μ_x)
                return PlotSpec[]
        end

        # See Std2D's matching comment above.
        transposed && ((μ_x, μ_y) = (μ_y, μ_x))

        # color=:black set explicitly -- see Cov2D's matching comment above.
        # (Mean2D itself is no longer invoked by the 2D stats overlay --
        # see get_stats_plotspecs's own comment -- kept consistent regardless.)
        vlines = S.VLines(μ_x; color=:black)
        hlines = S.HLines(μ_y; color=:black)
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
        return _empty_errorbars1d_primitives()
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Errorbars1D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        isempty(weights) && return _empty_errorbars1d_primitives()
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
        return _empty_errorbars2d_primitives()
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Errorbars2D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        isempty(weights) && return _empty_errorbars2d_primitives()
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
        config::NamedTuple;
        transposed::Bool=false
)
        (; μ_x, μ_y, err_x, err_y) = primitives

        if isempty(μ_x)
                return PlotSpec[]
        end

        # See Std2D's matching comment above. The error extents must swap
        # together with their means: the direction=:x bars below must carry
        # the transposed x-axis' error, not the original one.
        if transposed
                μ_x, μ_y = μ_y, μ_x
                err_x, err_y = err_y, err_x
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
        return _empty_pdf1d_primitives()
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
        isempty(weights) && return _empty_pdf1d_primitives()
        (; npoints_pdf) = config

        w_prob = ProbabilityWeights(weights)
        μ = mean(vec(marg_coords), w_prob)
        σ = std(vec(marg_coords), w_prob)
        # isempty(weights) above only guards zero *count* -- a nonempty but
        # all-zero-weight vector passes it, and std() with an all-zero
        # ProbabilityWeights returns NaN (0/0), which Normal(μ, NaN) rejects
        # with a DomainError. Unlike Errorbars1D/2D (confirmed dormant, no
        # picker entry), PDF1D is live in the recipe picker, so this is
        # reachable in normal use, not just latent.
        (isfinite(μ) && isfinite(σ)) || return _empty_pdf1d_primitives()
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

# See makie_hist.jl's _diag_y_extent for Hist1D for what this is for.
_diag_y_extent(primitives::NamedTuple, ::PDF1D) = isempty(primitives.y) ? 0.0 : maximum(primitives.y)

