# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type CellStatus end

struct LiveCell <: CellStatus end
struct DeadCell <: CellStatus end

abstract type RecipeStatus end

struct LiveRecipe <: RecipeStatus end
struct DeadRecipe <: RecipeStatus end


function determine_recipe_status(subject::R1, live_recipe::R1) where {R1<:BATMakieRecipe}
    return LiveRecipe()
end


function determine_recipe_status(
    subject::R1, live_recipe::R2
) where {R1<:BATMakieRecipe,R2<:BATMakieRecipe}
    return DeadRecipe()
end


function determine_recipe_status(
    subject::R1, live_recipe_1::R1, live_recipe_2::R1
) where {R1<:BATMakieRecipe}
    return LiveRecipe()
end


function determine_recipe_status(
    subject::R1, live_recipe_1::R1, live_recipe_2::R2
) where {R1<:BATMakieRecipe,R2<:BATMakieRecipe}
    return LiveRecipe()
end


function determine_recipe_status(
    subject::R1, live_recipe_1::R2, live_recipe_2::R1
) where {R1<:BATMakieRecipe,R2<:BATMakieRecipe}
    return LiveRecipe()
end


function determine_recipe_status(
    subject::R1, live_recipe_1::R2, live_recipe_2::R3
) where {R1<:BATMakieRecipe,R2<:BATMakieRecipe,R3<:BATMakieRecipe}
    return DeadRecipe()
end


function determine_recipe_status(
    subject::BATMakieRecipe, live_recipe::BATMakieRecipe, show_stats::Bool
)
    return determine_recipe_status(subject, live_recipe)
end


function determine_recipe_status(
    subject::BATMakieRecipe, live_recipe_1::BATMakieRecipe,
    live_recipe_2::BATMakieRecipe, show_stats::Bool
)
    return determine_recipe_status(subject, live_recipe_1, live_recipe_2)
end


"""
    compute_plotting_primitives(marg_coords, weights, recipe, recipe_status, cell_status, config)
    compute_plotting_primitives(marg_coords, weights, chainids, recipe::ChainScatter2D,
        recipe_status, cell_status, config)

*BAT-internal, not part of stable public API.*

Compute a recipe-specific `NamedTuple` for the Makie graph. For a live cell with
samples, `marg_coords` and `weights` are `SubArray` views: coordinate rows are
selected variables (one or two), columns are samples, and weights align with
columns. Empty or inactive coordinate views may be 0 × 0. Weights need not sum
to one. `config` supplies recipe options and optional `domain`/`support` bounds
in coordinate-row order. The `ChainScatter2D` overload also takes sample-aligned
chain IDs; `Trace2D` uses `_timed_trace_primitives` instead.

Only `LiveRecipe`/`LiveCell` methods compute a marginal; statistical overlays can
be live independently of the selected main recipe. Other status combinations
and empty samples yield recipe-shaped empty primitives. Validation may still
throw; zero-mass behavior and validation rules are recipe-specific.

Keep concrete field representations compatible across a graph node's live and
empty states. Recompute without mutating inputs or previously published outputs.
Allocate fresh empty arrays so ComputePipeline can compare successive empty
values. Metadata such as chain IDs may be borrowed and must remain read-only.

This is an internal graph protocol: defining a `BATMakieRecipe` subtype does not
register it with `_init_compute_graph`.
"""
function compute_plotting_primitives end

"""
    compose_plotspecs(primitives, recipe, config)
    compose_plotspecs(primitives, recipe, config; transposed::Bool = false)

*BAT-internal, not part of stable public API.*

Compose a fresh vector of Makie `PlotSpec`s from a recipe's primitive `NamedTuple`
and styling configuration. Return `PlotSpec[]` for its empty sentinel. Do not
mutate the primitives: plot specs may retain their arrays, which must remain
read-only after composition. The returned vector may be extended with overlays.

Every 2D method accepts `transposed::Bool = false`. Shared pair primitives use
coordinate row 1 as x and row 2 as y, oriented for the upper-triangle cell.
With `transposed=true`, swap the complete x/y geometry for the lower cell,
including raster dimensions and directional extents. Both cells then show the
column variable on x and the row variable on y; the shared primitives are unchanged.
"""
function compose_plotspecs end


# Truncates a requested vsel to the free-parameter range and to at most N_max
# entries (the fixed grid size), warning if that changes anything.
function _clamp_vsel(vsel::AbstractVector{<:Integer}, n_dof::Integer, N_max::Integer)
    vsel_clamped = filter(i -> 1 <= i <= n_dof, vsel)
    if length(vsel_clamped) > N_max
        vsel_clamped = vsel_clamped[1:N_max]
    end
    if vsel_clamped != vsel
        @warn "Requested vsel indices $vsel are invalid (each must be in 1:$n_dof, at most N_max=$N_max at a time); using $vsel_clamped instead."
    end
    return vsel_clamped
end


_clamp_vsel(vsel::AbstractVector{<:Integer}, samples::DensitySampleVector, N_max::Integer) =
    _clamp_vsel(vsel, totalndof(varshape(samples)), N_max)
