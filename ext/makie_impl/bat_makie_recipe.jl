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

function determine_recipe_status(subject::R1, live_recipe::R2) where {R1<:BATMakieRecipe,R2<:BATMakieRecipe}
        return DeadRecipe()
end


function determine_recipe_status(subject::R1, live_recipe_1::R1, live_recipe_2::R1) where {R1<:BATMakieRecipe}
        return LiveRecipe()
end

function determine_recipe_status(subject::R1, live_recipe_1::R1, live_recipe_2::R2) where {R1<:BATMakieRecipe,R2<:BATMakieRecipe}
        return LiveRecipe()
end

function determine_recipe_status(subject::R1, live_recipe_1::R2, live_recipe_2::R1) where {R1<:BATMakieRecipe,R2<:BATMakieRecipe}
        return LiveRecipe()
end

function determine_recipe_status(subject::R1, live_recipe_1::R2, live_recipe_2::R3) where {R1<:BATMakieRecipe,R2<:BATMakieRecipe,R3<:BATMakieRecipe}
        return DeadRecipe()
end








function compute_plotting_primitives end

# Every 2D recipe's compose_plotspecs method must accept a
# `transposed::Bool=false` keyword (swap x/y when true) -- lower-triangle
# cells render the shared per-pair primitive transposed to follow the
# standard pair-plot orientation (see _init_gridlayout). Keywords don't
# participate in dispatch, so forgetting it on a new 2D recipe fails loudly
# (MethodError on the first lower-cell render, already at precompile time for
# recipes in BAT_MAKIE_RECIPES_2D) rather than silently mis-orienting.
function compose_plotspecs end


# Truncates a requested vsel to the actual number of free parameters and to at
# most N_max entries (the fixed grid size), warning if that changes anything --
# avoids indexing a variable that doesn't exist, or a grid cell that doesn't.
function _clamp_vsel(vsel::AbstractVector{<:Integer}, n_dof::Integer, N_max::Integer)
        # Both bounds, not just the upper one: 0/negative indices previously
        # passed straight through and produced an uncontextualized BoundsError
        # deep inside a compute-graph closure (the marg-view/domain lookups)
        # instead of this function's own clear warning.
        vsel_clamped = filter(i -> 1 <= i <= n_dof, vsel)
        if length(vsel_clamped) > N_max
                vsel_clamped = vsel_clamped[1:N_max]
        end
        if vsel_clamped != vsel
                @warn "Requested vsel indices $vsel are invalid (each must be in 1:$n_dof, at most N_max=$N_max at a time); using $vsel_clamped instead."
        end
        return vsel_clamped
end

_clamp_vsel(vsel::AbstractVector{<:Integer}, samples::DensitySampleVector, N_max::Integer) = _clamp_vsel(vsel, totalndof(varshape(samples)), N_max)
