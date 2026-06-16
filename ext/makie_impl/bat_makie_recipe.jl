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

function compose_plotspecs end
