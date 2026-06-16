# This file is a part of BAT.jl, licensed under the MIT License (MIT).

const default_credibilities = [0.683, 0.955, 0.997]
const default_colors = [:chartreuse2, :yellow, :red]


# Implemented in BATPlotsExt:
function _Plots_backend end
function _Plots_cgrad end
function _Plots_grid end
function _Plots_Shape end
function _Plots_Surface end
function _Plots_backend_is_pyplot end


include("MarginalDist.jl")
include("recipes_MarginalDist_1D.jl")
include("recipes_MarginalDist_2D.jl")
include("MarginalDist_utils.jl")
include("recipes_stats.jl")
include("recipes_samples_overview.jl")
include("recipes_prior_overview.jl")
include("recipes_prior.jl")
include("recipes_samples_1D.jl")
include("recipes_samples_2D.jl")
include("recipes_diagnostics.jl")
include("valueshapes_utils_internal.jl")
include("vsel_processing.jl")
