# This file is a part of BAT.jl, licensed under the MIT License (MIT).

const standard_confidence_vals = [0.683, 0.955, 0.997]
const standard_colors = [:chartreuse2, :yellow, :red]


include("MarginalDist.jl")
include("recipes_MarginalDist_1D.jl")
include("recipes_MarginalDist_2D.jl")
include("MarginalDist_utils.jl")
include("recipes_stats.jl")
include("recipes_samples_overview.jl")
include("recipes_prior_overview.jl")
include("recipes_ahmi.jl")
include("recipes_prior.jl")
include("recipes_samples_1D.jl")
include("recipes_samples_2D.jl")
include("recipes_diagnostics.jl")
include("valueshapes_utils.jl")
