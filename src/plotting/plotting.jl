# This file is a part of BAT.jl, licensed under the MIT License (MIT).

const standard_confidence_vals = [0.683, 0.955, 0.997]
const standard_colors = [:chartreuse2, :yellow, :red]


include("recipes_stats.jl")
include("recipes_1d_histograms.jl")
include("recipes_2d_histograms.jl")
include("recipes_overview.jl")
include("split_histograms.jl")
include("localmodes.jl")
include("recipes_ahmi.jl")
