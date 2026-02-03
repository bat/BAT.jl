module BATMakieExt

using BAT
using Makie
using StatsBase
using ValueShapes

using BAT: MarginalDist, BAT.get_bin_centers, get_smallest_intervals, drop_low_weight_samples
using BAT: bat_marginalize

include("./makie_impl/makie_plotting.jl")

end
