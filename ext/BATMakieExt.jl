module BATMakieExt

using ArraysOfArrays
using BAT
using Colors
using Distributions
using LinearAlgebra
using Makie
using StatsBase
using ValueShapes

using BAT: MarginalDist, BAT.get_bin_centers, get_smallest_intervals, drop_low_weight_samples
using BAT: bat_marginalize
using BAT: asindex, getstring

using BAT: BATMeasure
using BAT: result_with_args
using BAT: mcmc_init!, mcmc_burnin!, mcmc_iterate!!, next_cycle!
using BAT: get_samples!
using BAT: transform_and_unshape, apply_trafo_to_init, nop_func, convert_for
using BAT: _empty_chain_outputs, _merge_chain_outputs, transform_samples
using BAT: MCMCSampleGenerator

using InverseFunctions: inverse

import BAT: bat_makie_plot, bat_makie_plot!
import BAT: bat_sample_and_visualize

include("./makie_impl/makie_plotting.jl")

end
