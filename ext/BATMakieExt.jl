module BATMakieExt

using BAT
BAT.pkgext(::Val{:Makie}) = BAT.PackageExtension{:Makie}()

using ArraysOfArrays
using Colors
using Distributions
using LinearAlgebra
using Makie
import Makie.SpecApi as S
using Parameters: @with_kw
using StatsBase
using ValueShapes


using BAT: DensitySampleVector
using BAT: _get_edges
# using BAT: MarginalDist, BAT.get_bin_centers, get_smallest_intervals, drop_low_weight_samples
# using BAT: bat_marginalize
using BAT: asindex, getstring

using BAT: BATMeasure
using BAT: result_with_args
using BAT: mcmc_init!, mcmc_burnin!, mcmc_iterate!!, next_cycle!
using BAT: get_samples!
using BAT: transform_and_unshape, apply_trafo_to_init, nop_func, convert_for
using BAT: _empty_chain_outputs, _merge_chain_outputs, transform_samples
using BAT: MCMCSampleGenerator

using InverseFunctions: inverse

import BAT: BATMakieVisualizer, init_visualizer!!, update_visualizer_impl
import BAT: AbstractSamplingAlgorithm
import BAT: bat_makie_plot, bat_makie_plot!
import BAT: hist1d, hist2d


include("./makie_impl/makie_plotting.jl")

end
