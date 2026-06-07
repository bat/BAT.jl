module BATMakieExt

using BAT
BAT.pkgext(::Val{:Makie}) = BAT.PackageExtension{:Makie}()

using ArraysOfArrays
using Colors
using Distributions
using KernelDensity
using LinearAlgebra
using Makie
import Makie.SpecApi as S

using Makie: ComputeGraph, add_input!, register_computation!

using Parameters: @with_kw
using StatsBase
using ValueShapes


using BAT: DensitySampleVector
using BAT: MCMCState, MCMCChainState
using BAT: AbstractSamplingAlgorithm
using BAT: _get_edges

using InverseFunctions: inverse

using BAT: BATMakieRecipe
using BAT: Hist1D, Hist2D, QuantileHist1D, QuantileHist2D, Hexbin2D
using BAT: Scatter2D
using BAT: KDE1D, KDE2D, QuantileKDE1D, QuantileKDE2D
using BAT: Cov2D, Std1D, Std2D, Mean1D, Mean2D, Errorbars1D, Errorbars2D, PDF1D

using BAT: BATVisualizer, BATMakieVisualization

import BAT: BATVisualizer
import BAT: init_visualizer!!, register_state_for_vis!, activate_visualizer, update_visualizer_impl!
import BAT: bat_makie_plot, bat_makie_plot!

include("./makie_impl/makie_plotting.jl")

end
