# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATMakieExt

using BAT
BAT.pkgext(::Val{:Makie}) = BAT.PackageExtension{:Makie}()

using ArraysOfArrays
using Colors
using Distributions
using ElasticArrays
using KernelDensity
using LaTeXStrings
using LinearAlgebra
using Makie
import Makie.SpecApi as S

using Makie: ComputeGraph, add_input!, register_computation!
using Makie.Observables: observe, setexcludinghandlers!

using StatsBase
using Statistics: quantile
using ValueShapes
using PrecompileTools: @setup_workload, @compile_workload


using BAT: DensitySampleVector
using BAT: _empty_chain_outputs, _append_chain_outputs, _append_walker_outputs
using BAT: _transform_chain_outputs, _transform_walker_outputs, get_samples!
using BAT: MCMCState, MCMCChainState
using BAT: mcmc_target
using BAT: _get_edges
using BAT: transform_samples

using InverseFunctions: inverse

using BAT: BATMakieRecipe
using BAT: Hist1D, Hist2D, QuantileHist1D, QuantileHist2D, Hexbin2D
using BAT: Scatter2D
using BAT: KDE1D, KDE2D, QuantileKDE1D, QuantileKDE2D
using BAT: Cov2D, Std1D, Std2D, Mean1D, Mean2D, Errorbars1D, Errorbars2D, PDF1D
using BAT: Trace2D

using BAT: BATVisualizer, BATMakieVisualization
using BAT: BasicUvStatistics, BasicMvStatistics

import BAT: BATVisualizer
import BAT: init_visualizer!, register_state_for_vis!, update_visualizer_impl!
import BAT: bat_makie_plot, bat_theme, bat_theme_dark

include("./makie_impl/makie_plotting.jl")

Makie.set_theme!(bat_theme())

# See makie_impl/makie_precompile.jl for what this covers and why (in short:
# every recipe's real compute/compose/SpecApi-reconciliation code path, so
# switching to it interactively for the first time doesn't pay that
# compilation cost then instead).
@setup_workload begin
    @compile_workload begin
        _makie_precompile_workload()
    end
end

end
