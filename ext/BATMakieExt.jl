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
using Random
using Makie
import Makie.SpecApi as S

using Makie: ComputeGraph, add_input!, register_computation!
using Makie.Observables: observe, setexcludinghandlers!

using StatsBase
using Statistics: quantile
using ValueShapes
using PrecompileTools: @setup_workload, @compile_workload


using BAT: DensitySampleVector
using BAT: _empty_chain_outputs, get_samples!
using BAT: MCMCState, MCMCChainState
using BAT: mcmc_target
using BAT: BATDistMeasure, BATWeightedMeasure, getprior
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
using BAT: _default_makie_triagonal_config, _default_makie_diagonal_config

using BAT: init_visualizer!, register_state_for_vis!, update_visualizer_impl!
using BAT: bat_makie_plot, bat_theme, bat_theme_dark

include("./makie_impl/makie_plotting.jl")

function __init__()
    # Compile-time rendering does not warm resources in the loading process.
    empty!(_WARMED_BACKENDS)
    return nothing
end


# Compile the entry points, recipes and controls exercised in makie_precompile.jl.
@setup_workload begin
    @compile_workload begin
        _makie_precompile_workload()
    end
end

end
