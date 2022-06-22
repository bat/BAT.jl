# This file is a part of BAT.jl, licensed under the MIT License (MIT).

__precompile__(true)

module BAT

using Base.Threads

# import Base.Math.JuliaLibm

using Dates
using Distributed
using LinearAlgebra
using Markdown
using Printf
using Random
using Statistics

using ArgCheck
using ArraysOfArrays
using ChangesOfVariables
using Clustering
using Colors
using DensityInterface
using Distributions
using DocStringExtensions
using DoubleFloats
using ElasticArrays
using FFTW
using FillArrays
using ForwardDiffPullbacks
using IntervalSets
using InverseFunctions
using KernelDensity
using LaTeXStrings
using MacroTools
using ParallelProcessingTools
using Parameters
using PDMats
using PositiveFactorizations
using RecipesBase
using Requires
using StaticArrays
using StatsBase
using StructArrays
using Tables
using ValueShapes

import AdvancedHMC
import ChainRulesCore
import DiffResults
import DistributionMeasures
import DistributionsAD
import EmpiricalDistributions
import FiniteDiff
import ForwardDiff
import HypothesisTests
import MeasureBase
import Measurements
import NamedArrays
import NLSolversBase
import Optim
import Random123
import Sobol
import StableRNGs
import TypedTables
import Zygote
import ZygoteRules

using MeasureBase: AbstractMeasure, DensityMeasure
using MeasureBase: basemeasure, getdof

using DistributionMeasures: DistributionMeasure

using ChainRulesCore: AbstractTangent, Tangent, NoTangent, ZeroTangent, AbstractThunk, unthunk

# For GaussianShell:
import AdaptiveRejectionSampling
import QuadGK
import SpecialFunctions


include("utils/utils.jl")
include("rngs/rngs.jl")
include("distributions/distributions.jl")
include("variates/variates.jl")
include("transforms/transforms.jl")
include("densities/densities.jl")
include("algotypes/algotypes.jl")
include("initvals/initvals.jl")
include("statistics/statistics.jl")
include("differentiation/differentiation.jl")
include("optimization/optimization.jl")
include("samplers/samplers.jl")
include("integration/integration.jl")
include("algodefaults/algodefaults.jl")
include("io/io.jl")
include("plotting/plotting.jl")
include("deprecations.jl")

# include("precompile.jl")


const _PLOTS_MODULE = Ref{Union{Module,Nothing}}(nothing)
_plots_module() = _PLOTS_MODULE[]

function __init__()
    @require Folds = "41a02a25-b8f0-4f67-bc48-60067656b558" @require Transducers = "28d57a85-8fef-5791-bfe6-a80928e7c999" include("utils/executors_folds.jl")
    @require Cuba = "8a292aeb-7a57-582c-b821-06e4c11590b1" include("integration/cuba_integration.jl")
    @require HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f" include("io/hdf5_specific.jl")
    @require NestedSamplers = "41ceaf6f-1696-4a54-9b49-2e7a9ec3782e" include("samplers/nested_sampling/ellipsoidal_nested_sampling/ellipsoidal_nested_sampling.jl")
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" _PLOTS_MODULE[] = Plots
    @require UltraNest = "6822f173-b0be-4018-9ee2-28bf56348d09" include("samplers/nested_sampling/ultranest.jl")
end

end # module
