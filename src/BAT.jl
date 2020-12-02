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
using Clustering
using Colors
using Distributions
using DoubleFloats
using ElasticArrays
using FFTW
using FillArrays
using IntervalSets
using KernelDensity
using LaTeXStrings
using ParallelProcessingTools
using Parameters
using PDMats
using PositiveFactorizations
using ProgressMeter
using RecipesBase
using Requires
using StatsBase
using StructArrays
using Tables
using ValueShapes

import AdvancedHMC
import DiffResults
import DistributionsAD
import EmpiricalDistributions
import ForwardDiff
import HypothesisTests
import Measurements
import NamedArrays
import NLSolversBase
import Optim
import Random123
import Sobol
import TypedTables

#for AHMI
using DataStructures
using QuadGK

#for Space Partitioning
import CPUTime


include("utils/utils.jl")
include("rngs/rngs.jl")
include("distributions/distributions.jl")
include("parameters/parameters.jl")
include("transforms/transforms.jl")
include("densities/densities.jl")
include("algotypes/algotypes.jl")
include("initvals/initvals.jl")
include("statistics/statistics.jl")
include("optimization/optimization.jl")
include("samplers/samplers.jl")
include("integration/integration.jl")
include("algodefaults/algodefaults.jl")
include("io/io.jl")
include("plotting/plotting.jl")


const _PLOTS_MODULE = Ref{Union{Module,Nothing}}(nothing)
_plots_module() = _PLOTS_MODULE[]

function __init__()
    @require Cuba = "8a292aeb-7a57-582c-b821-06e4c11590b1" include("integration/cuba_integration.jl")
    @require HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f" include("io/hdf5_specific.jl")
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" _PLOTS_MODULE[] = Plots
end

end # module
