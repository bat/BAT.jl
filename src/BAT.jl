# This file is a part of BAT.jl, licensed under the MIT License (MIT).

__precompile__(true)

module BAT

using Base.Threads

# import Base.Math.JuliaLibm

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
using FunctionWrappers
using IntervalSets
using ParallelProcessingTools
using Parameters
using PDMats
using RecipesBase
using StaticArrays
using StatsBase
using UnsafeArrays

#for AHMI
using Cuba
using ProgressMeter
using DataStructures
using LaTeXStrings

import Random123
import TypedTables


include("Logging/Logging.jl")
using BAT.Logging


include("utils/utils.jl")
include("rngs/rngs.jl")
include("distributions/distributions.jl")
include("scheduling/scheduling.jl")
include("parameters/parameters.jl")
include("statistics/statistics.jl")
include("densities/densities.jl")
include("samplers/samplers.jl")
include("integration/integration.jl")
include("plotting/plotting.jl")

Logging.@enable_logging

end # module
