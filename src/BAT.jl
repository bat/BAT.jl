# This file is a part of BAT.jl, licensed under the MIT License (MIT).

__precompile__(true)

module BAT

using Base.Threads

# import Base.Math.JuliaLibm

using Distributed
using LinearAlgebra
using Markdown
using Random
using Statistics

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
using StatsBase

import RandomNumbers


include("Logging/Logging.jl")
using BAT.Logging

include("utils/utils.jl")
include("rng/rng.jl")
include("distributions/distributions.jl")
include("scheduling/scheduling.jl")
include("parameters/parameters.jl")
include("statistics/statistics.jl")
include("densities/densities.jl")
include("samplers/samplers.jl")
include("plotting/plotting.jl")

Logging.@enable_logging

end # module
