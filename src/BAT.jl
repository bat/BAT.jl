# This file is a part of BAT.jl, licensed under the MIT License (MIT).

__precompile__(true)

module BAT

using Base.Threads

import Base.Math.JuliaLibm


using Clustering
using Colors
using Distributions
using DoubleDouble
using ElasticArrays
using FunctionWrappers
using IntervalSets
using MultiThreadingTools
using Parameters
using PDMats
using RecipesBase
using StatsBase

import RandomNumbers


include("logging.jl")
using BAT.Logging

include("shims.jl")
include("rng.jl")
include("distributions.jl")
include("util.jl")
include("execcontext.jl")
include("onlineuvstats.jl")
include("onlinemvstats.jl")
include("spatialvolume.jl")
include("parambounds.jl")
include("proposaldist.jl")
include("density.jl")
include("const_density.jl")
include("density_product.jl")
include("product_density.jl")
include("mvdist_density.jl")
include("algorithms.jl")
include("data_vector.jl")
include("density_sample.jl")
include("mcmc_algorithm.jl")
include("mcmc_sampleid.jl")
include("mcmc_stats.jl")
include("mcmc_convergence.jl")
include("mcmc_tuner.jl")
include("mcmc_accrejstate.jl")
include("mh_sampler.jl")
include("tjelmeland.jl")
include("mh_tuner.jl")
include("direct_sampler.jl")
include("mcmc_rand.jl")
include("plots_recipes.jl")

Logging.@enable_logging

end # module
