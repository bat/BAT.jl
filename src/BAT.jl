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
using DocStringExtensions
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
include("variates/variates.jl")
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

precompile(bat_sample, (PosteriorDensity{BAT.DistributionDensity{MvNormal{Float64,PDMats.PDiagMat{Float64,Array{Float64,1}},FillArrays.Zeros{Float64,1,Tuple{Base.OneTo{Int64}}}},BAT.HyperRectBounds{Float32}},BAT.DistributionDensity{NamedTupleDist{(:a, :b),Tuple{Weibull{Float64},MvNormal{Float64,PDMats.PDiagMat{Float64,Array{Float64,1}},FillArrays.Zeros{Float64,1,Tuple{Base.OneTo{Int64}}}}},Tuple{ValueAccessor{ScalarShape{Real}},ValueAccessor{ArrayShape{Real,1}}}},BAT.HyperRectBounds{Float64}},BAT.HyperRectBounds{Float32},NamedTupleShape{(:a, :b),Tuple{ValueAccessor{ScalarShape{Real}},ValueAccessor{ArrayShape{Real,1}}}}}, MCMCSampling{MetropolisHastings{BAT.MvTDistProposal,RepetitionWeighting{Int64},AdaptiveMHTuning},PriorToGaussian,MCMCChainPoolInit,MCMCMultiCycleBurnin,BrooksGelmanConvergence,typeof(BAT.nop_func)}))
precompile(bat_sample, (PosteriorDensity{BAT.DistributionDensity{MvNormal{Float64,PDiagMat{Float64,Array{Float64,1}},FillArrays.Zeros{Float64,1,Tuple{Base.OneTo{Int64}}}},BAT.HyperRectBounds{Float32}},BAT.DistributionDensity{NamedTupleDist{(:a, :b),Tuple{Weibull{Float64},MvNormal{Float64,PDiagMat{Float64,Array{Float64,1}},FillArrays.Zeros{Float64,1,Tuple{Base.OneTo{Int64}}}}},Tuple{ValueAccessor{ScalarShape{Real}},ValueAccessor{ArrayShape{Real,1}}}},BAT.HyperRectBounds{Float64}},BAT.HyperRectBounds{Float32},NamedTupleShape{(:a, :b),Tuple{ValueAccessor{ScalarShape{Real}},ValueAccessor{ArrayShape{Real,1}}}}}, MCMCSampling{HamiltonianMC,PriorToGaussian,MCMCChainPoolInit,MCMCMultiCycleBurnin,BrooksGelmanConvergence,typeof(BAT.nop_func)}))


const _PLOTS_MODULE = Ref{Union{Module,Nothing}}(nothing)
_plots_module() = _PLOTS_MODULE[]

function __init__()
    @require Cuba = "8a292aeb-7a57-582c-b821-06e4c11590b1" include("integration/cuba_integration.jl")
    @require HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f" include("io/hdf5_specific.jl")
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" _PLOTS_MODULE[] = Plots
    @require UltraNest = "6822f173-b0be-4018-9ee2-28bf56348d09" include("samplers/nested_sampling/ultranest.jl")
end

end # module
