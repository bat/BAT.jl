# This file is a part of BAT.jl, licensed under the MIT License (MIT).

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

import Adapt
using Adapt: adapt

using AffineMaps
using ArgCheck
using ArraysOfArrays
using AutoDiffOperators
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
using FunctionChains
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

import ChainRulesCore
import DiffResults
import DistributionsAD
import EmpiricalDistributions
import HypothesisTests
import Measurements
import NamedArrays
import Random123
import Sobol
import StableRNGs
import StatsFuns
import TypedTables
import ZygoteRules

using Accessors: @set

import HeterogeneousComputing
using HeterogeneousComputing: AbstractComputeUnit, CPUnit
using HeterogeneousComputing: GenContext, get_rng, get_precision, get_compute_unit, get_gencontext

import MeasureBase
using MeasureBase: AbstractMeasure, DensityMeasure, Likelihood
using MeasureBase: basemeasure, getdof, likelihoodof, testvalue
using MeasureBase: transport_to, transport_origin, from_origin, to_origin
using MeasureBase: StdMeasure, StdUniform, StdNormal
using MeasureBase: PowerMeasure, powermeasure, marginals
using MeasureBase: WeightedMeasure, weightedmeasure

using MeasureBase: PushforwardMeasure, gettransform
using MeasureBase: TransformVolCorr as PushFwdStyle, NoVolCorr as ChangeRootMeasure, WithVolCorr as KeepRootMeasure

@static if isdefined(MeasureBase, :pwr_base)
    import MeasureBase.pwr_base as _pwr_base
    import MeasureBase.pwr_axes as _pwr_axes
    import MeasureBase.pwr_size as _pwr_size
else
    _pwr_base(m::PowerMeasure) = m.parent
    _pwr_axes(m::PowerMeasure) = m.axes
    _pwr_size(m::PowerMeasure) = map(length, m.axes)
end


using IntervalSets: Domain

import DomainSets
using DomainSets: UnitInterval, UnitCube, Rectangle, FullSpace, RealNumbers

using ChainRulesCore: AbstractTangent, Tangent, NoTangent, ZeroTangent, AbstractThunk, unthunk

using Functors: fmap, @functor

# For Dual specializations:
import ForwardDiff

# For StandardMvNormal:
using IrrationalConstants: log2π, invsqrt2π


include("utils/utils.jl")
include("rngs/rngs.jl")
include("distributions/distributions.jl")
include("variates/variates.jl")
include("transforms/transforms.jl")
include("densities/densities.jl")
include("measures/measures.jl")
include("algotypes/algotypes.jl")
include("initvals/initvals.jl")
include("statistics/statistics.jl")
include("optimization/optimization.jl")
include("samplers/samplers.jl")
include("integration/integration.jl")
include("algodefaults/algodefaults.jl")
include("plotting/plotting.jl")
include("extdefs/extdefs.jl")
include("deprecations.jl")

# include("precompile.jl")


@static if !isdefined(Base, :get_extension)
    using Requires
end

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require AdvancedHMC = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d" include("../ext/BATAdvancedHMCExt.jl")
        @require Folds = "41a02a25-b8f0-4f67-bc48-60067656b558" @require Transducers = "28d57a85-8fef-5791-bfe6-a80928e7c999" include("../ext/BATFoldsExt.jl")
        @require Cuba = "8a292aeb-7a57-582c-b821-06e4c11590b1" include("../ext/BATCubaExt.jl")
        @require HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f" include("../ext/BATHDF5Ext.jl")
        @require NestedSamplers = "41ceaf6f-1696-4a54-9b49-2e7a9ec3782e" include("../ext/BATNestedSamplersExt.jl")
        @require Optim = "429524aa-4258-5aef-a3af-852621145aeb" include("../ext/BATOptimExt.jl")
        @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("../ext/BATPlotsExt.jl")
        @require UltraNest = "6822f173-b0be-4018-9ee2-28bf56348d09" include("../ext/BATUltraNestExt.jl")
    end
end

end # module
