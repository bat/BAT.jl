# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATAdvancedHMCExt

using AdvancedHMC

using BAT
BAT.pkgext(::Val{:AdvancedHMC}) = BAT.PackageExtension{:AdvancedHMC}()

using Random
using DensityInterface
using HeterogeneousComputing, AutoDiffOperators

using BAT: MeasureLike, BATMeasure

using BAT: get_context, get_adselector, _NoADSelected
using BAT: getalgorithm, mcmc_target
using BAT: MCMCIterator, MCMCIteratorInfo, MCMCChainPoolInit, MCMCMultiCycleBurnin, AbstractMCMCTunerInstance
using BAT: AbstractTransformTarget
using BAT: RNGPartition, set_rng!
using BAT: mcmc_step!, nsamples, nsteps, samples_available, eff_acceptance_ratio
using BAT: get_samples!, get_mcmc_tuning, reset_rng_counters!
using BAT: tuning_init!, tuning_postinit!, tuning_reinit!, tuning_update!, tuning_finalize!, tuning_callback
using BAT: totalndof, measure_support, checked_logdensityof
using BAT: CURRENT_SAMPLE, PROPOSED_SAMPLE, INVALID_SAMPLE, ACCEPTED_SAMPLE, REJECTED_SAMPLE

using BAT: HamiltonianMC
using BAT: AHMCSampleID, AHMCSampleIDVector
using BAT: HMCMetric, DiagEuclideanMetric, UnitEuclideanMetric, DenseEuclideanMetric
using BAT: HMCTuningAlgorithm, MassMatrixAdaptor, StepSizeAdaptor, NaiveHMCTuning, StanHMCTuning

using ValueShapes: varshape

using Accessors: @set


BAT.ext_default(::BAT.PackageExtension{:AdvancedHMC}, ::Val{:DEFAULT_INTEGRATOR}) = AdvancedHMC.Leapfrog(NaN)
BAT.ext_default(::BAT.PackageExtension{:AdvancedHMC}, ::Val{:DEFAULT_TERMINATION_CRITERION}) = AdvancedHMC.GeneralisedNoUTurn()


include("ahmc_impl/ahmc_config_impl.jl")
include("ahmc_impl/ahmc_sampler_impl.jl")
include("ahmc_impl/ahmc_tuner_impl.jl")


end # module BATAdvancedHMCExt
