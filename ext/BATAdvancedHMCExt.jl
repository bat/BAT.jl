# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATAdvancedHMCExt

@static if isdefined(Base, :get_extension)
    using AdvancedHMC
else
    using ..AdvancedHMC
end

using BAT
BAT.pkgext(::Val{:AdvancedHMC}) = BAT.PackageExtension{:AdvancedHMC}()

using Random
using DensityInterface
using HeterogeneousComputing, AutoDiffOperators

using BAT: BATMeasure

using BAT: get_context, get_adselector, _NoADSelected
using BAT: getalgorithm, getmeasure
using BAT: MCMCIterator, MCMCIteratorInfo, MCMCChainPoolInit, MCMCMultiCycleBurnin, AbstractMCMCTunerInstance
using BAT: AbstractTransformTarget
using BAT: RNGPartition, set_rng!
using BAT: mcmc_step!, nsamples, nsteps, samples_available, eff_acceptance_ratio
using BAT: get_samples!, get_mcmc_tuning, reset_rng_counters!
using BAT: tuning_init!, tuning_postinit!, tuning_reinit!, tuning_update!, tuning_finalize!, tuning_callback
using BAT: totalndof, checked_logdensityof
using BAT: CURRENT_SAMPLE, PROPOSED_SAMPLE, INVALID_SAMPLE, ACCEPTED_SAMPLE, REJECTED_SAMPLE

using BAT: HamiltonianMC
using BAT: AHMCSampleID, AHMCSampleIDVector
using BAT: HMCIntegrator,LeapfrogIntegrator,JitteredLeapfrogIntegrator,TemperedLeapfrogIntegrator
using BAT: HMCProposal, FixedStepNumber, FixedTrajectoryLength, NUTSProposal
using BAT: HMCMetric, DiagEuclideanMetric, UnitEuclideanMetric, DenseEuclideanMetric
using BAT: HMCTuningAlgorithm, MassMatrixAdaptor, StepSizeAdaptor, NaiveHMCTuning, StanHMCTuning

include("ahmc_impl/ahmc_config_impl.jl")
include("ahmc_impl/ahmc_sampler_impl.jl")
include("ahmc_impl/ahmc_tuner_impl.jl")


end # module BATAdvancedHMCExt
