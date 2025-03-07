# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATAdvancedHMCExt

using AdvancedHMC

using BAT
BAT.pkgext(::Val{:AdvancedHMC}) = BAT.PackageExtension{:AdvancedHMC}()

using Random
using DensityInterface
using HeterogeneousComputing, AutoDiffOperators

using Accessors: @set, @reset

using AffineMaps: MulAdd

using BAT: MeasureLike, BATMeasure

using BAT: get_context, get_adselector, _NoADSelected
using BAT: getproposal, mcmc_target
using BAT: MCMCChainState, HMCState, HamiltonianMC, HMCProposalState, MCMCChainStateInfo, MCMCChainPoolInit, MCMCMultiCycleBurnin
using BAT: MCMCBasicStats, push!, reweight_relative!
using BAT: RAMTuning
using BAT: MCMCProposalTunerState, MCMCTransformTunerState, NoMCMCTempering, NoMCMCTransformTuning
using BAT: _current_sample_idx, _proposed_sample_idx, _current_sample_z_idx, _proposed_sample_z_idx, _cleanup_samples, current_sample_z, proposed_sample_z, proposed_sample
using BAT: AbstractTransformTarget, NoAdaptiveTransform, TriangularAffineTransform, valgrad_func
using BAT: RNGPartition, get_rng, set_rng!
using BAT: mcmc_step!!, nsamples, nsteps, samples_available, eff_acceptance_ratio
using BAT: get_samples!, reset_rng_counters!
using BAT: create_trafo_tuner_state, create_proposal_tuner_state, mcmc_tuning_init!!, mcmc_tuning_postinit!!, mcmc_tuning_reinit!!, mcmc_tune_transform_post_cycle!!, transform_mcmc_tuning_finalize!!, set_mc_state_transform!!, mcmc_update_z_position!!
using BAT: totalndof, measure_support, checked_logdensityof
using BAT: CURRENT_SAMPLE, PROPOSED_SAMPLE, INVALID_SAMPLE, ACCEPTED_SAMPLE, REJECTED_SAMPLE

using BAT: HamiltonianMC
using BAT: AHMCSampleID, AHMCSampleIDVector
using BAT: HMCMetric, DiagEuclideanMetric, UnitEuclideanMetric, DenseEuclideanMetric
using BAT: HMCTuning, MassMatrixAdaptor, StepSizeAdaptor, NaiveHMCTuning, StanLikeTuning

using ChangesOfVariables: with_logabsdet_jacobian

using LinearAlgebra: cholesky

using MeasureBase: pullback 

using Parameters: @with_kw

using PositiveFactorizations: Positive

using ValueShapes: varshape

BAT.ext_default(::BAT.PackageExtension{:AdvancedHMC}, ::Val{:DEFAULT_INTEGRATOR}) = AdvancedHMC.Leapfrog(NaN)
BAT.ext_default(::BAT.PackageExtension{:AdvancedHMC}, ::Val{:DEFAULT_TERMINATION_CRITERION}) = AdvancedHMC.GeneralisedNoUTurn()


include("ahmc_impl/ahmc_stan_tuner_impl.jl")
include("ahmc_impl/ahmc_config_impl.jl")
include("ahmc_impl/ahmc_sampler_impl.jl")
include("ahmc_impl/ahmc_tuner_impl.jl")

end # module BATAdvancedHMCExt
