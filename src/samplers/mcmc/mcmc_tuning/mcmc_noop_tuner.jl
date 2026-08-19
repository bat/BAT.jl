# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    NoMCMCTransformTuning <: MCMCTransformTuning

Do not perform any MCMC transform tuning.
"""
struct NoMCMCTransformTuning <: MCMCTransformTuning end
export NoMCMCTransformTuning

struct NoMCMCTransformTuningState <: MCMCTransformTunerState end

create_trafo_tuner_state(
    ::NoMCMCTransformTuning,
    ::MCMCChainState,
    ::Integer
) = NoMCMCTransformTuningState()

"""
    NoMCMCProposalTuning <: MCMCProposalTuning

Do not perform any MCMC proposal tuning.
"""
struct NoMCMCProposalTuning <: MCMCProposalTuning end
export NoMCMCProposalTuning

struct NoMCMCProposalTunerState <: MCMCProposalTunerState end

create_proposal_tuner_state(
    ::NoMCMCProposalTuning,
    ::MCMCChainState,
    ::MCMCProposalState,
    ::Integer
) = NoMCMCProposalTunerState()


# Tuner states installed by mcmc_tuning_finalize!!: tuning has finished
# for good, post-tuning stabilization and retained sampling must run a
# fixed transition kernel. All tuning hooks hit their generic no-op
# fallbacks for these states, so neither the space transformation nor
# proposal parameters (like the HMC step size) can change anymore:
struct FrozenMCMCTransformTunerState <: MCMCTransformTunerState end

struct FrozenMCMCProposalTunerState <: MCMCProposalTunerState end
