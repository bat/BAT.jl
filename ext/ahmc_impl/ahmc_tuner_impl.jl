# This file is a part of BAT.jl, licensed under the MIT License (MIT).


mutable struct HMCProposalTunerState{A<:AdvancedHMC.AbstractAdaptor} <: MCMCProposalTunerState
    tuning::HMCTuning
    target_acceptance::Float64
    adaptor::A
end

function HMCProposalTunerState(tuning::HMCTuning, chain_state::MCMCChainState)
    θ = first(chain_state.current.z).v
    adaptor = ahmc_adaptor(tuning, chain_state.proposal.hamiltonian.metric, chain_state.proposal.kernel.τ.integrator, θ)
    HMCProposalTunerState(tuning, tuning.target_acceptance, adaptor)
end

BAT.create_proposal_tuner_state(tuning::HMCTuning, chain_state::MCMCChainState, iteration::Integer) = HMCProposalTunerState(tuning, chain_state)


function BAT.mcmc_tuning_init!!(tuner::HMCProposalTunerState, chain_state::HMCChainState, max_nsteps::Integer)
    AdvancedHMC.Adaptation.initialize!(tuner.adaptor, Int(max_nsteps - 1))
    nothing
end


function BAT.mcmc_tuning_reinit!!(tuner::HMCProposalTunerState, chain_state::HMCChainState, max_nsteps::Integer)
    AdvancedHMC.Adaptation.initialize!(tuner.adaptor, Int(max_nsteps - 1))
    nothing
end

BAT.mcmc_tuning_postinit!!(tuner::HMCProposalTunerState, chain_state::HMCChainState, samples::AbstractVector{<:DensitySampleVector}) = nothing

function BAT.mcmc_tune_post_cycle!!(tuner::HMCProposalTunerState, chain_state::HMCChainState, samples::AbstractVector{<:DensitySampleVector})
    logds = [walker_smpls.logd for walker_smpls in samples]
    max_log_posterior = maximum(maximum.(logds))
    accept_ratio = eff_acceptance_ratio(chain_state)
    if accept_ratio >= 0.9 * tuner.target_acceptance
        chain_state.info = MCMCChainStateInfo(chain_state.info, tuned = true)
        @debug "MCMC chain $(chain_state.info.id) tuned, acceptance ratio = $(Float32(accept_ratio)), integrator = $(chain_state.proposal.τ.integrator), max. log posterior = $(Float32(max_log_posterior))"
    else
        chain_state.info = MCMCChainStateInfo(chain_state.info, tuned = false)
        @debug "MCMC chain $(chain_state.info.id) *not* tuned, acceptance ratio = $(Float32(accept_ratio)), integrator = $(chain_state.proposal.τ.integrator), max. log posterior = $(Float32(max_log_posterior))"
    end
    return chain_state, tuner
end

function BAT.mcmc_tuning_finalize!!(tuner::HMCProposalTunerState, chain_state::HMCChainState)
    adaptor = tuner.adaptor
    proposal = chain_state.proposal
    AdvancedHMC.finalize!(adaptor)
    proposal.hamiltonian = AdvancedHMC.update(proposal.hamiltonian, adaptor)
    proposal.kernel = AdvancedHMC.update(proposal.kernel, adaptor)
    nothing
end

# TODO: MD, make actually !! function
function BAT.mcmc_tune_post_step!!(
    tuner_state::HMCProposalTunerState,
    chain_state::MCMCChainState,
    p_accept::AbstractVector{<:Real}
)
    adaptor = tuner_state.adaptor
    proposal_new = deepcopy(chain_state.proposal)
    tstat = AdvancedHMC.stat(proposal_new.transition)

    AdvancedHMC.adapt!(adaptor, proposal_new.transition.z.θ, tstat.acceptance_rate)
    h = proposal_new.hamiltonian
    h = AdvancedHMC.update(h, adaptor)
    
    proposal_new.kernel = AdvancedHMC.update(proposal_new.kernel, adaptor)
    tstat = merge(tstat, (is_adapt =true,))

    chain_state_tmp = @set chain_state.proposal.transition.stat = tstat
    chain_state_final = @set chain_state_tmp.proposal = proposal_new

    return chain_state_final, tuner_state
end
