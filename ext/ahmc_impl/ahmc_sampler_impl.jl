# This file is a part of BAT.jl, licensed under the MIT License (MIT).


BAT.bat_default(::Type{MCMCSampling}, ::Val{:trafo}, mcalg::HamiltonianMC) = PriorToGaussian()

BAT.bat_default(::Type{MCMCSampling}, ::Val{:nsteps}, mcalg::HamiltonianMC, trafo::AbstractTransformTarget, nchains::Integer) = 10^4

BAT.bat_default(::Type{MCMCSampling}, ::Val{:init}, mcalg::HamiltonianMC, trafo::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCChainPoolInit(nsteps_init = 25) # clamp(div(nsteps, 100), 25, 250)

BAT.bat_default(::Type{MCMCSampling}, ::Val{:burnin}, mcalg::HamiltonianMC, trafo::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCMultiCycleBurnin(nsteps_per_cycle = max(div(nsteps, 10), 250), max_ncycles = 4)


BAT.get_mcmc_tuning(algorithm::HamiltonianMC) = algorithm.tuning



function BAT._create_proposal_state(
    proposal::HamiltonianMC, 
    target::BATMeasure, 
    context::BATContext, 
    v_init::AbstractVector{P}, 
    rng::AbstractRNG
) where {P<:Real}
    vs = varshape(target)
    npar = totalndof(vs)

    params_vec = Vector{P}(undef, npar)
    params_vec .= v_init

    adsel = get_adselector(context)
    f = checked_logdensityof(target)
    fg = valgrad_func(f, adsel)

    metric = ahmc_metric(proposal.metric, params_vec)
    init_hamiltonian = AdvancedHMC.Hamiltonian(metric, f, fg)
    hamiltonian, init_transition = AdvancedHMC.sample_init(rng, init_hamiltonian, params_vec)
    integrator = _ahmc_set_step_size(proposal.integrator, hamiltonian, params_vec)
    termination = _ahmc_convert_termination(proposal.termination, params_vec)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, termination))

    # Perform a dummy step to get type-stable transition value:
    transition = AdvancedHMC.transition(deepcopy(rng), deepcopy(hamiltonian), deepcopy(kernel), init_transition.z)

    HMCProposalState(
        integrator,
        termination,
        hamiltonian,
        kernel,
        transition,
        proposal.weighting
    )
end


function BAT._get_sampleid(proposal::HMCProposalState, id::Int32, cycle::Int32, stepno::Int64, sampletype::Integer)
    return AHMCSampleID(id, cycle, stepno, sampletype, 0.0, 0, false, 0.0), AHMCSampleID
end

# # MCMCState subtype for HamiltonianMC
# mutable struct HMCState{
#     AL<:HamiltonianMC,
#     D<:BATMeasure,
#     PR<:RNGPartition,
#     SV<:DensitySampleVector,
#     HA<:AdvancedHMC.Hamiltonian,
#     TR<:AdvancedHMC.Transition,
#     KRNL<:AdvancedHMC.HMCKernel,
#     CTX<:BATContext
# } <: MCMCState
#     algorithm::AL
#     target::D
#     rngpart_cycle::PR
#     info::MCMCStateInfo
#     samples::SV
#     nsamples::Int64
#     stepno::Int64
#     hamiltonian::HA
#     transition::TR
#     kernel::KRNL
#     context::CTX
# end


# function HMCState(
#     algorithm::HamiltonianMC,
#     target::BATMeasure,
#     info::MCMCStateInfo,
#     x_init::AbstractVector{P},
#     context::BATContext,
# ) where {P<:Real}
#     rng = get_rng(context)
#     stepno::Int64 = 0

#     vs = varshape(target)

#     npar = totalndof(vs)

#     params_vec = Vector{P}(undef, npar)
#     params_vec .= x_init

#     log_posterior_value = checked_logdensityof(target, params_vec)

#     T = typeof(log_posterior_value)
#     W = Float64 # ToDo: Support other sample weight types

#     sample_info = AHMCSampleID(info.id, info.cycle, 1, CURRENT_SAMPLE, 0.0, 0, false, 0.0)
#     current_sample = DensitySample(params_vec, log_posterior_value, one(W), sample_info, nothing)
#     samples = DensitySampleVector{Vector{P},T,W,AHMCSampleID,Nothing}(undef, 0, npar)
#     push!(samples, current_sample)

#     nsamples::Int64 = 0

#     rngpart_cycle = RNGPartition(rng, 0:(typemax(Int16) - 2))

#     metric = ahmc_metric(algorithm.metric, params_vec)

#     # ToDo!: Pass context explicitly:
#     adsel = get_adselector(context)
#     if adsel isa _NoADSelected
#         throw(ErrorException("HamiltonianMC requires an ADSelector to be specified in the BAT context"))
#     end

#     f = checked_logdensityof(target)
#     fg = valgrad_func(f, adsel)

#     init_hamiltonian = AdvancedHMC.Hamiltonian(metric, f, fg)
#     hamiltonian, init_transition = AdvancedHMC.sample_init(rng, init_hamiltonian, params_vec)
#     integrator = _ahmc_set_step_size(algorithm.integrator, hamiltonian, params_vec)
#     termination = _ahmc_convert_termination(algorithm.termination, params_vec)
#     kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, termination))

#     # Perform a dummy step to get type-stable transition value:
#     transition = AdvancedHMC.transition(deepcopy(rng), deepcopy(hamiltonian), deepcopy(kernel), init_transition.z)

#     chain = HMCState(
#         algorithm,
#         target,
#         rngpart_cycle,
#         info,
#         samples,
#         nsamples,
#         stepno,
#         hamiltonian,
#         transition,
#         kernel,
#         context
#     )

#     reset_rng_counters!(chain)

#     chain
# end


# function MCMCState(
#     algorithm::HamiltonianMC,
#     target::BATMeasure,
#     chainid::Integer,
#     startpos::AbstractVector{<:Real},
#     context::BATContext
# )
#     cycle = 0
#     tuned = false
#     converged = false
#     info = MCMCStateInfo(chainid, cycle, tuned, converged)
#     HMCState(algorithm, target, info, startpos, context)
# end


# TODO: MD, unecessary? If no special behaviour is desired, 
#           the methods for general MCMC states cover this
# @inline _current_sample_idx(mc_state::HMCState) = firstindex(mc_state.samples)
# @inline _proposed_sample_idx(mc_state::HMCState) = lastindex(mc_state.samples)


# BAT.getproposal(chain::HMCState) = chain.algorithm

# BAT.mcmc_target(chain::HMCState) = chain.target

# BAT.get_context(chain::HMCState) = chain.context

# BAT.mcmc_info(chain::HMCState) = chain.info

# BAT.nsteps(chain::HMCState) = chain.stepno

# BAT.nsamples(chain::HMCState) = chain.nsamples

# BAT.current_sample(mc_state::HMCState) = mc_state.samples[_current_sample_idx(mc_state)]

# BAT.proposed_sample(mc_state::HMCState) = mc_state.samples[_proposed_sample_idx(mc_state)]

# BAT.sample_type(chain::HMCState) = eltype(chain.samples)



function BAT.reset_rng_counters!(mc_state::HMCState)
    rng = get_rng(get_context(mc_state))
    set_rng!(rng, mc_state.rngpart_cycle, mc_state.info.cycle)
    rngpart_step = RNGPartition(rng, 0:(typemax(Int32) - 2))
    set_rng!(rng, rngpart_step, mc_state.stepno)
    nothing
end


function BAT.samples_available(mc_state::HMCState)
    i = _current_sample_idx(mc_state::HMCState)
    mc_state.samples.info.sampletype[i] == ACCEPTED_SAMPLE
end


function BAT.get_samples!(appendable, mc_state::HMCState, nonzero_weights::Bool)::typeof(appendable)
    if samples_available(mc_state)
        samples = mc_state.samples

        for i in eachindex(samples)
            st = samples.info.sampletype[i]
            if (
                (st == ACCEPTED_SAMPLE || st == REJECTED_SAMPLE) &&
                (samples.weight[i] > 0 || !nonzero_weights)
            )
                push!(appendable, samples[i])
            end
        end
    end
    appendable
end


function BAT.next_cycle!(mc_state::HMCState)
    _cleanup_samples(mc_state)

    mc_state.info = MCMCStateInfo(mc_state.info, cycle = mc_state.info.cycle + 1)
    mc_state.nsamples = 0
    mc_state.stepno = 0

    reset_rng_counters!(mc_state)

    resize!(mc_state.samples, 1)

    i = _proposed_sample_idx(mc_state)
    @assert mc_state.samples.info[i].sampletype == CURRENT_SAMPLE
    mc_state.samples.weight[i] = 1

    t_stat = mc_state.proposal.transition.stat
    
    mc_state.samples.info[i] = AHMCSampleID(
        mc_state.info.id, mc_state.info.cycle, mc_state.stepno, CURRENT_SAMPLE,
        t_stat.hamiltonian_energy, t_stat.tree_depth,
        t_stat.numerical_error, t_stat.step_size
    )

    mc_state
end


function _cleanup_samples(mc_state::HMCState)
    samples = mc_state.samples
    current = _current_sample_idx(mc_state)
    proposed = _proposed_sample_idx(mc_state)
    if (current != proposed) && samples.info.sampletype[proposed] == CURRENT_SAMPLE
        # Proposal was accepted in the last step
        @assert samples.info.sampletype[current] == ACCEPTED_SAMPLE
        samples.v[current] .= samples.v[proposed]
        samples.logd[current] = samples.logd[proposed]
        samples.weight[current] = samples.weight[proposed]
        samples.info[current] = samples.info[proposed]

        resize!(samples, 1)
    end
end


function BAT.mcmc_step!(mc_state::HMCState)
    _cleanup_samples(mc_state)

    samples = mc_state.samples
    proposal = mc_state.proposal

    mc_state.stepno += 1
    reset_rng_counters!(mc_state)

    rng = get_rng(get_context(mc_state))
    target = mcmc_target(mc_state)


    # Grow samples vector by one:
    resize!(samples, size(samples, 1) + 1)
    samples.info[lastindex(samples)] = AHMCSampleID(
        mc_state.info.id, mc_state.info.cycle, mc_state.stepno, PROPOSED_SAMPLE,
        0.0, 0, false, 0.0
    )
    
    current = _current_sample_idx(mc_state)
    proposed = _proposed_sample_idx(mc_state)
    @assert current != proposed

    current_params = samples.v[current]
    proposed_params = samples.v[proposed]

    # Propose new variate:
    samples.weight[proposed] = 0

    proposal.transition = AdvancedHMC.transition(rng, proposal.hamiltonian, proposal.kernel, proposal.transition.z)
    proposed_params[:] = proposal.transition.z.θ
    
    current_log_posterior = samples.logd[current]
    T = typeof(current_log_posterior)

    # Evaluate prior and likelihood with proposed variate:
    proposed_log_posterior = logdensityof(target, proposed_params)

    samples.logd[proposed] = proposed_log_posterior

    accepted = current_params != proposed_params

    if accepted
        samples.info.sampletype[current] = ACCEPTED_SAMPLE
        samples.info.sampletype[proposed] = CURRENT_SAMPLE
        mc_state.nsamples += 1

        tstat = AdvancedHMC.stat(proposal.transition)
        samples.info.hamiltonian_energy[proposed] = tstat.hamiltonian_energy
        # ToDo: Handle proposal-dependent tstat (only NUTS has tree_depth):
        samples.info.tree_depth[proposed] = tstat.tree_depth
        samples.info.divergent[proposed] = tstat.numerical_error
        samples.info.step_size[proposed] = tstat.step_size
    else
        samples.info.sampletype[proposed] = REJECTED_SAMPLE
    end

    delta_w_current, w_proposed = if accepted
        (0, 1)
    else
        (1, 0)
    end
    
    samples.weight[current] += delta_w_current
    samples.weight[proposed] = w_proposed

    nothing
end


BAT.eff_acceptance_ratio(mc_state::HMCState) = nsamples(mc_state) / nsteps(mc_state)
