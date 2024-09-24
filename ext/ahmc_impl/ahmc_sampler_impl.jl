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
