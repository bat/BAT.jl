# This file is a part of BAT.jl, licensed under the MIT License (MIT).


BAT.bat_default(::Type{TransformedMCMC}, ::Val{:pretransform}, proposal::HamiltonianMC) = PriorToNormal()

BAT.bat_default(::Type{TransformedMCMC}, ::Val{:proposal_tuning}, proposal::HamiltonianMC) = StanHMCTuning()

BAT.bat_default(::Type{TransformedMCMC}, ::Val{:adaptive_transform}, proposal::HamiltonianMC) = NoAdaptiveTransform()

BAT.bat_default(::Type{TransformedMCMC}, ::Val{:tempering}, proposal::HamiltonianMC) = NoMCMCTempering()

BAT.bat_default(::Type{TransformedMCMC}, ::Val{:nsteps}, proposal::HamiltonianMC, pretransform::AbstractTransformTarget, nchains::Integer) = 10^4

BAT.bat_default(::Type{TransformedMCMC}, ::Val{:init}, proposal::HamiltonianMC, pretransform::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCChainPoolInit(nsteps_init = 25) # clamp(div(nsteps, 100), 25, 250)

BAT.bat_default(::Type{TransformedMCMC}, ::Val{:burnin}, proposal::HamiltonianMC, pretransform::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCMultiCycleBurnin(nsteps_per_cycle = max(div(nsteps, 10), 250), max_ncycles = 4)


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
        transition
    )
end


function BAT._get_sample_id(proposal::HMCProposalState, id::Int32, cycle::Int32, stepno::Integer, sample_type::Integer)
    return AHMCSampleID(id, cycle, stepno, sample_type, 0.0, 0, false, 0.0), AHMCSampleID
end

function BAT.next_cycle!(mc_state::HMCState)
    _cleanup_samples(mc_state)

    mc_state.info = MCMCChainStateInfo(mc_state.info, cycle = mc_state.info.cycle + 1)
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

# TODO: MD, should this be a !! function?  
function BAT.mcmc_propose!!(mc_state::HMCState)
    # @unpack target, proposal, f_transform, samples, context = mc_state
    target = mc_state.target
    proposal = mc_state.proposal
    f_transform = mc_state.f_transform
    samples = mc_state.samples
    context = mc_state.context

    rng = get_rng(context)

    current = _current_sample_idx(mc_state)
    proposed = _proposed_sample_idx(mc_state)

    x_current = samples.v[current]
    x_proposed = samples.v[proposed]
    current_log_posterior = samples.logd[current]


    proposal.transition = AdvancedHMC.transition(rng, proposal.hamiltonian, proposal.kernel, proposal.transition.z)
    x_proposed[:] = proposal.transition.z.θ

    proposed_log_posterior = logdensityof(target, x_proposed)
    
    samples.logd[proposed] = proposed_log_posterior

    accepted = x_current != x_proposed

    # TODO: Setting p_accept to 1 or 0 for now.
    # Use AdvancedHMC.stat(transition).acceptance_rate in the future?
    p_accept = Float64(accepted)

    return mc_state, accepted, p_accept
end

function BAT._accept_reject!(mc_state::HMCState, accepted::Bool, p_accept::Float64, current::Integer, proposed::Integer)
    # @unpack samples, proposal = mc_state
    samples = mc_state.samples
    proposal = mc_state.proposal

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

    delta_w_current, w_proposed = BAT.mcmc_weight_values(mc_state.weighting, p_accept, accepted)
    
    samples.weight[current] += delta_w_current
    samples.weight[proposed] = w_proposed
end


BAT.eff_acceptance_ratio(mc_state::HMCState) = nsamples(mc_state) / nsteps(mc_state)
