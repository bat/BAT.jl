# This file is a part of BAT.jl, licensed under the MIT License (MIT).


BAT.bat_default(::Type{TransformedMCMC}, ::Val{:pre_transform}, proposal::HamiltonianMC) = PriorToGaussian()

BAT.bat_default(::Type{TransformedMCMC}, ::Val{:proposal_tuning}, proposal::HamiltonianMC) = StanHMCTuning()

BAT.bat_default(::Type{TransformedMCMC}, ::Val{:adaptive_transform}, proposal::HamiltonianMC) = NoAdaptiveTransform()

BAT.bat_default(::Type{TransformedMCMC}, ::Val{:tempering}, proposal::HamiltonianMC) = NoMCMCTempering()

BAT.bat_default(::Type{TransformedMCMC}, ::Val{:nsteps}, proposal::HamiltonianMC, pre_transform::AbstractTransformTarget, nchains::Integer) = 10^4

BAT.bat_default(::Type{TransformedMCMC}, ::Val{:init}, proposal::HamiltonianMC, pre_transform::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCChainPoolInit(nsteps_init = 25) # clamp(div(nsteps, 100), 25, 250)

BAT.bat_default(::Type{TransformedMCMC}, ::Val{:burnin}, proposal::HamiltonianMC, pre_transform::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
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
    τ = Trajectory{MultinomialTS}(integrator, termination)

    # TODO: MD, remove, for debugging
    init_rng = deepcopy(rng)

    z = AdvancedHMC.phasepoint(hamiltonian, init_transition.z.θ, rand(init_rng, hamiltonian.metric, hamiltonian.kinetic))

    # Perform a dummy step to get type-stable transition value:
    transition = AdvancedHMC.transition(init_rng, deepcopy(τ), deepcopy(hamiltonian), z)

    HMCProposalState(
        integrator,
        termination,
        hamiltonian,
        τ,
        transition
    )
end


function BAT._get_sample_id(proposal::HMCProposalState, id::Int32, cycle::Int32, stepno::Integer, sample_type::Integer)
    return AHMCSampleID(id, cycle, stepno, sample_type, 0.0, 0, false, 0.0), AHMCSampleID
end

function BAT.next_cycle!(chain_state::HMCState)
    _cleanup_samples(chain_state)

    chain_state.info = MCMCChainStateInfo(chain_state.info, cycle = chain_state.info.cycle + 1)
    chain_state.nsamples = 0
    chain_state.stepno = 0

    reset_rng_counters!(chain_state)

    resize!(chain_state.samples, 1)

    i = _proposed_sample_idx(chain_state)
    @assert chain_state.samples.info[i].sampletype == CURRENT_SAMPLE
    chain_state.samples.weight[i] = 1

    t_stat = chain_state.proposal.transition.stat
    
    chain_state.samples.info[i] = AHMCSampleID(
        chain_state.info.id, chain_state.info.cycle, chain_state.stepno, CURRENT_SAMPLE,
        t_stat.hamiltonian_energy, t_stat.tree_depth,
        t_stat.numerical_error, t_stat.step_size
    )

    chain_state
end

# TODO: MD, Make Properly !!
function BAT.mcmc_propose!!(chain_state::HMCState)
    target = chain_state.target
    proposal = chain_state.proposal
    f_transform = chain_state.f_transform
    samples = chain_state.samples
    context = chain_state.context

    rng = get_rng(context)

    current = _current_sample_idx(chain_state)
    proposed = _proposed_sample_idx(chain_state)

    x_current = samples.v[current]
    x_proposed = samples.v[proposed]
    current_log_posterior = samples.logd[current]
    


    τ = deepcopy(proposal.τ)
    @reset τ.integrator = AdvancedHMC.jitter(rng, τ.integrator)
    
    hamiltonian = proposal.hamiltonian
    z = AdvancedHMC.phasepoint(hamiltonian, Vector(x_current), rand(rng, hamiltonian.metric, hamiltonian.kinetic))

    trans = AdvancedHMC.transition(rng, τ, hamiltonian, z)
    tstat = AdvancedHMC.stat(trans)
    p_accept = tstat.acceptance_rate
    x_proposed[:] = trans.z.θ

    logd_x_proposed = logdensityof(target, x_proposed)

    samples.logd[proposed] = logd_x_proposed
    
    accepted = x_current != x_proposed

    chain_state_new = @set chain_state.samples.v[proposed] = x_proposed
    chain_state_new = @set chain_state.samples.logd[proposed] = logd_x_proposed
    chain_state_new.proposal.transition = trans
    # chain_state_new = @set chain_state.proposal.transition = trans # For some reason this doesn't save the new transition. Do @reset for each field of trans?

    return chain_state_new, accepted, p_accept
end

function BAT._accept_reject!(chain_state::HMCState, accepted::Bool, p_accept::Float64, current::Integer, proposed::Integer)
    samples = chain_state.samples
    proposal = chain_state.proposal

    if accepted
        samples.info.sampletype[current] = ACCEPTED_SAMPLE
        samples.info.sampletype[proposed] = CURRENT_SAMPLE
        chain_state.nsamples += 1

        tstat = AdvancedHMC.stat(proposal.transition)
        samples.info.hamiltonian_energy[proposed] = tstat.hamiltonian_energy
        # ToDo: Handle proposal-dependent tstat (only NUTS has tree_depth):
        samples.info.tree_depth[proposed] = tstat.tree_depth
        samples.info.divergent[proposed] = tstat.numerical_error
        samples.info.step_size[proposed] = tstat.step_size
    else
        samples.info.sampletype[proposed] = REJECTED_SAMPLE
    end

    delta_w_current, w_proposed = BAT.mcmc_weight_values(chain_state.weighting, p_accept, accepted)
    
    samples.weight[current] += delta_w_current
    samples.weight[proposed] = w_proposed
end

BAT.eff_acceptance_ratio(mc_state::HMCState) = nsamples(mc_state) / nsteps(mc_state)
