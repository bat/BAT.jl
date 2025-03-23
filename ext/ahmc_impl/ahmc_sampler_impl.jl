# This file is a part of BAT.jl, licensed under the MIT License (MIT).


BAT.bat_default(::Type{TransformedMCMC}, ::Val{:pretransform}, proposal::HamiltonianMC) = PriorToNormal()

BAT.bat_default(::Type{TransformedMCMC}, ::Val{:proposal_tuning}, proposal::HamiltonianMC) = StepSizeAdaptor()

BAT.bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, proposal::HamiltonianMC) = RAMTuning()

BAT.bat_default(::Type{TransformedMCMC}, ::Val{:adaptive_transform}, proposal::HamiltonianMC) = TriangularAffineTransform()

BAT.bat_default(::Type{TransformedMCMC}, ::Val{:tempering}, proposal::HamiltonianMC) = NoMCMCTempering()

BAT.bat_default(::Type{TransformedMCMC}, ::Val{:nsteps}, proposal::HamiltonianMC, pretransform::AbstractTransformTarget, nchains::Integer) = 10^4

BAT.bat_default(::Type{TransformedMCMC}, ::Val{:init}, proposal::HamiltonianMC, pretransform::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCChainPoolInit(nsteps_init = 25) # clamp(div(nsteps, 100), 25, 250)

BAT.bat_default(::Type{TransformedMCMC}, ::Val{:burnin}, proposal::HamiltonianMC, pretransform::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCMultiCycleBurnin(nsteps_per_cycle = max(div(nsteps, 10), 250), max_ncycles = 4)

# Change to incorporate the initial adaptive transform into f and fg
function BAT._create_proposal_state(
    proposal::HamiltonianMC, 
    target::BATMeasure, 
    context::BATContext, 
    v_init::AbstractVector{P}, 
    f_transform::Function,
    rng::AbstractRNG
) where {P<:Real}
    vs = varshape(target)
    npar = totalndof(vs)

    params_vec = Vector{P}(undef, npar)
    params_vec .= v_init

    adsel = get_adselector(context)
    f = checked_logdensityof(pullback(f_transform, target))
    fg = valgrad_func(f, adsel)

    metric = ahmc_metric(proposal.metric, params_vec)
    init_hamiltonian = AdvancedHMC.Hamiltonian(metric, f, fg)
    hamiltonian, init_transition = AdvancedHMC.sample_init(rng, init_hamiltonian, params_vec)
    integrator = _ahmc_set_step_size(proposal.integrator, hamiltonian, params_vec, rng)
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

function BAT.mcmc_propose!!(mc_state::HMCState)
    target = mc_state.target
    proposal = mc_state.proposal
    f_transform = mc_state.f_transform
    samples = mc_state.samples
    sample_z = mc_state.sample_z
    context = mc_state.context

    rng = get_rng(context)

    current_z_idx = _current_sample_z_idx(mc_state)
    proposed_z_idx = _proposed_sample_z_idx(mc_state)

    proposed_x_idx = _proposed_sample_idx(mc_state)

    # location in normalized (or generally transformed) space ("z-space")
    z_current = sample_z.v[current_z_idx]
    z_proposed = sample_z.v[proposed_z_idx]

    # location in target space ("x-space") which is generally pre-transformed
    x_proposed = samples.v[proposed_x_idx]
    
    τ = deepcopy(proposal.kernel.τ)
    @reset τ.integrator = AdvancedHMC.jitter(rng, τ.integrator)

    hamiltonian = proposal.hamiltonian

    @static if isdefined(AdvancedHMC, :rand_momentum) #isdefined(AdvancedHMC.rand_momentum, Tuple{AbstractRNG, AdvancedHMC.AbstractMetric, AdvancedHMC.AbstractKinetic, AbstractVecOrMat})
        # For AdvnacedHMC.jl v >= 0.7 
        momentum = rand_momentum(rng, hamiltonian.metric, hamiltonian.kinetic, z_current[:])
    else
        momentum = rand(rng, hamiltonian.metric, hamiltonian.kinetic)
    end
    
    z_phase = AdvancedHMC.phasepoint(hamiltonian, vec(z_current[:]), momentum)
    # Note: `RiemannianKinetic` requires an additional position argument, but including this causes issues. So only support the other kinetics.

    proposal.transition = AdvancedHMC.transition(rng, τ, hamiltonian, z_phase)
    p_accept = AdvancedHMC.stat(proposal.transition).acceptance_rate

    z_proposed[:] =  proposal.transition.z.θ
    accepted = z_current[:] != z_proposed[:]

    p_accept = AdvancedHMC.stat(proposal.transition).acceptance_rate

    x_proposed[:], ladj = with_logabsdet_jacobian(f_transform, z_proposed)    
    logd_x_proposed = logdensityof(target, x_proposed)
    samples.logd[proposed_x_idx] = logd_x_proposed

    sample_z.logd[proposed_z_idx] = logd_x_proposed + ladj

    return mc_state, accepted, p_accept
end

function BAT._accept_reject!(mc_state::HMCState, accepted::Bool, p_accept::Float64, current::Integer, proposed::Integer)
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

function BAT.set_mc_state_transform!!(mc_state::HMCState, f_transform_new::Function) 
    adsel = get_adselector(mc_state.context)
    f = checked_logdensityof(pullback(f_transform_new, mc_state.target))
    fg = valgrad_func(f, adsel)
    
    h = mc_state.proposal.hamiltonian 

    h = @set h.ℓπ = f
    h = @set h.∂ℓπ∂θ = fg 

    mc_state_new = @set mc_state.proposal.hamiltonian = h

    mc_state_new = @set mc_state_new.f_transform = f_transform_new
    return mc_state_new
end
