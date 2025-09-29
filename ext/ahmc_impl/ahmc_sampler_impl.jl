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
    v_init::AbstractVector{PV},
    f_transform::Function,
    rng::AbstractRNG
) where {P<:Real, PV<:AbstractVector{P}}
    vs = varshape(target)
    npar = totalndof(vs)

    params_vec = Vector{P}(undef, npar)
    params_vec .= v_init[1]# TODO, MD: How to handle proposals for multiple walkers?

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

function BAT.mcmc_propose!!(chain_state::MCMCChainState, proposal::HMCProposalState)
    (; target, f_transform, current, proposed, context) = chain_state
    n_walkers = nwalkers(chain_state)
    rng = get_rng(context)

    τ = deepcopy(proposal.kernel.τ)
    @reset τ.integrator = AdvancedHMC.jitter(rng, τ.integrator)

    hamiltonian = proposal.hamiltonian

    @static if pkgversion(AdvancedHMC) >= v"0.07"
	    foo = [1, 1] # For some reason AdvancedHMC.rand_momentum wants an AbstractVecOrMat in the last argument, but its unused...?
        momentum = AdvancedHMC.rand_momentum(rng, hamiltonian.metric, hamiltonian.kinetic, foo)
    else
        momentum = rand(rng, hamiltonian.metric, hamiltonian.kinetic)
    end

    p_accept = Vector{Float64}(undef, n_walkers)

    # TODO, MD: How should HMC handle multiple walkers?
    for i in 1:n_walkers
        z_phase = AdvancedHMC.phasepoint(hamiltonian, current.z.v[i][:], momentum)

        # TODO: MD, Make properly in !! style. Still overwrites the proposal
        @static if pkgversion(AdvancedHMC) >= v"0.8" 
            # proposal = @set proposal.transition = AdvancedHMC.transition(rng, hamiltonian, τ, z_phase)
            proposal.transition = AdvancedHMC.transition(rng, hamiltonian, τ, z_phase)
        else
            # proposal = @set proposal.transition = AdvancedHMC.transition(rng, τ, hamiltonian, z_phase)
            proposal.transition = AdvancedHMC.transition(rng, τ, hamiltonian, z_phase)
        end

        proposed.z.v[i] = proposal.transition.z.θ
        p_accept[i] = AdvancedHMC.stat(proposal.transition).acceptance_rate
    end

    chain_state.accepted = proposed.z.v .!= current.z.v
    
    trafo_proposed = with_logabsdet_jacobian.(f_transform, proposed.z.v)
    proposed.x.v .= getfield.(trafo_proposed, 1)
    ladj = getfield.(trafo_proposed, 2)

    logd_x_proposed = BAT.checked_logdensityof.(target, proposed.x.v)
    logd_z_proposed = logd_x_proposed .+ ladj

    chain_state.proposed.x.logd .= logd_x_proposed
    chain_state.proposed.z.logd .= logd_z_proposed

    return chain_state, p_accept
end

BAT.eff_acceptance_ratio_impl(chain_state::MCMCChainState, proposal::HMCProposalState) = nsamples(chain_state) / (nsteps(chain_state) * nwalkers(chain_state))

function BAT.set_proposal_transform!!(proposal::HMCProposalState, chain_state::MCMCChainState) 
    f_transform_new = chain_state.f_transform
    adsel = get_adselector(chain_state.context)
    f = checked_logdensityof(pullback(f_transform_new, chain_state.target))
    fg = valgrad_func(f, adsel)
    
    h = proposal.hamiltonian 

    h = @set h.ℓπ = f
    h = @set h.∂ℓπ∂θ = fg

    proposal_new = @set proposal.hamiltonian = h

    return proposal_new
end

