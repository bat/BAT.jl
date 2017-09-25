# This file is a part of BAT.jl, licensed under the MIT License (MIT).


pdist = GenericProposalDist(MvTDist(1.0, PDMat([2.2 0.0; 0.0 2.2])))


mutable struct ProposalCovTuner{T}
    iteration::Int # initially 1
    lambda::T # e.g. 0.5
    scale::T # initially 2.38^2/m
end

export ProposalCovTuner


function ProposalCovTuner(chain::MCMCChain, lambda::Real = 0.5)
    m = nparams(chain)
    iteration = 0
    scale = 2.38^2 / m
    ProposalCovTuner(iteration, lambda, scale)
end


function tuning_init!(chain::MCMCChain{<:MetropolisHastings}, tuner::ProposalCovTuner)
    chain.info.cycle != 0 && error("MCMC chain tuning already initialized")

    state = chain.state

    # ToDo: Generalize for non-hypercube bounds
    bounds = chain.target.bounds
    flat_var = (bounds.hi - bounds.lo).^2 / 12

    m = length(flat_var)
    Σ_unscaled = full(PDiagMat(flat_var))
    Σ = Σ_unscaled * tuner.scale
    state.pdist = set_cov!(state.pdist, Σ)

    tuner.iteration = 1
    next_cycle!(chain)
end


function tuning_step!(chain::MCMCChain{<:MetropolisHastings}, tuner::ProposalCovTuner, stats::MCMCChainStats)
    chain.info.cycle == 0 && error("MCMC chain tuning not initialized")

    α_min = 0.15
    α_max = 0.35

    c_min = 1e-4
    c_max = 1e2

    β = 1.5

    state = chain.state

    t = tuner.iteration
    λ = tuner.lambda
    c = tuner.scale
    Σ_old = full(get_cov(state.pdist))

    S = convert(Array, stats.param_stats.cov)
    a_t = 1 / t^λ
    new_Σ_unscal = (1 - a_t) * (Σ_old/c) + a_t * S

    α = acceptance_ratio(state)

    new_c = if α > α_max && c < c_max
        convert(typeof(c), c * β)
    elseif α < α_min && c > c_min
        convert(typeof(c), c / β)
    else
        c
    end

    tuner.scale = new_c

    Σ_new = full(Hermitian(new_Σ_unscal * tuner.scale))
    state.pdist = set_cov!(state.pdist, Σ_new)

    tuner.iteration += 1
    next_cycle!(chain)
end
