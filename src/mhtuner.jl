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
    tuner = ProposalCovTuner(iteration, lambda, scale)
    tuner
end


function tuning_init!(chain::MCMCChain{<:MetropolisHastings}, tuner::ProposalCovTuner)
    chain.info.cycle != 0 && error("MCMC chain tuning already initialized")

    state = chain.state

    # ToDo: Generalize for non-hypercube bounds
    bounds = chain.target.bounds
    flat_var = (bounds.vol.hi - bounds.vol.lo).^2 / 12

    m = length(flat_var)
    Σ_unscaled = full(PDiagMat(flat_var))
    c = tuner.scale / m
    Σ = Σ_unscaled * c

    next_cycle!(chain)
    state.pdist = set_cov!(state.pdist, Σ)
    tuner.iteration = 1

    chain
end


function tuning_step!(chain::MCMCChain{<:MetropolisHastings}, tuner::ProposalCovTuner, stats::MCMCBasicStats)
    chain.info.cycle == 0 && error("MCMC chain tuning not initialized")

    α_min = 0.15
    α_max = 0.35

    c_min = 1e-4
    c_max = 1e2

    β = 1.5

    state = chain.state

    t = tuner.iteration
    λ = tuner.lambda
    c = tuner.scale / m
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

    if new_c != c
        tuner.scale = new_c * m
    end

    Σ_new = full(Hermitian(new_Σ_unscal * tuner.scale))

    next_cycle!(chain)
    state.pdist = set_cov!(state.pdist, Σ_new)
    tuner.iteration += 1

    chain
end


# function mcmc_auto_tune!(
#     callback,
#     chains::MCMCChain{<:MetropolisHastings},
#     exec_context::ExecContext = ExecContext(),
#     chains_stats::AbstractVector{<:MCMCBasicStats};
#     max_nsamples_per_cycle::Int64 = Int64(1),
#     max_nsteps_per_cycle::Int = 10000,
#     max_nsamples_per_cycle::Int64 = 1000,
#     max_ncycles::Int = 30,
#     max_time::Float64 = Inf,
#     granularity::Int = 1
# )
