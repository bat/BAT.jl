# This file is a part of BAT.jl, licensed under the MIT License (MIT).


mutable struct MHState{
    Q<:AbstractProposalDist,
    R<:AbstractRNG,
    S<:MCMCSample
} <: AbstractMCMCState
    pdist::Q
    rng::R
    current_sample::S
    proposed_sample::S
    proposal_accepted::Bool
    current_nreject::Int64
    nsamples::Int64
    nsteps::Int64
    naccept::Int64
end


function MHState(
    pdist::AbstractProposalDist,
    rng::AbstractRNG,
    current_sample::MCMCSample
)
    proposed_sample = MCMCSample(
        similar(current_sample.params),
        convert(typeof(current_sample.log_value), NaN),
        zero(current_sample.weight)
    )

    MHState(
        pdist,
        rng,
        current_sample,
        proposed_sample,
        false,
        0,
        0,
        1,
        1
    )
end


nparams(state::MHState) = nparams(state.pdist)

function next_cycle!(state::MHState)
    # state.nsamples = 0 # ToDo: Reset number of samples or not?
    state.nsteps = 0
    state.naccept = 0
    state
end

acceptance_ratio(state::MHState) = state.naccept / state.nsteps


function MCMCChainStats(state::MHState)
    s = state.current_sample
    L = promote_type(typeof(s.log_value), Float64)
    P = promote_type(eltype(s.params), Float64)
    m = length(s.params)
    MCMCChainStats{L, P}(m)
end




struct MetropolisHastings <: MCMCAlgorithm{MHState} end
export MetropolisHastings



function MCMCChain(
    algorithm::MetropolisHastings,
    target::AbstractTargetSubject,
    pdist::Union{AbstractProposalDist,ProposalDistSpec},
    id::Integer = 1,
    exec_context::ExecContext = ExecContext(),
    rng::AbstractRNG = create_rng(MersenneTwisterSeed()),
    initial_params::AbstractVector{P} = rand(rng, target.bounds),
    status::MCMChainStatus = UNCONVERGED,
    cycle::Integer = 0
) where {P<:Real}
    params_vec = convert(Vector{P}, initial_params)
    m = length(params_vec)
    apply_bounds!(params_vec, target.bounds)

    log_value = target_logval(target.tdensity, params_vec, exec_context)
    L = typeof(log_value)
    isoob(params_vec) && throw(ArgumentError("Parameter(s) out of bounds"))

    current_sample = MCMCSample(
        params_vec,
        log_value,
        zero(Int)
    )

    proposed_sample = MCMCSample(
        similar(current_sample.params),
        convert(typeof(current_sample.log_value), NaN),
        zero(Int)
    )

    state = MHState(
        convert(AbstractProposalDist, pdist, P, m),
        rng,
        current_sample
    )

    info = MCMCChainInfo(id, cycle, status)

    stats = MCMCChainStats{L, P}(2)

    chain = MCMCChain(
        algorithm,
        target,
        state,
        info
    )

    chain
end


function (::Type{Vector{MCMCChain}})(
    algorithm::MetropolisHastings,
    target::AbstractTargetSubject,
    pdist::Union{AbstractProposalDist,ProposalDistSpec},
    n::Integer,
    exec_context::ExecContext = ExecContext(),
    rngseed::AbstractRNGSeed = MersenneTwisterSeed()
)
    # tasks = [@schedule MCMCChain(algorithm, target, pdist, id, exec_context, rng) for id in 1:n]
    [MCMCChain(deepcopy(algorithm), deepcopy(target), deepcopy(pdist), id, exec_context, create_rng(rngseed)) for id in 1:n]
end


mcmc_compatible(::MetropolisHastings, pdist::AbstractProposalDist, bounds::UnboundedParams) = true

mcmc_compatible(::MetropolisHastings, pdist::AbstractProposalDist, bounds::HyperRectBounds) =
    issymmetric(pdist) || all(x -> x == hard_bounds, bounds.bt)



function mcmc_iterate(
    callback,
    chain::MCMCChain{<:MetropolisHastings},
    exec_context::ExecContext = ExecContext();
    max_nsamples::Int64 = Int64(1),
    max_nsteps::Int = 1000,
    max_time::Float64 = Inf,
    granularity::Int = 1
)
    algorithm = chain.algorithm

    start_time = time()
    nsteps = 0
    nsamples = 0

    target = chain.target
    state = chain.state
    rng = state.rng

    tdensity = target.tdensity
    bounds = target.bounds

    pdist = state.pdist
    current_sample = state.current_sample
    proposed_sample = state.proposed_sample

    T = typeof(current_sample.log_value)

    if !mcmc_compatible(algorithm, pdist, bounds)
        error("Implementation of algorithm $algorithm does not support current parameter bounds with current proposal distribution")
    end

    current_params = current_sample.params
    proposed_params = proposed_sample.params

    while nsamples < max_nsamples && nsteps < max_nsteps && (time() - start_time) < max_time
        if state.proposal_accepted
            copy!(current_sample, proposed_sample)
            state.current_nreject = 0
            state.proposal_accepted = false
        end

        current_log_value = current_sample.log_value

        # Propose new parameters:
        # TODO: First mofify/tag counter(s) for counter-based RNG
        proposal_rand!(rng, pdist, proposed_params, current_params)
        apply_bounds!(proposed_params, bounds)

        # log of ratio of forward/reverse transition probability
        log_tpr = if issymmetric(pdist)
            T(0)
        else
            log_tp_fwd = proposal_logpdf(pdist, proposed_params, current_params)
            log_tp_rev = proposal_logpdf(pdist, current_params, proposed_params)
            T(log_tp_fwd - log_tp_rev)
        end

        # Evaluate target density at new parameters:
        proposed_log_value = if !isoob(proposed_params)
            T(target_logval(tdensity, proposed_params, exec_context))
        else
            T(-Inf)
        end

        proposed_sample.log_value = proposed_log_value

        # Metropolis-Hastings accept/reject:
        # TODO: First mofify/tag counter(s) for counter-based RNG before
        accepted = rand(rng) < exp(proposed_log_value - current_log_value - log_tpr)

        nsteps += 1
        state.nsteps += 1

        if accepted
            current_sample.weight = state.current_nreject + 1
            state.proposal_accepted = true
            state.naccept += 1
            nsamples += 1
            state.nsamples += 1
        else
            state.current_nreject += 1
        end

        if accepted || (granularity >= 2)
            callback(chain)
        end
    end
end

export mcmc_iterate




#=



function MHSampler(
    log_f::Any, # target density, log_f(params::AbstractVector, aux_values::)
    param_bounds::Union{AbstractParamBounds, Vector{NTuple{2}}},
    q::AbstractProposalFunction = MvNormal(...),     # proposal distribution
    tune_q::Any # tune_q(q, history::MCSamplerOutput) -> q', tune_q may mutate it's state
    callback::Any # sampling loop callback: callback(state)
    ;
    n_chains::Integer = 1,
    max_iterations::Nullable{Int} = Nullable{Int}(),
    max_runtime::Nullable{Float64} = Nullable{Float64}()
)
    
    bounds, transformed_log_f = _param_bounds(param_bounds)
end


abstract AbstractMCSamplerOutput

# Single chain output (same type after merge?):
mutable struct MCSamplerOutput{T,Arr<:AbstractArray} <: AbstractMCSamplerOutput
    log_f::Arr{T,1} # Target function may be factorized
    weight::Arr{T,1}
    params::Arr{T, 2}
    aux::Arr{T, 2} # Auxiliary values like likelihood, prior, observables, etc.
end


mutable struct SigmaDistTuner{T}
    iteration::Int # initially 1
    lambda::T # e.g. 0.5
    scale::T # initially 2.38^2/ndims
end

function tuning_init(::Type{StudentTProposalFunction}, tuner::SigmaDistTuner, bounds::HyperRectBounds)
    flat_var = (bounds.to - bounds.from).^2 / 12
    ndims = length(flat_var)
    new_Σ_unscal_pd = PDiagMat(flat_var)
    tuner.scale = 2.38^2/ndims
    StudentTProposalFunction(new_Σ_unscal_pd * tuner.scale)
end

function tuning_adapt(tuner::SigmaDistTuner, q::StudentTProposalFunction, history::MCSamplerOutput)
    t = tuner.iteration
    λ = tuner.lambda
    c = tuner.scale
    Σ = q.Σ

    S = cov(history.params, 1)
    a_t = 1/t^λ
    new_Σ_unscal = (1 - a_t) * (Σ/c) + a_t * S
    new_Σ_unscal_pd = PDMat(cholfact(Hermitian(new_Σ_unscal_pd)))

    α_min = 0.15
    α_max = 0.35

    c_min = 1e-4
    c_max = 1e2

    β = 1.5

    α = 1 / mean(history.weight) # acceptance

    if α > α_max && c < c_max
        new_c = c * β
    elseif α < α_min && c > c_min
        new_c /=  c / β
    else
        new_c = c
    end

    tuner.iteration += 1
    tuner.scale = new_c

    StudentTProposalFunction(new_Σ_unscal_pd * tuner.scale)
end


# User:

sampler = MHSampler(x -> -x^2/2, [(-4, 4)], n_chains = 4)
output = rand(sampler, 1000000) = ...::SamplerOutput

=#
