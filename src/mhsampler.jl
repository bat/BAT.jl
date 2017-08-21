# This file is a part of BAT.jl, licensed under the MIT License (MIT).



function proposal_rand_target_logval!(
    params_new::AbstractVector{P},
    target::AbstractTargetFunction,
    pdist::ProposalDist,
    params_old::AbstractVector{P},
    bounds::AbstractParamBounds,
    executor::SerialExecutor
)::Real where {P<:Real}
    proposal_rand!(executor.rng, pdist, params_new, params_old)
    apply_bounds!(params_new, bounds)
    target_logval(target, params_new, executor.ec)
end



mutable struct MHChainState{
    P<:Real,
    R<:Real,
    F<:AbstractTargetFunction,
    Q<:ProposalDist,
    B<:AbstractParamBounds
} <: AbstractMCMCState
    target::F
    pdist::Q
    bounds::B
    #rng:RNG
    params::Vector{P}
    log_value::R
    #exec_context::ExecContext = ExecContext()
    nsamples::Int
    multiplicity::Int

    function MHChainState{P,R,F,Q,B}(
        target::F,
        pdist::Q,
        bounds::B,
        params::Vector{P},
        log_value::R,
        nsamples::Int = 0,
        multiplicity::Int = 0
    ) where {
        P<:Real,
        R<:Real,
        F<:AbstractTargetFunction,
        Q<:ProposalDist,
        B<:AbstractParamBounds
    }
        length(params) != length(bounds) && throw(DimensionMismatch("length(params) != length(bounds)"))
        new{P,R,F,Q,B}(
            target,
            pdist,
            bounds,
            params,
            log_value,
            nsamples,
            multiplicity
        )
    end
end


function MHChainState(
    target::F,
    pdist::Q,
    bounds::B,
    params::Vector{P},
    log_value::R,
    nsamples::Int = 0,
    multiplicity::Int = 0
) where {
    P<:Real,
    R<:Real,
    F<:AbstractTargetFunction,
    Q<:ProposalDist,
    B<:AbstractParamBounds
}
    MHChainState{P,R,F,Q,B}(
        target,
        pdist,
        bounds,
        params,
        log_value,
        nsamples,
        multiplicity
    )
end




function Base.push!(state::MHChainState, params::Vector{<:Real}, log_value::Real, rng::AbstractRNG)
    isnan(log_value) && error("Encountered NaN log_value")
    accepted = log(rand(rng)) < log_value - state.log_value
    if accepted
        copy!(state.params, params)
        state.log_value = log_value
        state.nsamples += 1
        state.multiplicity = 1
    else
        state.multiplicity += 1
    end
    state
end


function mcmc_step(state::MHChainState, rng::AbstractRNG, exec_context::ExecContext = ExecContext())
    params_old = state.params
    params_new = similar(params_old) # TODO: Avoid memory allocation
    proposal_rand!(rng, state.pdist, params_new, params_old)
    apply_bounds!(params_new, state.bounds)
    log_value_new = target_logval(state.target, params_new, exec_context)
    push!(state, params_new, log_value_new, rng)
    state
end


function mcmc_step(states::AbstractVector{MHChainState{P, R}}, rng::AbstractRNG, exec_context::ExecContext = ExecContext()) where {P,R}
    # TODO: Avoid memory allocation:
    params_old = hcat((s.params for s in states)...)
    params_new = similar(params_old)
    log_values_new = Vector{R}(length(states))

    proposal_rand!(rng, state.pdist, params_new, params_old)
    apply_bounds!(params_new, state.bounds)
    log_values_new = target_logval!(log_values_new, state.target, params_new, exec_context)

    for i in eachindex(log_values_new)
        # TODO: Run multithreaded if length(states) is large?
        p_new = view(params_new, :, i) # TODO: Avoid memory allocation
        push!(states[i], p_new, log_values_new[i], rng)
    end
    states
end






#=



function MHSampler(
    log_f::Any, # target function, log_f(params::AbstractVector, aux_values::)
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

function tuning_init(::Type{StudentTProposalFunction}, tuner::SigmaDistTuner, bounds::HyperCubeBounds)
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
