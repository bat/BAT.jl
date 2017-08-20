# This file is a part of BAT.jl, licensed under the MIT License (MIT).



#=

# Define a scheduler.

# Use FunctionWrappers, e.g. FunctionWrapper{Float64,Tuple{Float64, ...}}(f)?


mutable struct MHChainState{
    T<:Real,
    P<:Real,
    F<:AbstractTargetFunction,
    Q<:ProposalDist,
    B<:AbstractParamBounds,
    RNG<:AbstractRNG
}
    f::F,
    q::Q,
    bounds::B,
    log_value::T,
    params::Vector{P},
    multiplicity::Int
end



function propose_and_eval!(
    params_new::Vector{P},
    f::AbstractTargetFunction,
    q::ProposalDist,
    params_old::Vector{P},
    bounds::AbstractParamBounds,
    scheduler::ExecScheduler
)
    proposal_rand!(rng, state.q, params_new, params_old)
    apply_bounds!(par_new, bounds)
    new_log_value = state.log_f(state.λ_tmp)::typeof(state.p)
end



mh_chain_step(state::AbstractVector{MHChainState}, scheduler::ExecScheduler) = begin
    rng = state.rng
    params_old = state.params
    params_new = similar(params) # TODO: Avoid memory allocation

    log_value_new = propose_and_eval!(
        params_new,
        state.f,
        state.q,
        params_old,
        state.bounds,
        scheduler
    )

    proposal_rand!(rng, state.q, new_params, params)
    apply_bounds!(par_new, bounds)
    new_log_value = state.log_f(state.λ_tmp)::typeof(state.p)

    if isnan(new_log_value) error("Encountered NaN value for target function")
    accept = log(rand(state.rng)) < new_log_value - state.log_value
    if accept
        copy!(state.λ, state.λ_tmp)
        state.log_value = new_log_value
    end
    accept
end


abstract MCSampler

mutable struct MHSampler{F} <: AbstractMCSampler
    log_f::F
    ...
end


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
