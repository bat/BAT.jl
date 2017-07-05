# This file is a part of BAT.jl, licensed under the MIT License (MIT).



#=

# Define a scheduler.

# Use FunctionWrappers, e.g. FunctionWrapper{Float64,Tuple{Float64, ...}}(f)?


mutable struct MetropolisChainState{
    T<:Real,
    P<:Real,
    F<:TargetFunction,
    Q<:AbstractProposalFunction,
    QS<:Sampleable{Multivariate}
    RNG<:AbstractRNG,
}
    # TODO: Unicode symbols? Group log_f and param_bounds?
    target::F,
    q::Q,
    q_sampler::QS,
    rng::RNG,
    params::Vector{P},
    log_value::T,
    new_params::Vector{P}
    # stats::MCChainStats
end


metropolis_step(state::MetropolisChainState) = begin
    rand!(state.rng, state.q, state.new_params)
    state.new_params .+= state.params

    accept = false
    if new_params in state.param_bounds
        new_log_value = state.log_f(state.λ_tmp)::typeof(state.p)
        if isnan(new_log_value) error("Encountered NaN value for target function")
        accept = log(rand(state.rng)) < new_log_value - state.log_value
        if accept
            copy!(state.λ, state.λ_tmp)
            state.log_value = new_log_value
        end
    end
    accept
end



abstract MCSampler

mutable struct MetropolisSampler{F} <: AbstractMCSampler
    log_f::F
    ...
end


function MetropolisSampler(
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

sampler = MetropolisSampler(x -> -x^2/2, [(-4, 4)], n_chains = 4)
output = rand(sampler, 1000000) = ...::SamplerOutput

=#
