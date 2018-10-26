# This file is a part of BAT.jl, licensed under the MIT License (MIT).

mutable struct DirectSamplingState{
    Q<:AbstractProposalDist,
    S<:DensitySample
} <: AbstractMCMCState
    pdist::Q
    current_sample::S
    proposed_sample::S
    proposal_accepted::Bool
    current_nreject::Int64
    nsamples::Int64
    nsteps::Int64
end

function DirectSamplingState(
    pdist::AbstractProposalDist,
    current_sample::DensitySample
)
    proposed_sample = DensitySample(
        similar(current_sample.params),
        convert(typeof(current_sample.log_value), NaN),
        zero(current_sample.weight)
    )

    DirectSamplingState(
        pdist,
        current_sample,
        proposed_sample,
        false,
        zero(Int64),
        zero(Int64),
        one(Int64)
    )
end


nparams(state::DirectSamplingState) = nparams(state.pdist)

nsteps(state::DirectSamplingState) = state.nsteps

nsamples(state::DirectSamplingState) = state.nsamples

eff_acceptance_ratio(state::DirectSamplingState) = nsamples(state) / nsteps(state)


function next_cycle!(state::DirectSamplingState)
    state.current_sample.weight = one(state.current_sample.weight)
    state.nsamples = zero(Int64)
    state.nsteps = zero(Int64)
    state
end


density_sample_type(state::DirectSamplingState{Q,S}) where {Q,S} = S


_accrej_current_sample(state::DirectSamplingState) =
    ifelse(state.proposal_accepted, state.current_sample, state.proposed_sample)


function nsamples_available(chain::MCMCIterator{<:MCMCAlgorithm{DirectSamplingState}}, nonzero_weights::Bool = false)
    if nonzero_weights
        (_accrej_current_sample(chain.state).weight > 0) ? 1 : 0
    else
        1
    end
end


function get_samples!(appendable, chain::MCMCIterator{<:MCMCAlgorithm{DirectSamplingState}}, nonzero_weights::Bool)::typeof(appendable)
    sample = _accrej_current_sample(chain.state)
    if (!nonzero_weights || sample.weight > 0)
        push!(appendable, sample)
    end
    appendable
end


function get_sample_ids!(appendable, chain::MCMCIterator{<:MCMCAlgorithm{DirectSamplingState}}, nonzero_weights::Bool)::typeof(appendable)
    state = chain.state
    sample = _accrej_current_sample(state)
    if (!nonzero_weights || sample.weight > 0)
        sampletype = state.proposal_accepted ? 2 : 1
        sampleid = MCMCSampleID(chain.id, chain.cycle, state.nsteps, sampletype)
        push!(appendable, sampleid)
    end
    appendable
end


function MCMCIterator(
    algorithm::MCMCAlgorithm{DirectSamplingState},
    likelihood::AbstractDensity,
    prior::AbstractDensity,
    id::Int64,
    rng::AbstractRNG,
    initial_params::AbstractVector{P} = Vector{P}(),
    exec_context::ExecContext = ExecContext(),
) where {P<:Real}
    target = likelihood * prior

    cycle = zero(Int)
    reset_rng_counters!(rng, id, cycle, 0)

    params_vec = Vector{P}(undef, nparams(target))
    if isempty(initial_params)
        rand_initial_params!(rng, algorithm, prior, params_vec)
    else
        params_vec .= initial_params
    end

    !(params_vec in param_bounds(target)) && throw(ArgumentError("Parameter(s) out of bounds"))

    m = length(params_vec)

    log_value = density_logval(target, params_vec, exec_context)
    L = typeof(log_value)
    W = sample_weight_type(typeof(algorithm))

    current_sample = DensitySample(
        params_vec,
        log_value,
        one(W)
    )

    proposed_sample = DensitySample(
        similar(current_sample.params),
        convert(typeof(current_sample.log_value), NaN),
        zero(W)
    )

    state = DirectSamplingState(
        algorithm,
        target,
        current_sample
    )

    chain = MCMCIterator(
        algorithm,
        target,
        state,
        rng,
        id,
        cycle,
        false,
        false
    )

    chain
end


exec_capabilities(mcmc_step!, callback::AbstractMCMCCallback, chain::MCMCIterator{<:MCMCAlgorithm{DirectSamplingState}}) =
    exec_capabilities(density_logval, chain.target, chain.state.proposed_sample.params)


function mcmc_step!(
    callback::AbstractMCMCCallback,
    chain::MCMCIterator{<:MCMCAlgorithm{DirectSamplingState}},
    exec_context::ExecContext,
    ll::LogLevel
)
    state = chain.state
    algorithm = chain.algorithm

    if !mcmc_compatible(algorithm, chain.state.pdist, param_bounds(chain.target))
        error("Implementation of algorithm $algorithm does not support current parameter bounds with current proposal distribution")
    end

    current_sample = state.current_sample
    proposed_sample = state.proposed_sample

    state.nsteps += 1
    reset_rng_counters!(chain)

    mcmc_propose_accept_reject!(callback, chain, exec_context)

    if state.proposal_accepted
        copyto!(current_sample, proposed_sample)
        state.current_nreject = 0
        state.proposal_accepted = false
    end

    chain
end


struct DirectSampling <: MCMCAlgorithm{DirectSamplingState} end
export DirectSampling


mcmc_compatible(::DirectSampling, ::AbstractProposalDist, ::AbstractParamBounds) = true


# ToDo: Specialized version of rand_initial_params for DirectSampling:
#
#     rand_initial_params!(rng::AbstractRNG, algorithm::DirectSampling, target::DensityFunction, x::StridedVecOrMat{<:Real}) = ...


AbstractMCMCTunerConfig(algorithm::DirectSampling) = NoOpTunerConfig()


sample_weight_type(::Type{DirectSampling}) = Int


function DirectSamplingState(
    algorithm::DirectSampling,
    target::BAT.DensityProduct{2,<:Tuple{MvDistDensity, ConstDensity}},
    current_sample::DensitySample{P,T,W}
) where {P,T,W}
    DirectSamplingState(
        GenericProposalDist(parent(target)[1].d),
        current_sample
    )
end


function mcmc_propose_accept_reject!(
    callback::AbstractMCMCCallback,
    chain::MCMCIterator{<:DirectSampling},
    exec_context::ExecContext
)
    state = chain.state
    target = chain.target

    proposed_sample = state.proposed_sample
    proposed_params = proposed_sample.params

    # Propose new parameters:
    rand!(chain.rng, state.pdist.s, proposed_params)

    # Accept iff in bounds:
    if proposed_params in param_bounds(target)
        proposed_sample.log_value = density_logval(target, proposed_params, exec_context)

        state.proposed_sample.weight = 1
        @assert state.current_sample.weight == 1
        state.proposal_accepted = true
        state.nsamples += 1
    else
        proposed_sample.log_value = -Inf

        state.current_nreject += 1
    end
    callback(1, chain)

    chain
end

