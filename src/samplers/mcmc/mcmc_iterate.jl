# This file is a part of BAT.jl, licensed under the MIT License (MIT).
# TODO Rename to "MCMCState"
struct TransformedMCMCIterator{
    PR<:RNGPartition,
    M<:BATMeasure,
    F,
    Q<:TransformedMCMCProposal,
    SV<:DensitySampleVector,
    CTX<:BATContext,
} <: MCMCIterator
    rngpart_cycle::PR
    μ::M
    f_transform::F
    proposal::Q
    samples::SV #  Copy from old BAT 
    sample_z::SV
    stepno::Int
    n_accepted::Int
    info::MCMCIteratorInfo
    context::CTX
end

# TODO Copy handling of samples from old bat 
mutable struct MCMCState{
    M<:BATMeasure,
    PR<:RNGPartition,
    FT<:Function,
    TP<:TransformedMCMCProposal,
    Q<:Distribution{Multivariate,Continuous},
    S<:DensitySample,
    SV<:DensitySampleVector{S},
    CTX<:BATContext
} <: MCMCIterator
    target::M
    f_transform::FT
    rngpart_cycle::PR
    info::MCMCIteratorInfo
    proposal::TP
    samples::SV
    sample_z::S
    nsamples::Int64
    stepno::Int64
    context::CTX
end


export TransformedMCMCIterator

@inline _current_sample_idx(chain::TransformedMCMCIterator) = firstindex(chain.samples)
@inline _proposed_sample_idx(chain::TransformedMCMCIterator) = lastindex(chain.samples)

getmeasure(chain::TransformedMCMCIterator) = chain.μ

get_context(chain::TransformedMCMCIterator) = chain.context

mcmc_info(chain::TransformedMCMCIterator) = chain.info 

mcmc_target(chain::TransformedMCMCIterator) = chain.μ

nsteps(chain::TransformedMCMCIterator) = chain.stepno

nsamples(chain::TransformedMCMCIterator) = chain.n_accepted

current_sample(chain::TransformedMCMCIterator) = chain.samples[_current_sample_idx(chain)]

sample_type(chain::TransformedMCMCIterator) = eltype(chain.samples)

samples_available(chain::TransformedMCMCIterator) = size(chain.samples,1) > 0

isvalidchain(chain::TransformedMCMCIterator) = current_sample(chain).logd > -Inf

isviablechain(chain::TransformedMCMCIterator) = nsamples(chain) >= 2

eff_acceptance_ratio(chain::TransformedMCMCIterator) = nsamples(chain) / chain.stepno


# TODO: MD remove
isvalidchain(chain::MCMCIterator) = current_sample(chain).logd > -Inf
isviablechain(chain::MCMCIterator) = nsamples(chain) >= 2


#ctor
function TransformedMCMCIterator(
    proposal::Union{TransformedMCMCSampling, MCMCAlgorithm}, # TODO: Resolve type
    target,
    id::Integer,
    v_init::AbstractVector{<:Real},
    context::BATContext
) 
    TransformedMCMCIterator(proposal, target, Int32(id), v_init, context)
end


#ctor
function TransformedMCMCIterator(
    algorithm::Union{TransformedMCMCSampling, MCMCAlgorithm}, # TODO: Resolve type
    target,
    id::Int32,
    v_init::AbstractVector{P},
    context::BATContext,
) where {P<:Real}
    rngpart_cycle = RNGPartition(get_rng(context), 0:(typemax(Int16) - 2))

    μ = target
    n_dims = getdof(μ)
    proposal = _get_proposal(algorithm, target, context, v_init) # TODO: MD Resolve handling of algorithms as proposals 
    stepno = 0
    cycle = 1
    n_accepted = 0

    adaptive_transform_spec = _get_adaptive_transform(algorithm) # TODO: MD Resolve
    g = init_adaptive_transform(adaptive_transform_spec, μ, context)

    logd_x = logdensityof(μ, v_init)
    inverse_g = inverse(g)
    z = inverse_g(v_init)
    logd_z = logdensityof(MeasureBase.pullback(g, μ),z)

    W = Int # TODO: MD: Resolve weighting schemes in transformed MCMC
    T = typeof(logd_x)

    info = MCMCSampleID(id, one(Int32), 0, CURRENT_SAMPLE)
    sample_x = DensitySample(v_init, logd_x, 1, info, nothing)
    
    samples = DensitySampleVector{Vector{P}, T, W, MCMCSampleID, Nothing}(undef, 0, n_dims)
    push!(samples, sample_x)

    sample_z = DensitySampleVector{Vector{P}, T, W, MCMCSampleID, Nothing}(undef, 0, n_dims)
    push!(sample_z, DensitySample(z, logd_z, 1, MCMCSampleID(id, one(Int32), 0, CURRENT_SAMPLE), nothing)) # TODO: MD: More elegant solution?
    push!(sample_z, DensitySample(z, logd_z, 1, MCMCSampleID(id, one(Int32), 0, PROPOSED_SAMPLE), nothing))

    TransformedMCMCIterator(
        rngpart_cycle,
        target,
        g,
        proposal,
        samples,
        sample_z,
        stepno,
        n_accepted,
        MCMCIteratorInfo(id, cycle, false, false),
        context
    )
end


function propose_mcmc!(
    iter::TransformedMCMCIterator{<:Any, <:Any, <:Any, <:TransformedMHProposal}
    )
    @unpack μ, f_transform, proposal, samples, sample_z, stepno, context = iter
    rng = get_rng(context)

    proposed_x = _proposed_sample_idx(iter)
    current_z = 1
    proposed_z = 2

    z_current = sample_z.v[current_z]

    n = size(z_current, 1)
    sample_z.v[proposed_z] = z_current + rand(rng, proposal.proposal_dist, n) #TODO: check if proposal is symmetric? otherwise need additional factor?
    samples.v[proposed_x], ladj = with_logabsdet_jacobian(f_transform, sample_z.v[proposed_z])
    samples.logd[proposed_x] = BAT.checked_logdensityof(μ, samples.v[proposed_x])
    sample_z.logd[proposed_z] = samples.logd[proposed_x] + ladj
    @assert sample_z.logd[proposed_z] ≈ logdensityof(MeasureBase.pullback(f_transform, μ), sample_z.v[proposed_z]) #TODO: remove

    
    # TODO AC: do we need to check symmetry of proposal distribution?
    # T = typeof(logd_z)
    # p_accept = if logd_z_proposed > -Inf
    #     # log of ratio of forward/reverse transition probability
    #     log_tpr = if issymmetric(proposal.proposal_dist)
    #         T(0)
    #     else
    #         log_tp_fwd = proposaldist_logpdf(proposaldist, proposed_params, current_params)
    #         log_tp_rev = proposaldist_logpdf(proposaldist, current_params, proposed_params)
    #         T(log_tp_fwd - log_tp_rev)
    #     end

    #     p_accept_unclamped = exp(proposed_log_posterior - current_log_posterior - log_tpr)
    #     T(clamp(p_accept_unclamped, 0, 1))
    # else
    #     zero(T)
    # end

    p_accept = clamp(exp(sample_z.logd[proposed_z] - sample_z.logd[current_z]), 0, 1)

    return p_accept
end



function transformed_mcmc_step!!(
    iter::TransformedMCMCIterator,
    tuner::AbstractMCMCTunerInstance,
    tempering::TransformedMCMCTemperingInstance,
)
    _cleanup_samples(iter)     # TODO: MD should this stay?
    reset_rng_counters!(iter) # TODO: MD should this stay? 
    @unpack  μ, f_transform, proposal, samples, sample_z, stepno, context = iter
    rng = get_rng(context)

    # Grow samples vector by one:
    resize!(samples, size(samples, 1) + 1)
    samples.info[lastindex(samples)] = MCMCSampleID(iter.info.id, iter.info.cycle, iter.stepno, PROPOSED_SAMPLE)
    current_x = _current_sample_idx(iter)
    proposed_x = _proposed_sample_idx(iter)
    @assert current_x != proposed_x

    samples.weight[proposed_x] = 0

    p_accept = propose_mcmc!(iter)

    tuner_new, f_transform, transform_tuned = tune_mcmc_transform!!(tuner, f_transform, p_accept, sample_z, stepno, context)
    
    accepted = rand(rng) <= p_accept

    # f_transform may have changed
    if transform_tuned 
        _update_iter_transform!(iter, f_transform)
    end

    if accepted
        samples.info.sampletype[current_x] = ACCEPTED_SAMPLE
        samples.info.sampletype[proposed_x] = CURRENT_SAMPLE
        iter.n_accepted += 1 # TODO MD behaviour or n_accepted vs nsamples? 
    else
        samples.info.sampletype[proposed_x] = REJECTED_SAMPLE
    end
    
    delta_w_current, w_proposed = _mcmc_weights(proposal.weighting, p_accept, accepted)
    samples.weight[current_x] += delta_w_current
    samples.weight[proposed_x] = w_proposed

    tempering_new, μ_new = temper_mcmc_target!!(tempering, μ, stepno)

    iter.stepno += 1
    iter.μ = μ_new
    @assert iter.context === context # TODO MD Remove? 
    return (iter, tuner_new, tempering_new)
end


# Copy old version from BAT 
function transformed_mcmc_iterate!(
    mc_state::MCMCState,
    tuner::MCMCTuningState,
    tempering::MCMCTemperingState;
    max_nsteps::Integer = 1,
    max_time::Real = Inf,
    nonzero_weights::Bool = true,
    callback::Function = nop_func,
)
    @debug "Starting iteration over MCMC chain $(mcmc_info(mc_state).id) with $max_nsteps steps in max. $(@sprintf "%.1f seconds." max_time)"

    start_time = time()
    last_progress_message_time = start_time
    start_nsteps = nsteps(mc_state)
    start_nsteps = nsteps(mc_state)

    while (
        (nsteps(mc_state) - start_nsteps) < max_nsteps &&
        (time() - start_time) < max_time
    )
        transformed_mcmc_step!!(mc_state, tuner, tempering)
        callback(Val(:mcmc_step), mc_state)
  
        #TODO: output schemes

        current_time = time()
        elapsed_time = current_time - start_time
        logging_interval = 5 * round(log2(elapsed_time/60 + 1) + 1)
        if current_time - last_progress_message_time > logging_interval
            last_progress_message_time = current_time
            @debug "Iterating over MCMC chain $(mcmc_info(mc_state).id), completed $(nsteps(mc_state) - start_nsteps) (of $(max_nsteps)) steps and produced $(nsteps(mc_state) - start_nsteps) samples in $(@sprintf "%.1f s" elapsed_time) so far."
        end
    end

    current_time = time()
    elapsed_time = current_time - start_time
    @debug "Finished iteration over MCMC chain $(mcmc_info(mc_state).id), completed $(nsteps(mc_state) - start_nsteps) steps and produced $(nsteps(mc_state) - start_nsteps) samples in $(@sprintf "%.1f s" elapsed_time)."
    
    return nothing
end


function transformed_mcmc_iterate!(
    chain::MCMCIterator,
    tuner::AbstractMCMCTunerInstance,
    tempering::TransformedMCMCTemperingInstance;
    # tuner::AbstractMCMCTunerInstance;
    max_nsteps::Integer = 1,
    max_time::Real = Inf,
    nonzero_weights::Bool = true,
    callback::Function = nop_func
)
    cb = callback# combine_callbacks(tuning_callback(tuner), callback) #TODO CA: tuning_callback
    
    transformed_mcmc_iterate!(
        chain, tuner, tempering,
        max_nsteps = max_nsteps, max_time = max_time, nonzero_weights = nonzero_weights, callback = cb
    )

    return nothing
end


function transformed_mcmc_iterate!(
    chains::AbstractVector{<:MCMCIterator},
    tuners::AbstractVector{<:AbstractMCMCTunerInstance},
    temperers::AbstractVector{<:TransformedMCMCTemperingInstance};
    kwargs...
)
    if isempty(chains)
        @debug "No MCMC chain(s) to iterate over."
        return chains
    else
        @debug "Starting iteration over $(length(chains)) MCMC chain(s)"
    end

    @sync for i in eachindex(chains, tuners, temperers)
        Base.Threads.@spawn transformed_mcmc_iterate!(chains[i], tuners[i], temperers[i]#= , tnrs[i] =#; kwargs...)
    end

    return nothing
end


# TODO: MD Remove, Used during transformed transition
function mcmc_iterate!(
    output::DensitySampleVector,
    chain::TransformedMCMCIterator,
    tuner::AbstractMCMCTunerInstance = MCMCNoOpTuner(); # TODO: MD What tuner to use?
    max_nsteps::Integer = 1,
    max_time::Real = Inf,
    nonzero_weights::Bool = true,
    callback::Function = nop_func
)
    transformed_mcmc_iterate!(
        chain,
        tuner,
        NoTransformedMCMCTemperingInstance(); #TODO: MD What tempering to use?
        max_nsteps,
        max_time,
        nonzero_weights,
        callback
    )
    push!(output, chain.samples...)
end


# function mcmc_iterate!(
#     output::Union{DensitySampleVector,Nothing},
#     chain::MCMCIterator,
#     tuner::Nothing = nothing;
#     max_nsteps::Integer = 1,
#     max_time::Real = Inf,
#     nonzero_weights::Bool = true,
#     callback::Function = nop_func
# )
#     @debug "Starting iteration over MCMC chain $(chain.info.id) with $max_nsteps steps in max. $(@sprintf "%.1f s" max_time)"

#     start_time = time()
#     last_progress_message_time = start_time
#     start_nsteps = nsteps(chain)
#     start_nsamples = nsamples(chain)

#     while (
#         (nsteps(chain) - start_nsteps) < max_nsteps &&
#         (time() - start_time) < max_time
#     )
#         mcmc_step!(chain)
#         callback(Val(:mcmc_step), chain)
#         if !isnothing(output)
#             get_samples!(output, chain, nonzero_weights)
#         end
#         current_time = time()
#         elapsed_time = current_time - start_time
#         logging_interval = 5 * round(log2(elapsed_time/60 + 1) + 1)
#         if current_time - last_progress_message_time > logging_interval
#             last_progress_message_time = current_time
#             @debug "Iterating over MCMC chain $(chain.info.id), completed $(nsteps(chain) - start_nsteps) (of $(max_nsteps)) steps and produced $(nsamples(chain) - start_nsamples) samples in $(@sprintf "%.1f s" elapsed_time) so far."
#         end
#     end

#     current_time = time()
#     elapsed_time = current_time - start_time
#     @debug "Finished iteration over MCMC chain $(chain.info.id), completed $(nsteps(chain) - start_nsteps) steps and produced $(nsamples(chain) - start_nsamples) samples in $(@sprintf "%.1f s" elapsed_time)."

#     return nothing
# end


function mcmc_iterate!(
    output::Union{DensitySampleVector,Nothing},
    chain::MCMCIterator,
    tuner::AbstractMCMCTunerInstance;
    max_nsteps::Integer = 1,
    max_time::Real = Inf,
    nonzero_weights::Bool = true,
    callback::Function = nop_func
)
    cb = combine_callbacks(tuning_callback(tuner), callback)
    mcmc_iterate!(
        output, chain;
        max_nsteps = max_nsteps, max_time = max_time, nonzero_weights = nonzero_weights, callback = cb
    )

    return nothing
end


# TODO: MD: Remove, temporary
function mcmc_iterate!(
    outputs::AbstractVector{<:DensitySampleVector},
    chains::AbstractVector{<:MCMCIterator};
    kwargs...
)
    mcmc_iterate!(outputs, chains, fill(MCMCNoOpTuner(), length(chains)); kwargs...)
end

function mcmc_iterate!(
    outputs::AbstractVector{<:DensitySampleVector},
    chains::AbstractVector{<:MCMCIterator},
    tuners::AbstractVector{<:AbstractMCMCTunerInstance};
    kwargs...
)
    if isempty(chains)
        @debug "No MCMC chain(s) to iterate over."
        return chains
    else
        @debug "Starting iteration over $(length(chains)) MCMC chain(s)"
    end

    outs = isnothing(outputs) ? fill(nothing, size(chains)...) : outputs
    tnrs = isnothing(tuners) ? fill(nothing, size(chains)...) : tuners

    @sync for i in eachindex(outs, chains, tnrs)
        Base.Threads.@spawn mcmc_iterate!(outs[i], chains[i], tnrs[i]; kwargs...)
    end

    return nothing
end
