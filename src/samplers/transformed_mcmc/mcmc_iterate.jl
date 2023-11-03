mutable struct TransformedMCMCIterator{
    PR<:RNGPartition,
    D<:BATMeasure,
    F,
    Q<:TransformedMCMCProposal,
    SV<:DensitySampleVector,
    S<:DensitySample,
    CTX<:BATContext,
} <: MCMCIterator
    rngpart_cycle::PR
    μ::D
    f_transform::F
    proposal::Q
    samples::SV
    sample_z::S
    stepno::Int
    n_accepted::Int
    info::TransformedMCMCIteratorInfo
    context::CTX
end

getmeasure(chain::TransformedMCMCIterator) = chain.μ

get_context(chain::TransformedMCMCIterator) = chain.context

mcmc_info(chain::TransformedMCMCIterator) = chain.info 

nsteps(chain::TransformedMCMCIterator) = chain.stepno

nsamples(chain::TransformedMCMCIterator) = size(chain.samples, 1)

current_sample(chain::TransformedMCMCIterator) = last(chain.samples)

sample_type(chain::TransformedMCMCIterator) = eltype(chain.samples)

samples_available(chain::TransformedMCMCIterator) = size(chain.samples,1) > 0

isvalidchain(chain::TransformedMCMCIterator) = current_sample(chain).logd > -Inf

isviablechain(chain::TransformedMCMCIterator) = nsamples(chain) >= 2

eff_acceptance_ratio(chain::TransformedMCMCIterator) = nsamples(chain) / chain.stepno



#ctor
function TransformedMCMCIterator(
    algorithm::TransformedMCMCSampling,
    target,
    id::Integer,
    v_init::AbstractVector{<:Real},
    context::BATContext
) 
     TransformedMCMCIterator(algorithm, target, Int32(id), v_init, context)
end


#ctor
function TransformedMCMCIterator(
    algorithm::TransformedMCMCSampling,
    target,
    id::Int32,
    v_init::AbstractVector{<:Real},
    context::BATContext,
)
    rngpart_cycle = RNGPartition(get_rng(context), 0:(typemax(Int16) - 2))

    μ = target
    proposal = algorithm.proposal
    stepno = 1
    cycle = 1
    n_accepted = 0

    adaptive_transform_spec = algorithm.adaptive_transform
    g = init_adaptive_transform(adaptive_transform_spec, μ, context)

    logd_x = logdensityof(μ, v_init)
    sample_x = DensitySample(v_init, logd_x, 1, TransformedMCMCTransformedSampleID(id, 1, 0), nothing) # TODO
    inverse_g = inverse(g)
    z = inverse_g(v_init) # sample_x.v
    logd_z = logdensityof(MeasureBase.pullback(g, μ),z)
    sample_z = _rebuild_density_sample(sample_x, z, logd_z)

    samples = DensitySampleVector(([sample_x.v], [sample_x.logd], [sample_x.weight], [sample_x.info], [sample_x.aux] ))
    
    iter = TransformedMCMCIterator(
        rngpart_cycle,
        target,
        g,
        proposal,
        samples,
        sample_z,
        stepno,
        n_accepted,
        TransformedMCMCIteratorInfo(id, cycle, false, false),
        context
    )
    

end



function _rebuild_density_sample(s::DensitySample, x, logd, weight=1)
    @unpack info, aux = s
    DensitySample(x, logd, weight, info, aux)
end



function propose_mcmc(
    iter::TransformedMCMCIterator{<:Any, <:Any, <:Any, <:TransformedMHProposal}
)
    @unpack μ, f_transform, proposal, samples, sample_z, stepno, context = iter
    rng = get_rng(context)
    sample_x = last(samples)
    x, logd_x = sample_x.v, sample_x.logd
    z, logd_z = sample_z.v, sample_z.logd

    n = size(z, 1)
    z_proposed = z + rand(rng, proposal.proposal_dist, n) #TODO: check if proposal is symmetric? otherwise need additional factor?
    x_proposed, ladj = with_logabsdet_jacobian(f_transform, z_proposed)
    logd_x_proposed = BAT.checked_logdensityof(μ, x_proposed)
    logd_z_proposed = logd_x_proposed + ladj
    @assert logd_z_proposed ≈ logdensityof(MeasureBase.pullback(f_transform, μ), z_proposed) #TODO: remove

    
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

    p_accept = clamp(exp(logd_z_proposed-logd_z), 0, 1)

    sample_z_proposed = _rebuild_density_sample(sample_z, z_proposed, logd_z_proposed)
    sample_x_proposed = _rebuild_density_sample(sample_x, x_proposed, logd_x_proposed)

    return sample_x_proposed, sample_z_proposed, p_accept
end



function transformed_mcmc_step!!(
    iter::TransformedMCMCIterator,
    tuner::TransformedAbstractMCMCTunerInstance,
    tempering::TransformedMCMCTemperingInstance,
)
    @unpack  μ, f_transform, proposal, samples, sample_z, stepno, context = iter
    rng = get_rng(context)
    sample_x = last(samples)
    x, logd_x = sample_x.v, sample_x.logd
    z, logd_z = sample_z.v, sample_z.logd
    @unpack n_accepted, stepno = iter

    sample_x_proposed, sample_z_proposed, p_accept = propose_mcmc(iter)

    z_proposed, logd_z_proposed = sample_z_proposed.v, sample_z_proposed.logd
    x_proposed, logd_x_proposed = sample_x_proposed.v, sample_x_proposed.logd

    tuner_new, f_transform = tune_mcmc_transform!!(tuner, f_transform, p_accept, z_proposed, z, stepno, context)
    
    accepted = rand(rng) <= p_accept

    # f_transform may have changed
    inverse_f = inverse(f_transform)
    x_new, z_new, logd_x_new, logd_z_new = if accepted
        x_proposed, inverse_f(x_proposed), logd_x_proposed, logd_z_proposed
    else
        x, inverse_f(x), logd_x, logd_z
    end

    sample_x_new, sample_z_new, samples_new = if accepted
        sample_x_new = DensitySample(x_new, logd_x_new, 1, TransformedMCMCTransformedSampleID(iter.info.id, iter.info.cycle, iter.stepno), nothing)
        push!(samples, sample_x_new) 
        sample_x_new, _rebuild_density_sample(sample_z, z_new, logd_z_new), samples
    else
        samples.weight[end] += 1
        _rebuild_density_sample(sample_x, x_new, logd_x_new, sample_x.weight+1), _rebuild_density_sample(sample_z, z_new, logd_z_new), samples
    end

    tempering_new, μ_new = temper_mcmc_target!!(tempering, μ, stepno)

    f_new = f_transform

    # iter_new = TransformedMCMCIterator(μ_new, f_new, proposal, samples_new, sample_z_new, stepno, n_accepted+Int(accepted), context)
    iter.μ, iter.f_transform, iter.samples, iter.sample_z = μ_new, f_new, samples_new, sample_z_new
    iter.n_accepted += Int(accepted)
    iter.stepno += 1
    @assert iter.context === context

    return (iter, tuner_new, tempering_new)
end



function transformed_mcmc_iterate!(
    chain::TransformedMCMCIterator,
    tuner::TransformedAbstractMCMCTunerInstance,
    tempering::TransformedMCMCTemperingInstance;
    max_nsteps::Integer = 1,
    max_time::Real = Inf,
    nonzero_weights::Bool = true,
    callback::Function = nop_func,
)
    @debug "Starting iteration over MCMC chain $(mcmc_info(chain).id) with $max_nsteps steps in max. $(@sprintf "%.1f seconds." max_time)"

    start_time = time()
    last_progress_message_time = start_time
    start_nsteps = nsteps(chain)
    start_nsteps = nsteps(chain)

    while (
        (nsteps(chain) - start_nsteps) < max_nsteps &&
        (time() - start_time) < max_time
    )
        transformed_mcmc_step!!(chain, tuner, tempering)
        callback(Val(:mcmc_step), chain)
  
        #TODO: output schemes

        current_time = time()
        elapsed_time = current_time - start_time
        logging_interval = 5 * round(log2(elapsed_time/60 + 1) + 1)
        if current_time - last_progress_message_time > logging_interval
            last_progress_message_time = current_time
            @debug "Iterating over MCMC chain $(mcmc_info(chain).id), completed $(nsteps(chain) - start_nsteps) (of $(max_nsteps)) steps and produced $(nsteps(chain) - start_nsteps) samples in $(@sprintf "%.1f s" elapsed_time) so far."
        end
    end

    current_time = time()
    elapsed_time = current_time - start_time
    @debug "Finished iteration over MCMC chain $(mcmc_info(chain).id), completed $(nsteps(chain) - start_nsteps) steps and produced $(nsteps(chain) - start_nsteps) samples in $(@sprintf "%.1f s" elapsed_time)."
    
    return nothing
end


function transformed_mcmc_iterate!(
    chain::MCMCIterator,
    tuner::TransformedAbstractMCMCTunerInstance,
    tempering::TransformedMCMCTemperingInstance;
    # tuner::TransformedAbstractMCMCTunerInstance;
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
    tuners::AbstractVector{<:TransformedAbstractMCMCTunerInstance},
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


#=
# Unused?
function reset_chain(
    rng::AbstractRNG,
    chain::TransformedMCMCIterator,
)
    rngpart_cycle = RNGPartition(rng, 0:(typemax(Int16) - 2))
    #TODO reset cycle count?
    chain.rngpart_cycle = rngpart_cycle
    chain.info = TransformedMCMCIteratorInfo(chain.info, cycle=0)
    chain.context = set_rng(chain.context, rng)
    # wants a next_cycle!
    # reset_rng_counters!(chain)
end
=#


function reset_rng_counters!(chain::TransformedMCMCIterator)
    rng = get_rng(get_context(chain))
    set_rng!(rng, chain.rngpart_cycle, chain.info.cycle)
    rngpart_step = RNGPartition(rng, 0:(typemax(Int32) - 2))
    set_rng!(rng, rngpart_step, chain.stepno)
    nothing
end


function next_cycle!(
    chain::TransformedMCMCIterator,

)
    chain.info = TransformedMCMCIteratorInfo(chain.info, cycle = chain.info.cycle + 1)
    chain.stepno = 0

    reset_rng_counters!(chain)

    chain.samples[1] = last(chain.samples)
    resize!(chain.samples, 1)

    chain.samples.weight[1] = 1
    chain.samples.info[1] = TransformedMCMCTransformedSampleID(chain.info.id, chain.info.cycle, chain.stepno)
    
    chain
end

