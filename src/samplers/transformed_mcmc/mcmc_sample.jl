abstract type TransformedMCMCProposal end
"""
    BAT.TransformedMHProposal

*BAT-internal, not part of stable public API.*
"""
struct TransformedMHProposal{
    D<:Union{Distribution, AbstractMeasure}
}<: TransformedMCMCProposal
    proposal_dist::D
end      


# TODO AC: find a better solution for this. Problem is that in the with_kw constructor below, we need to dispatch on this type.
struct TransformedMCMCDispatch end

@with_kw struct TransformedMCMCSampling{
    TR<:AbstractTransformTarget,
    IN<:TransformedMCMCInitAlgorithm,
    BI<:TransformedMCMCBurninAlgorithm,
    CT<:ConvergenceTest,
    CB<:Function
} <: AbstractSamplingAlgorithm
    pre_transform::TR = bat_default(TransformedMCMCDispatch, Val(:pre_transform))
    tuning_alg::TransformedMCMCTuningAlgorithm = TransformedRAMTuner() # TODO: use bat_defaults
    adaptive_transform::AdaptiveTransformSpec = default_adaptive_transform(tuning_alg)
    proposal::TransformedMCMCProposal = TransformedMHProposal(Normal()) #TODO: use bat_defaults
    tempering = TransformedNoTransformedMCMCTempering() # TODO: use bat_defaults
    nchains::Int = 4
    nsteps::Int = 10^5
    #TODO: max_time ?
    init::IN =  bat_default(TransformedMCMCDispatch, Val(:init), pre_transform, nchains, nsteps) #TransformedMCMCChainPoolInit()#TODO AC: use bat_defaults bat_default(MCMCSampling, Val(:init), MetropolisHastings(), pre_transform, nchains, nsteps) #TODO
    burnin::BI = bat_default(TransformedMCMCDispatch, Val(:burnin), pre_transform, nchains, nsteps)
    convergence::CT = TransformedBrooksGelmanConvergence()
    strict::Bool = true
    store_burnin::Bool = false
    nonzero_weights::Bool = true
    callback::CB = nop_func
end

bat_default(::Type{TransformedMCMCDispatch}, ::Val{:pre_transform}) = PriorToGaussian()

bat_default(::Type{TransformedMCMCDispatch}, ::Val{:nsteps}, trafo::AbstractTransformTarget, nchains::Integer) = 10^5

bat_default(::Type{TransformedMCMCDispatch}, ::Val{:init}, trafo::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    TransformedMCMCChainPoolInit(nsteps_init = max(div(nsteps, 100), 250))

bat_default(::Type{TransformedMCMCDispatch}, ::Val{:burnin}, trafo::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    TransformedMCMCMultiCycleBurnin(nsteps_per_cycle = max(div(nsteps, 10), 2500))



function bat_sample_impl(
    target::BATMeasure,
    algorithm::TransformedMCMCSampling,
    context::BATContext
)
    m, trafo = transform_and_unshape(algorithm.pre_transform, target, context)

    init = mcmc_init!(
        algorithm,
        m,
        algorithm.nchains,
        apply_trafo_to_init(trafo, algorithm.init),
        algorithm.tuning_alg,
        algorithm.nonzero_weights,
        algorithm.store_burnin ? algorithm.callback : nop_func,
        context
    )
    
    @unpack chains, tuners, temperers = init

    # output_init = reduce(vcat, getproperty(chains, :samples))

    burnin_outputs_coll = if algorithm.store_burnin
        DensitySampleVector(first(chains))
    else
        nothing
    end

    # burnin and tuning 
    mcmc_burnin!(
        burnin_outputs_coll,
        chains,
        tuners,
        temperers,
        algorithm.burnin,
        algorithm.convergence,
        algorithm.strict,
        algorithm.nonzero_weights,
        algorithm.store_burnin ? algorithm.callback : nop_func
    )

    # sampling
    run_sampling  = _run_sample_impl(
        m,
        algorithm,
        chains,
    )
    samples_trafo, generator = run_sampling.result_trafo, run_sampling.generator

    # prepend burnin samples to output
    if algorithm.store_burnin
        burnin_samples_trafo = varshape(m).(burnin_outputs_coll)
        append!(burnin_samples_trafo, samples_trafo)
        samples_trafo = burnin_samples_trafo
    end

    samples_notrafo = inverse(trafo).(samples_trafo)
    

    (result = samples_notrafo, result_trafo = samples_trafo, trafo = trafo, generator = TransformedMCMCSampleGenerator(chains, algorithm))
end

#=
function _bat_sample_continue(
    target::BATMeasure,
    generator::TransformedMCMCSampleGenerator,
    ;description::AbstractString = "MCMC iterate"
)
    @unpack algorithm, chains = generator
    density_notrafo = convert(BATMeasure, target)
    density, trafo = transform_and_unshape(algorithm.pre_transform, density_notrafo)

    run_sampling = _run_sample_impl(density, algorithm, chains, description=description)

    samples_trafo, generator = run_sampling.result_trafo, run_sampling.generator

    samples_notrafo = inverse(trafo).(samples_trafo)

    (result = samples_notrafo, result_trafo = samples_trafo, trafo = trafo, generator = TransformedMCMCSampleGenerator(chains, algorithm))
end
=#

function _run_sample_impl(
    m::BATMeasure,
    algorithm::TransformedMCMCSampling,
    chains::AbstractVector{<:MCMCIterator},
    ;description::AbstractString = "MCMC iterate"
)
    next_cycle!.(chains) 

    progress_meter = ProgressMeter.Progress(algorithm.nchains*algorithm.nsteps, desc=description, barlen=80-length(description), dt=0.1)

    # tuners are set to 'NoOpTuner' for the sampling phase
    transformed_mcmc_iterate!(
        chains,
        get_tuner.(Ref(TransformedMCMCNoOpTuning()),chains),
        get_temperer.(Ref(TransformedNoTransformedMCMCTempering()), chains),
        max_nsteps = algorithm.nsteps, #TODO: maxtime
        nonzero_weights = algorithm.nonzero_weights,
        callback = (kwargs...) -> let pm=progress_meter; ProgressMeter.next!(pm) ; end,
    )
    ProgressMeter.finish!(progress_meter)

    output = reduce(vcat, getproperty.(chains, :samples))
    samples_trafo = varshape(m).(output)

    (result_trafo = samples_trafo, generator = TransformedMCMCSampleGenerator(chains, algorithm))
end
