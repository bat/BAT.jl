# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct MCMCSampling <: AbstractSamplingAlgorithm

Samples a probability density using Markov chain Monte Carlo.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MCMCSampling{
    AL<:MCMCAlgorithm,
    TR<:AbstractTransformTarget,
    IN<:MCMCInitAlgorithm,
    BI<:MCMCBurninAlgorithm,
    CT<:ConvergenceTest,
    CB<:Function
} <: AbstractSamplingAlgorithm
    mcalg::AL = MetropolisHastings()
    trafo::TR = bat_default(MCMCSampling, Val(:trafo), mcalg)
    nchains::Int = 4
    nsteps::Int = bat_default(MCMCSampling, Val(:nsteps), mcalg, trafo, nchains)
    init::IN = bat_default(MCMCSampling, Val(:init), mcalg, trafo, nchains, nsteps)
    burnin::BI = bat_default(MCMCSampling, Val(:burnin), mcalg, trafo, nchains, nsteps)
    convergence::CT = BrooksGelmanConvergence()
    strict::Bool = true
    store_burnin::Bool = false
    nonzero_weights::Bool = true
    callback::CB = nop_func
end

export MCMCSampling


function bat_sample_impl(m::BATMeasure, algorithm::MCMCSampling, context::BATContext)
    transformed_m, trafo = transform_and_unshape(algorithm.trafo, m, context)

    mcmc_algorithm = algorithm.mcalg

    (chains, tuners, chain_outputs) = mcmc_init!(
        mcmc_algorithm,
        transformed_m,
        algorithm.nchains,
        apply_trafo_to_init(trafo, algorithm.init),
        get_mcmc_tuning(mcmc_algorithm),
        algorithm.nonzero_weights,
        algorithm.store_burnin ? algorithm.callback : nop_func,
        context,
    )

    if !algorithm.store_burnin
        chain_outputs .= DensitySampleVector.(chains)
    end

    run_sampling = _run_sample_impl(transformed_m, algorithm, chains, tuners, context, chain_outputs=chain_outputs)
    samples_trafo, generator = run_sampling.result_trafo, run_sampling.generator

    samples_notrafo = inverse(trafo).(samples_trafo)

    (result = samples_notrafo, result_trafo = samples_trafo, trafo = trafo, generator = generator)
end

function _run_sample_impl(
    m::BATMeasure,
    algorithm::MCMCSampling,
    chains::AbstractVector{<:MCMCIterator},
    tuners,
    context::BATContext;
    description::AbstractString="MCMC iterate",
    chain_outputs=DensitySampleVector.(chains)
)
    mcmc_burnin!(
        algorithm.store_burnin ? chain_outputs : nothing,
        tuners,
        chains,
        algorithm.burnin,
        algorithm.convergence,
        algorithm.strict,
        algorithm.nonzero_weights,
        algorithm.store_burnin ? algorithm.callback : nop_func
    )

    next_cycle!.(chains)

    progress_meter = ProgressMeter.Progress(algorithm.nchains * algorithm.nsteps, desc=description, barlen=80 - length(description), dt=0.1)

    mcmc_iterate!(
        chain_outputs,
        chains;
        max_nsteps = algorithm.nsteps,
        nonzero_weights = algorithm.nonzero_weights,
        callback = (kwargs...) -> let pm=progress_meter, callback=algorithm.callback ; callback(kwargs) ; ProgressMeter.next!(pm) ; end,
    )

    ProgressMeter.finish!(progress_meter)

    output = DensitySampleVector(first(chains))
    isnothing(output) || append!.(Ref(output), chain_outputs)
    samples_trafo = varshape(m).(output)

    (result_trafo = samples_trafo, generator = MCMCSampleGenerator(chains))
end

function _bat_sample_continue(
    target::BATMeasure,
    algorithm::MCMCSampling,
    generator::MCMCSampleGenerator,
    context,
    ;description::AbstractString = "MCMC iterate"
)
    @unpack chains = generator
    m, trafo = transform_and_unshape(algorithm.trafo, target, context)

    chain_outputs = DensitySampleVector.(chains)

    tuners = map(v -> get_mcmc_tuning(getproperty(v, :algorithm))(v), chains)

    run_sampling = _run_sample_impl(m, algorithm, chains, tuners, context, description=description, chain_outputs=chain_outputs)
    samples_trafo, generator_new = run_sampling.result_trafo, run_sampling.generator

    smpls = inverse(trafo).(transformed_smpls)

    (result = samples_notrafo, result_trafo = samples_trafo, trafo = trafo, generator = generator_new)
end
