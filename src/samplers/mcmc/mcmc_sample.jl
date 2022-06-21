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
    TR<:AbstractDensityTransformTarget,
    IN<:MCMCInitAlgorithm,
    BI<:MCMCBurninAlgorithm,
    CT<:MCMCConvergenceTest,
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


function bat_sample_impl(
    rng::AbstractRNG,
    target::AnyDensityLike,
    algorithm::MCMCSampling
)
    
    density_notrafo = convert(AbstractDensity, target)
    density, trafo = transform_and_unshape(algorithm.trafo, density_notrafo)
    mcmc_algorithm = algorithm.mcalg

    (chains, tuners, chain_outputs) = mcmc_init!(
        rng,
        mcmc_algorithm,
        density,
        algorithm.nchains,
        apply_trafo_to_init(trafo, algorithm.init),
        get_mcmc_tuning(mcmc_algorithm),
        algorithm.nonzero_weights,
        algorithm.store_burnin ? algorithm.callback : nop_func
    )

    if !algorithm.store_burnin
        chain_outputs .= DensitySampleVector.(chains)
    end

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

    mcmc_iterate!(
        chain_outputs,
        chains;
        max_nsteps = algorithm.nsteps,
        nonzero_weights = algorithm.nonzero_weights,
        callback = algorithm.callback
    )

    output = DensitySampleVector(first(chains))
    isnothing(output) || append!.(Ref(output), chain_outputs)
    samples_trafo = varshape(density).(output)

    samples_notrafo = inverse(trafo).(samples_trafo)

    isvalid = check_convergence(algorithm.convergence, samples_notrafo).converged

    (result = samples_notrafo, result_trafo = samples_trafo, trafo = trafo, generator = MCMCSampleGenerator(chains), isvalid = isvalid)
end
