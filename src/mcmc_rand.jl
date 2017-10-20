# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function Base.rand(
    chainspec::MCMCSpec,
    nsamples::Integer,
    nchains::Integer,
    tuner_config::AbstractMCMCTunerConfig,
    convergence_test::MCMCConvergenceTest,
    init_strategy::MCMCInitStrategy,
    burnin_strategy::MCMCBurninStrategy,
    exec_context::ExecContext;
    max_nsteps::Int = 100 * nsamples,
    max_time::Float64 = Inf,
    granularity::Int = 1,
    strict_mode::Bool = false,
    ll::LogLevel = LOG_INFO,
)
    tuners = mcmc_init(
        chainspec,
        nchains,
        exec_context,
        tuner_config,
        convergence_test,
        init_strategy;
        ll = ll,
    )

    mcmc_tune_burnin!(
        (),
        tuners,
        convergence_test,
        burnin_strategy,
        exec_context;
        strict_mode = strict_mode,
        ll = ll
    )

    chains = map(x -> x.chain, tuners)

    samples = DensitySampleVector.(chains)
    callbacks = [mcmc_callback(granularity, samples[i]) for i in eachindex(chains)]

    mcmc_iterate!(
        callbacks,
        chains,
        exec_context;
        max_nsamples = nsamples,
        max_nsteps = max_nsteps,
        max_time = max_time,
        ll = ll
    )

    result = DensitySampleVector(tuners[1].chain)
    for s in samples
        append!(result, s)
    end
    result
end


Base.rand(
    chainspec::MCMCSpec,
    nsamples::Integer,
    nchains::Integer,
    exec_context::ExecContext = ExecContext();
    tuner_config::AbstractMCMCTunerConfig = AbstractMCMCTunerConfig(chainspec.algorithm),
    convergence_test::MCMCConvergenceTest = GRConvergence(),
    init_strategy::MCMCInitStrategy = MCMCInitStrategy(tuner_config),
    burnin_strategy::MCMCBurninStrategy = MCMCBurninStrategy(tuner_config),
    kwargs...
) = rand(
    chainspec,
    nsamples,
    nchains,
    tuner_config,
    convergence_test,
    init_strategy,
    burnin_strategy,
    exec_context;
    kwargs...
)
