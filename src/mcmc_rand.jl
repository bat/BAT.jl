# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function Base.rand!(
    result::Tuple{DensitySampleVector, MCMCBasicStats},
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
    result_samples = result[1]
    result_stats = result[2]

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
    stats = MCMCBasicStats.(chains)
    callbacks = [mcmc_callback(granularity, (samples[i], stats[i])) for i in eachindex(chains)]

    mcmc_iterate!(
        callbacks,
        chains,
        exec_context;
        max_nsamples = nsamples,
        max_nsteps = max_nsteps,
        max_time = max_time,
        ll = ll
    )

    for x in samples
        append!(result_samples, x)
    end

    for x in stats
        merge!(result_stats, x)
    end

    result
end


function Base.rand(
    chainspec::MCMCSpec,
    nsamples::Integer,
    nchains::Integer,
    exec_context::ExecContext = ExecContext();
    tuner_config::AbstractMCMCTunerConfig = AbstractMCMCTunerConfig(chainspec.algorithm),
    convergence_test::MCMCConvergenceTest = GRConvergence(),
    init_strategy::MCMCInitStrategy = MCMCInitStrategy(tuner_config),
    burnin_strategy::MCMCBurninStrategy = MCMCBurninStrategy(tuner_config),
    kwargs...
)
    result = (
        DensitySampleVector(chainspec(0)),
        MCMCBasicStats(chainspec(0))
    )

    rand!(
        result,
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
end
