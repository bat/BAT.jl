# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function Base.rand(
    chainspec::MCMCSpec,
    nsamples::Integer,
    nchains::Integer,
    exec_context::ExecContext = ExecContext(),
    tuner_config::AbstractMCMCTunerConfig = AbstractMCMCTunerConfig(chainspec.algorithm),
    convergence_test::MCMCConvergenceTest = GRConvergence();
    max_nsteps::Int = 100 * nsamples,
    max_time::Float64 = Inf,
    ninit_tries_per_chain::ClosedInterval{Int64} = 4..128,
    #max_nsamples_pretune::Int64 = Int64(1000),
    #max_nsteps_pretune::Int64 = Int64(10000),
    #max_time_pretune::Float64 = Inf,
    max_tuning_cycles::Int = 30,
    #max_nsamples_per_tuning_cycle = Int64(max(round(Int, nsamples / 10), 1000)),
    #max_nsteps_per_tuning_cycle = Int64(10*max_nsamples_per_tuning_cycle),
    #max_time_per_tuning_cycle = max_time / 10,
    granularity::Int = 1,
    strict_mode::Bool = false,
    ll::LogLevel = LOG_INFO,
)
    tuners = mcmc_init(
        chainspec,
        nchains,
        exec_context,
        tuner_config,
        convergence_test;
        #ninit_tries_per_chain = ninit_tries_per_chain,
        #max_nsamples_pretune = max_nsamples_pretune,
        #max_nsteps_pretune = max_nsteps_pretune,
        #max_time_pretune = max_time_pretune,
        ll = ll,
    )

    mcmc_tune_burnin!(
        (),
        tuners,
        convergence_test,
        exec_context;
        #max_nsamples_per_cycle = max_nsamples_per_tuning_cycle,
        #max_nsteps_per_cycle = max_nsteps_per_tuning_cycle,
        #max_time_per_cycle = max_time_per_tuning_cycle,
        max_ncycles = max_tuning_cycles,
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
