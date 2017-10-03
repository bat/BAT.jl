# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function Base.rand(
    chainspec::MCMCSpec,
    nsamples::Integer,
    nchains::Integer,
    exec_context::ExecContext = ExecContext(),
    tunerconfig::AbstractMCMCTunerConfig = AbstractMCMCTunerConfig(chainspec.algorithm),
    convergence_test::MCMCConvergenceTest = GRConvergence();
    max_nsteps::Int = 100 * nsamples,
    max_time::Float64 = Inf,
    granularity::Int = 1,  # Keep this? Potentially dangerous, in the wrong hands.
    ll::LogLevel = LOG_INFO
)
    chains = chainspec(1:nchains, exec_context)

    samples_per_tuning_cycle = Int64(max(round(Int, nsamples / 10), 1000))

    mcmc_tune_burnin!(
        (),
        chains,
        exec_context,
        tunerconfig,
        max_nsamples_per_cycle = Int64(samples_per_tuning_cycle),
        max_nsteps_per_cycle = Int(10*samples_per_tuning_cycle),
        max_time_per_cycle = max_time / 10,
        max_ncycles = 30,
        ll = ll
    )

    samples = DensitySampleVector.(chains)
    cb = [mcmc_callback(granularity, samples[i]) for i in eachindex(chains)]
    # cb = mcmc_callback.(samples)

    mcmc_iterate!(cb, chains, exec_context, max_nsamples = nsamples, max_nsteps = max_nsteps, max_time = max_time, ll = ll)

    merge(samples...)
end
