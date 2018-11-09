# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# TODO: Add MCMCSampleIDVector to output
# TODO: Fix granularity forwarding

function Random.rand!(
    result::Tuple{DensitySampleVector, MCMCSampleIDVector, MCMCBasicStats, AbstractVector{<:MCMCIterator}},
    chainspec::MCMCSpec,
    nsamples::Integer,
    nchains::Integer,
    tuner_config::AbstractMCMCTunerConfig,
    convergence_test::MCMCConvergenceTest,
    init_strategy::MCMCInitStrategy,
    burnin_strategy::MCMCBurninStrategy,
    exec_context::ExecContext;
    max_nsteps::Int64 = Int64(100 * nsamples),
    max_time::Float64 = Inf,
    granularity::Int = 1,
    strict_mode::Bool = false,
    ll::LogLevel = LOG_INFO,
)
    result_samples, result_sampleids, result_stats, result_chains = result

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
    sampleids = MCMCSampleIDVector.(chains)
    stats = MCMCBasicStats.(chains)

    nonzero_weights = granularity <= 1
    callbacks = [
        MCMCMultiCallback(
            MCMCAppendCallback(samples[i], nonzero_weights),
            MCMCAppendCallback(sampleids[i], nonzero_weights),
            MCMCAppendCallback(stats[i], nonzero_weights)
        ) for i in eachindex(chains)
    ]

    mcmc_iterate!(
        callbacks,
        chains,
        exec_context;
        max_nsamples = Int64(nsamples),
        max_nsteps = max_nsteps,
        max_time = max_time,
        ll = ll
    )

    for x in samples
        merge!(result_samples, x)
    end

    for x in sampleids
        merge!(result_sampleids, x)
    end

    for x in stats
        merge!(result_stats, x)
    end

    append!(result_chains, chains)

    result
end


# # ToDo:
# function Random.rand!(
#     result::Tuple{DensitySampleVector, MCMCSampleIDVector, MCMCBasicStats},
#     chainspec::MCMCSpec,
#     nsamples::Integer,
#     initial_params::VectorOfSimilarVectors{<:Real},
#     ...
# )
#     ...
# end


function Random.rand(
    chainspec::MCMCSpec,
    nsamples::Integer,
    nchains::Integer,
    exec_context::ExecContext = ExecContext();
    tuner_config::AbstractMCMCTunerConfig = AbstractMCMCTunerConfig(chainspec.algorithm),
    convergence_test::MCMCConvergenceTest = BGConvergence(),
    init_strategy::MCMCInitStrategy = MCMCInitStrategy(tuner_config),
    burnin_strategy::MCMCBurninStrategy = MCMCBurninStrategy(tuner_config),
    kwargs...
)
    dummy_chain = chainspec(zero(Int64))
    result = (
        DensitySampleVector(dummy_chain),
        MCMCSampleIDVector(dummy_chain),
        MCMCBasicStats(dummy_chain),
        Vector{typeof(dummy_chain)}()
    )

    Random.rand!(
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
