# This file is a part of BAT.jl, licensed under the MIT License (MIT).


const MCMCOutputWithChains = Tuple{DensitySampleVector, MCMCSampleIDVector, MCMCBasicStats, AbstractVector{<:MCMCIterator}}

function MCMCOutputWithChains(chainspec::MCMCSpec)
    dummy_chain = chainspec(zero(Int64))

    (
        DensitySampleVector(dummy_chain),
        MCMCSampleIDVector(dummy_chain),
        MCMCBasicStats(dummy_chain),
        Vector{typeof(dummy_chain)}()
    )
end



const MCMCOutput = Tuple{DensitySampleVector, MCMCSampleIDVector, MCMCBasicStats}

function MCMCOutput(chainspec::MCMCSpec)
    samples, sampleids, stats = MCMCOutputWithChains(chainspec::MCMCSpec)
    (samples, sampleids, stats)
end



# TODO: Fix granularity forwarding (still an issue?)

function Random.rand(
    chainspec::MCMCSpec,
    nsamples::Integer,
    nchains::Integer,
    exec_context::ExecContext = ExecContext();
    tuner_config::AbstractMCMCTunerConfig = AbstractMCMCTunerConfig(chainspec),
    convergence_test::MCMCConvergenceTest = BGConvergence(),
    init_strategy::MCMCInitStrategy = MCMCInitStrategy(tuner_config),
    burnin_strategy::MCMCBurninStrategy = MCMCBurninStrategy(chainspec, nsamples, tuner_config),
    max_nsteps::Int64 = Int64(100 * nsamples),
    max_time::Float64 = Inf,
    granularity::Int = 1,
    strict_mode::Bool = false,
    ll::LogLevel = LOG_INFO,
)
    result = MCMCOutputWithChains(chainspec)

    result_samples, result_sampleids, result_stats, result_chains = result

    (chains, tuners) = mcmc_init(
        chainspec,
        nchains,
        tuner_config,
        init_strategy,
        exec_context;
        ll = ll,
    )

    mcmc_tune_burnin!(
        (),
        tuners,
        chains,
        convergence_test,
        burnin_strategy,
        exec_context;
        strict_mode = strict_mode,
        ll = ll
    )

    append!(result_chains, chains)

    rand!(
        (result_samples, result_sampleids, result_stats),
        result_chains,
        nsamples,
        exec_context;
        max_nsteps = max_nsteps,
        max_time = max_time,
        granularity = granularity,
        ll = ll
    )

    result
end


function Random.rand!(
    result::MCMCOutput,
    chains::AbstractVector{<:MCMCIterator},
    nsamples::Integer,
    exec_context::ExecContext;
    max_nsteps::Int64 = Int64(100 * nsamples),
    max_time::Float64 = Inf,
    granularity::Int = 1,
    ll::LogLevel = LOG_INFO,
)
    result_samples, result_sampleids, result_stats = result

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

    result
end


# # ToDo ?:
# function Random.rand!(
#     result::Tuple{DensitySampleVector, MCMCSampleIDVector, MCMCBasicStats},
#     chainspec::MCMCSpec,
#     nsamples::Integer,
#     initial_params::VectorOfSimilarVectors{<:Real},
#     ...
# )
#     ...
# end
