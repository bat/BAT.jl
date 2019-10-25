# This file is a part of BAT.jl, licensed under the MIT License (MIT).


const MCMCOutputWithChains = Tuple{DensitySampleVector, MCMCBasicStats, AbstractVector{<:MCMCIterator}}

function MCMCOutputWithChains(chainspec::MCMCSpec)
    dummy_chain = chainspec(zero(Int64))

    (
        DensitySampleVector(dummy_chain),
        MCMCBasicStats(dummy_chain),
        Vector{typeof(dummy_chain)}()
    )
end



const MCMCOutput = Tuple{DensitySampleVector, MCMCBasicStats}

function MCMCOutput(chainspec::MCMCSpec)
    samples, stats = MCMCOutputWithChains(chainspec::MCMCSpec)
    (samples, stats)
end



# TODO: Fix granularity forwarding (still an issue?)

function mcmc_sample(
    chainspec::MCMCSpec,
    nsamples::Integer,
    nchains::Integer;
    tuner_config::AbstractMCMCTunerConfig = AbstractMCMCTunerConfig(chainspec),
    convergence_test::MCMCConvergenceTest = BGConvergence(),
    init_strategy::MCMCInitStrategy = MCMCInitStrategy(tuner_config),
    burnin_strategy::MCMCBurninStrategy = MCMCBurninStrategy(chainspec, nsamples, tuner_config),
    max_nsteps::Int64 = Int64(100 * nsamples),
    max_time::Float64 = Inf,
    granularity::Int = 1,
    strict_mode::Bool = false
)
    result = MCMCOutputWithChains(chainspec)

    result_samples, result_stats, result_chains = result

    (chains, tuners) = mcmc_init(
        chainspec,
        nchains,
        tuner_config,
        init_strategy
    )

    mcmc_tune_burnin!(
        (),
        tuners,
        chains,
        convergence_test,
        burnin_strategy;
        strict_mode = strict_mode
    )

    append!(result_chains, chains)

    mcmc_sample!(
        (result_samples, result_stats),
        result_chains,
        nsamples;
        max_nsteps = max_nsteps,
        max_time = max_time,
        granularity = granularity
    )

    result
end


function mcmc_sample!(
    result::MCMCOutput,
    chains::AbstractVector{<:MCMCIterator},
    nsamples::Integer;
    max_nsteps::Int64 = Int64(100 * nsamples),
    max_time::Float64 = Inf,
    granularity::Int = 1
)
    result_samples, result_stats = result

    samples = DensitySampleVector.(chains)
    stats = MCMCBasicStats.(chains)

    nonzero_weights = granularity <= 1
    callbacks = [
        MCMCMultiCallback(
            MCMCAppendCallback(samples[i], nonzero_weights),
            MCMCAppendCallback(stats[i], nonzero_weights)
        ) for i in eachindex(chains)
    ]

    mcmc_iterate!(
        callbacks,
        chains;
        max_nsamples = Int64(nsamples),
        max_nsteps = max_nsteps,
        max_time = max_time
    )

    for x in samples
        merge!(result_samples, x)
    end

    for x in stats
        merge!(result_stats, x)
    end

    result
end
