# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# BAT-internal:
const MCMCOutputWithChains = Tuple{DensitySampleVector, MCMCBasicStats, AbstractVector{<:MCMCIterator}}

# BAT-internal:
function MCMCOutputWithChains(rng::AbstractRNG, chainspec::MCMCSpec)
    dummy_chain = chainspec(deepcopy(rng), zero(Int64))

    (
        DensitySampleVector(dummy_chain),
        MCMCBasicStats(dummy_chain),
        Vector{typeof(dummy_chain)}()
    )
end



# BAT-internal:
const MCMCOutput = Tuple{DensitySampleVector, MCMCBasicStats}

# BAT-internal:
function MCMCOutput(rng::AbstractRNG, chainspec::MCMCSpec)
    samples, stats = MCMCOutputWithChains(rng, chainspec::MCMCSpec)
    (samples, stats)
end



# BAT-internal:
function mcmc_sample(
    rng::AbstractRNG,
    chainspec::MCMCSpec,
    nsamples::Integer,
    nchains::Integer;
    max_nsteps::Int64 = Int64(10 * nsamples),
    max_time::Float64 = Inf,
    tuner_config::AbstractMCMCTuningStrategy = AbstractMCMCTuningStrategy(chainspec.algorithm),
    convergence_test::MCMCConvergenceTest = BrooksGelmanConvergence(),
    init_strategy::MCMCInitStrategy = MCMCInitStrategy(tuner_config),
    burnin_strategy::MCMCBurninStrategy = MCMCBurninStrategy(chainspec.algorithm, nsamples, max_nsteps, tuner_config),
    granularity::Int = 1,
    strict_mode::Bool = false
)
    result = MCMCOutputWithChains(rng, chainspec)

    result_samples, result_stats, result_chains = result

    (chains, tuners) = mcmc_init(
        rng,
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


# BAT-internal:
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



function bat_sample_impl(
    rng::AbstractRNG,
    posterior::AbstractPosteriorDensity,
    n::Tuple{Integer,Integer},
    algorithm::MCMCAlgorithm;
    max_nsteps::Integer = 10 * n[1],
    max_time::Real = Inf,
    tuning::AbstractMCMCTuningStrategy = AbstractMCMCTuningStrategy(algorithm),
    init::MCMCInitStrategy = MCMCInitStrategy(tuning),
    burnin::MCMCBurninStrategy = MCMCBurninStrategy(algorithm, n[1], max_nsteps, tuning),
    convergence::MCMCConvergenceTest = BrooksGelmanConvergence(),
    strict::Bool = false,
    filter::Bool = true
)
    chainspec = MCMCSpec(algorithm, posterior)

    nsamples_per_chain, nchains = n

    unshaped_samples, mcmc_stats, chains = mcmc_sample(
        rng,
        chainspec,
        nsamples_per_chain,
        nchains;
        tuner_config = tuning,
        convergence_test = convergence,
        init_strategy = init,
        burnin_strategy = burnin,
        max_nsteps = Int64(max_nsteps),
        max_time = Float64(max_time),
        granularity = filter ? 1 : 2,
        strict_mode = strict
    )

    samples = varshape(posterior).(unshaped_samples)
    (result = samples, chains = chains)
end


function bat_sample_impl(
    rng::AbstractRNG,
    posterior::AbstractPosteriorDensity,
    n::Integer,
    algorithm::MCMCAlgorithm;
    kwargs...
)
    nchains = 4
    nsamples = div(n, nchains)
    bat_sample_impl(rng, posterior, (nsamples, nchains), algorithm; kwargs...)
end
