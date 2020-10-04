# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc doc"""
    MCMCSampling <: AbstractSamplingAlgorithm

Constructor:

```julia
MCMCSampling(;
    algorithm::MCMCAlgorithm = MetropolisHastings(),
    init::MCMCInitAlgorithm = MCMCChainPoolInit(),
    burnin::MCMCBurninAlgorithm = MCMCMultiCycleBurnin(),
    convergence::MCMCConvergenceTest = BrooksGelmanConvergence(),
    strict::Bool = false,
)
```
"""
@with_kw struct MCMCSampling{
    AL<:MCMCAlgorithm,
    IN<:MCMCInitAlgorithm,
    BI<:MCMCBurninAlgorithm,
    CT<:MCMCConvergenceTest
} <: AbstractSamplingAlgorithm
    algorithm::AL = MetropolisHastings(),
    init::IN = MCMCChainPoolInit(),
    burnin::BI = MCMCMultiCycleBurnin(),
    convergence::CT = BrooksGelmanConvergence(),
    strict::Bool = false,
end

export MCMCSampling


function bat_sample_impl(
    rng::AbstractRNG,
    target::PosteriorDensity,
    n::Union{Integer, Tuple{Integer,Integer}},
    algorithm::MCMCAlgorithm;
    max_nsteps::Integer = 10 * _mcmc_nsamples_tuple(n)[1],
    max_time::Real = Inf,
    tuning::MCMCTuningAlgorithm = MCMCTuningAlgorithm(algorithm),
    init::MCMCInitAlgorithm = MCMCChainPoolInit(),
    burnin::MCMCBurninAlgorithm = MCMCMultiCycleBurnin(
        max_nsamples_per_cycle = max(div(_mcmc_nsamples_tuple(n)[1], 10), 100)
        max_nsteps_per_cycle = max(div(max_nsteps, 10), 100)
    ),
    convergence::MCMCConvergenceTest = BrooksGelmanConvergence(),
    strict::Bool = false,
    filter::Bool = true
)
    density = convert(AbstractDensity, target)

    nsamples_per_chain, nchains = _mcmc_nsamples_tuple(n)

    result = MCMCOutputWithChains(rng, chainspec)

    result_samples, result_stats, result_chains = result

    (chains, tuners) = mcmc_init!(
        init_output,
        rng,
        chainspec,
        nchains,
        tuning,
        init,
        callback = init_callback
    )

    mcmc_burnin!(
        burnin_output,
        tuners,
        chains,
        convergence,
        burnin,
        strict_mode = strict,
        callback = burnin_callback
    )

    append!(result_chains, chains)

    mcmc_sample!(
        (result_samples, result_stats),
        result_chains,
        nsamples_per_chain;
        max_nsteps = Int64(max_nsteps),
        max_time = Float64(max_time),
        granularity = filter ? 1 : 2
    )

    samples = varshape(density).(unshaped_samples)

    (result = samples, chains = chains)
end


_mcmc_nsamples_tuple(n::NTuple{2, Integer}) = n

function _mcmc_nsamples_tuple(n::Integer)
    nchains = 4
    nsamples = div(n, nchains)
    (nsamples, nchains)
end


# ToDo: Remove once more generalized init-value generation is in place:
function bat_sample_impl(
    rng::AbstractRNG,
    dist::Union{Distribution,Histogram},
    n::Any,
    algorithm::MCMCAlgorithm;
    kwargs...
)
    posterior = PosteriorDensity(LogDVal(0), dist)
    bat_sample_impl(rng, posterior, n, algorithm; kwargs...)
end
