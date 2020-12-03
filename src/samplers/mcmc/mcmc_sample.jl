# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc doc"""
    MCMCSampling <: AbstractSamplingAlgorithm

Constructor:

```julia
MCMCSampling(;kwargs...)
```
"""
@with_kw struct MCMCSampling{
    AL<:MCMCAlgorithm,
    IN<:MCMCInitAlgorithm,
    BI<:MCMCBurninAlgorithm,
    CT<:MCMCConvergenceTest,
    CB<:Function
} <: AbstractSamplingAlgorithm
    sampler::AL = MetropolisHastings()
    nchains::Int = 4
    nsteps::Int = 10^5
    init::IN = MCMCChainPoolInit(
        max_nsteps_init = max(div(nsteps, 100), 250)
    )
    burnin::BI = MCMCMultiCycleBurnin(
        max_nsteps_per_cycle = max(div(nsteps, 10), 2500)
    )
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
    density = convert(AbstractDensity, target)
    mcmc_algorithm = algorithm.sampler

    (chains, tuners, chain_outputs) = mcmc_init!(
        rng,
        mcmc_algorithm,
        density,
        algorithm.nchains,
        algorithm.init,
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

    mcmc_iterate!(
        chain_outputs,
        chains;
        max_nsteps = algorithm.nsteps,
        max_time = Inf,
        nonzero_weights = algorithm.nonzero_weights,
        callback = algorithm.callback
    )

    output = DensitySampleVector(first(chains))
    isnothing(output) || append!.(Ref(output), chain_outputs)
    samples = varshape(density).(output)

    (result = samples, generator = MCMCSampleGenerator(chains))
end
