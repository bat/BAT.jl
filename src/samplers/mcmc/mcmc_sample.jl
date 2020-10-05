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
    sampler::AL = MetropolisHastings(),
    nchains::Int = 4,
    init::IN = MCMCChainPoolInit(),
    burnin::BI = MCMCMultiCycleBurnin(),
    convergence::CT = BrooksGelmanConvergence(),
    strict::Bool = true,
    nonzero_weights::Bool = true
end

export MCMCSampling


function bat_sample_impl(
    rng::AbstractRNG,
    target::AnyDensityLike,
    n::Integer,
    algorithm::MCMCSampling;
    max_neval::Integer = 10 * n,
    max_time::Real = Inf
)
    density = convert(AbstractDensity, target)
    mcmc_algorithm = algorithm.sampler

    dummy_startpos = bat_initval(deepcopy(rng), density, InitFromTarget())
    dummy_chain = MCMCIterator(deepcopy(rng), mcmc_algorithm, density, 0, dummy_startpos)
    unshaped_samples = DensitySampleVector(dummy_chain)

    init_output = nothing
    init_callback = nop_func

    (chains, tuners) = mcmc_init!(
        rng,
        init_output,
        mcmc_algorithm,
        density
        algorithm.nchains,
        callback = init_callback
    )     

    burnin_output = nothing
    burnin_callback = nop_func

    mcmc_burnin!(
        burnin_output,
        tuners,
        chains,
        algorithm.burnin,
        algorithm.convergence,
        strict_mode = algorithm.strict,
        callback = burnin_callback
    )

    sampling_output = unshaped_samples
    samling_callback = nop_func

    mcmc_iterate!(
        sampling_output,
        chains;
        max_nsamples = div(n, length(chains)),
        max_nsteps = div(max_neval, length(chains)),
        max_time = max_time,
        nonzero_weights = nonzero_weights,
        callback = samling_callback
    )

    samples = varshape(density).(unshaped_samples)

    (result = samples, generator = MCMCSampleGenerator(chains))
end
