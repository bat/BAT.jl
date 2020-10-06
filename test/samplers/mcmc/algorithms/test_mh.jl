# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using LinearAlgebra
using StatsBase, Distributions, StatsBase, ValueShapes

@testset "mh" begin
    rng = bat_rng()
    target = NamedTupleDist(a = Normal(), b = MvNormal(Diagonal(fill(1.0, 2))))
    n = 10^4
    algorithm = MCMCSampling(sampler = MetropolisHastings(), nchains = 4)
    max_neval = 10 * n
    max_time = Inf

    density = convert(AbstractDensity, target)
    mcmc_algorithm = algorithm.sampler

    let
        chain = MCMCIterator(deepcopy(rng), mcmc_algorithm, density, 0, Float32)
    end

    (chains, tuners, chain_outputs) = mcmc_init!(
        rng,
        mcmc_algorithm,
        density,
        algorithm.nchains,
        callback = algorithm.store_burnin ? algorithm.callback : nop_func
    )     

    if !store_burnin
        empty!.(chain_outputs)
    end

    mcmc_burnin!(
        algorithm.store_burnin ? chain_outputs : nothing,
        tuners,
        chains,
        algorithm.burnin,
        algorithm.convergence,
        strict_mode = algorithm.strict,
        callback = algorithm.store_burnin ? algorithm.callback : nop_func
    )

    mcmc_iterate!(
        chain_outputs,
        chains;
        max_nsamples = div(n, length(chains)),
        max_nsteps = div(max_neval, length(chains)),
        max_time = max_time,
        nonzero_weights = nonzero_weights,
        callback = algorithm.callback
    )

    output = DensitySampleVector(first(chains))
    isnothing(output) || append!.(Ref(output), chain_outputs)
    samples = varshape(density).(output)

    (result = samples, generator = MCMCSampleGenerator(chains))
end
