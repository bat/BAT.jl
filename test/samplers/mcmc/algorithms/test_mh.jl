# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using LinearAlgebra
using StatsBase, Distributions, StatsBase, ValueShapes

@testset "mh" begin
    rng = bat_rng()
    target = NamedTupleDist(a = Normal(), b = MvNormal(Diagonal(fill(1.0, 2))))

    density = @inferred(convert(AbstractDensity, target))
    @test density isa BAT.DistributionDensity

    mcmc_alg = MetropolisHastings()
    nchains = 4
    n = 10^4
    init_alg = MCMCChainPoolInit()
    tuning_alg = AdaptiveMHTuning()
    burnin_alg = MCMCMultiCycleBurnin()
    convergence_test = BrooksGelmanConvergence()
    strict = true
    nonzero_weights = true
    callback = x -> nothing
    max_neval = 10 * n
    max_time = Inf

    x = bat_initval(rng, density, InitFromTarget()).result
    @inferred(MCMCIterator(deepcopy(rng), mcmc_alg, density, 1, unshaped(x, varshape(density)))) isa BAT.MHIterator

    init_result = BAT.mcmc_init!(
        rng,
        mcmc_alg,
        density,
        nchains,
        init_alg,
        tuning_alg,
        callback
    )
#=
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
=#
end
