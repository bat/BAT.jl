# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using LinearAlgebra
using StatsBase, Distributions, StatsBase, ValueShapes

@testset "mh" begin
    rng = bat_rng()
    target = NamedTupleDist(a = Normal(1, 1.5), b = MvNormal([-1.0, 2.0], [2.0 1.5; 1.5 3.0]))

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
    callback = (x...) -> nothing
    max_neval = 10 * n
    max_time = Inf

    x = bat_initval(rng, density, InitFromTarget()).result

    let
        @test @inferred(MCMCIterator(deepcopy(rng), mcmc_alg, density, 1, unshaped(x, varshape(density)))) isa BAT.MHIterator
        chain = @inferred(MCMCIterator(deepcopy(rng), mcmc_alg, density, 1, unshaped(x, varshape(density)))) 
        samples = DensitySampleVector(chain)
        BAT.mcmc_iterate!(samples, chain, max_nsamples = 10^5, max_nsteps = 10^5, nonzero_weights = false)
        @test chain.stepno == 10^5
        @test isapprox(length(samples), 10^5, atol = 20)
        @test length(samples) == sum(samples.weight)
        @test isapprox(mean(samples), [1, -1, 2], atol = 0.2)
        @test isapprox(cov(samples), cov(unshaped(target)), atol = 0.3)
    end

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
