# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, StatsBase, ValueShapes, DensityInterface, InverseFunctions
using IntervalSets
using ForwardDiff

@testset "forwarddiff" begin
    posterior = ForwardDiffAD()|BAT.example_posterior()

    mh_notrafo_sampling_alg = MCMCSampling(
        mcalg = MetropolisHastings(),
        nchains = 2,
        nsteps = 10^4,
        trafo = DoNotTransform(),
        burnin = MCMCMultiCycleBurnin(nsteps_per_cycle = 1000, max_ncycles = 2),
        strict = false
    )

    hmc_sampling_alg = MCMCSampling(
        mcalg = HamiltonianMC(),
        nchains = 2,
        nsteps = 100,
        init = MCMCChainPoolInit(init_tries_per_chain = 2..2, nsteps_init = 5),
        burnin = MCMCMultiCycleBurnin(nsteps_per_cycle = 100, max_ncycles = 1),
        strict = false
    )

    # Test basic sampling of posterior wrapped in WithDiff works, with all required functions forwarded:
    @test @inferred(bat_sample(posterior, mh_notrafo_sampling_alg)).result isa DensitySampleVector
    
    v = rand(posterior.prior)
    f = logdensityof(posterior)
    gradlogp = valgradof(f)
    @test isapprox(@inferred(gradlogp(v))[1], f(v), rtol = 10^-5)
    @test @inferred(gradlogp(v))[2] == gradient_shape(varshape(posterior))(ForwardDiff.gradient(unshaped(f), (unshaped(v, varshape(posterior)))))

    @test bat_sample(posterior, hmc_sampling_alg).result isa DensitySampleVector
end
