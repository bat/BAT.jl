# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, StatsBase, ValueShapes
using LinearAlgebra
using ForwardDiff, Zygote, DistributionsAD

@testset "zygote" begin
    posterior = ZygoteAD()|BAT.example_posterior_with_dirichlet()

    target, trafo = bat_transform(PriorToGaussian(), posterior)
    v = bat_initval(target).result
    f = logdensityof(target)
    gradlogp_zg = bat_valgrad(f).result
    gradlogp_fd = bat_valgrad(logdensityof(bat_transform(PriorToGaussian(), ForwardDiffAD() | BAT.example_posterior_with_dirichlet()).result)).result
    @test @inferred(gradlogp_zg(v))[1] ≈ gradlogp_fd(v)[1]
    @test @inferred(gradlogp_zg(v))[2] ≈ gradlogp_fd(v)[2]

    utarget, utrafo = bat_transform(PriorToGaussian(), unshaped(posterior))
    x = bat_initval(utarget).result
    @test @inferred(Zygote.jacobian(inv(utrafo), x)[1]) ≈ ForwardDiff.jacobian(inv(utrafo), x)

    hmc_sampling_alg = MCMCSampling(
        mcalg = HamiltonianMC(),
        nchains = 2,
        nsteps = 100,
        init = MCMCChainPoolInit(init_tries_per_chain = 2, nsteps_init = 5),
        burnin = MCMCMultiCycleBurnin(nsteps_per_cycle = 100, max_ncycles = 1),
        strict = false
    )

    @test bat_sample(posterior, hmc_sampling_alg).result isa DensitySampleVector
end
