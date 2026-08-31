# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, Logging, Random123


mutable struct RetryInitFixture <: InitvalAlgorithm
    calls::Int
end

function BAT.bat_initval_impl(::BAT.MeasureLike, init::RetryInitFixture, ::BATContext)
    init.calls += 1
    return (result = [init.calls == 1 ? 10.5 : 0.5],)
end


@testset "MCMCRetryInit" begin
    initval = RetryInitFixture(0)
    algorithm = TransformedMCMC(
        proposal = RandomWalk(proposaldist = Normal(0, 1e-12)),
        pretransform = DoNotTransform(),
        adaptive_transform = BAT.NoAdaptiveTransform(),
        init = MCMCRetryInit(
            max_init_tries = 2,
            nsteps_init = 1,
            initval_alg = initval,
        ),
        burnin = MCMCMultiCycleBurnin(max_ncycles = 0, nsteps_final = 0),
        convergence = AssumeConvergence(),
        nchains = 1,
        nwalkers = 1,
        nsteps = 2,
        nonzero_weights = false,
        strict = false,
    )
    result = with_logger(NullLogger()) do
        bat_sample(
            Uniform(0, 1),
            algorithm,
            BATContext(rng = Philox4x((564, 83))),
        ).result
    end

    @test all(isfinite, result.logd)
    @test initval.calls == 2
end
