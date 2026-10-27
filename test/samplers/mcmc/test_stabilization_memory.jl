# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestStabilizationMemory

using BAT, Distributions, Logging, Test

function stabilization_memory(nsteps_final)
    calls = Ref(0)
    live_bytes = Int[]
    function likelihood(x)
        calls[] += 1
        if calls[] % 1000 == 0
            GC.gc()
            push!(live_bytes, Base.gc_live_bytes())
        end
        return BAT.LogDVal(-sum(abs2, x) / 2)
    end
    target = BAT.lbqintegral(likelihood, MvNormal(zeros(32), ones(32)))
    algorithm = TransformedMCMC(
        nchains = 1, nwalkers = 1, nsteps = 4,
        pretransform = DoNotTransform(),
        proposal = RandomWalk(proposaldist = Normal(0, 0.02)),
        adaptive_transform = BAT.NoAdaptiveTransform(),
        transform_tuning = NoMCMCTransformTuning(),
        convergence = AssumeConvergence(), strict = false,
        store_burnin = false,
        init = MCMCRetryInit(nsteps_init = 10),
        burnin = MCMCMultiCycleBurnin(
            nsteps_per_cycle = 10, max_ncycles = 1, nsteps_final = nsteps_final),
    )
    result = with_logger(NullLogger()) do
        BAT.evalmeasure(target, algorithm, BATContext(rng = BAT.Philox4x((42, 73))))
    end
    return BAT.samplesof(result), live_bytes
end

@testset "Discarded stabilization does not retain sample history" begin
    stabilization_memory(2000)
    samples, live_bytes = stabilization_memory(8000)
    @test sum(samples.weight) == 4
    @test length(live_bytes) == 8
    # Measure the live heap after warming compilation and collecting garbage.
    # A retained scratch history grows by over 2 MB; allow 1 MiB for runtime noise.
    @test last(live_bytes) - first(live_bytes) < 1024^2
end

end
