# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, Random123


@testset "MCMC callback contract" begin
    function sample_with_callback(;
        callback = BAT.nop_func,
        store_burnin = true,
        max_ncycles = 1,
        nsteps_final = 5,
        init = MCMCChainPoolInit(nsteps_init = 30),
    )
        algorithm = TransformedMCMC(
            nchains = 2,
            nsteps = 20,
            strict = false,
            store_burnin = store_burnin,
            adaptive_transform = BAT.NoAdaptiveTransform(),
            transform_tuning = NoMCMCTransformTuning(),
            init = init,
            convergence = AssumeConvergence(),
            burnin = MCMCMultiCycleBurnin(
                nsteps_per_cycle = 30,
                max_ncycles = max_ncycles,
                nsteps_final = nsteps_final,
            ),
            callback = callback,
        )
        return bat_sample(Normal(), algorithm, BATContext(rng = Philox4x((564, 84)))).result
    end

    @testset "Cycle-only callback through tuning and stabilization" begin
        for init in (MCMCChainPoolInit(nsteps_init = 30), MCMCRetryInit(nsteps_init = 30))
            calls = NamedTuple[]
            coordinator = current_task()
            function cycle_callback(::Val{:mcmc_burnin}, states::AbstractVector)
                push!(calls, (
                    task = current_task(),
                    chainids = [s.chain_state.info.id for s in states],
                    steps = [s.chain_state.stepno for s in states],
                ))
                return :ignored
            end

            samples = sample_with_callback(; callback = cycle_callback, init = init)
            @test !isempty(samples)
            @test length(calls) == 1
            @test only(calls).task === coordinator
            @test length(unique(only(calls).chainids)) == 2
            @test only(calls).steps == [30, 30]
        end
    end

    @testset "Stabilization does not emit user events" begin
        calls = Ref(0)
        cycle_callback(::Val{:mcmc_burnin}, ::AbstractVector) = (calls[] += 1)
        samples = sample_with_callback(; callback = cycle_callback, max_ncycles = 0)
        @test !isempty(samples)
        @test calls[] == 0
    end

    @testset "Public event sequence and sample reproducibility" begin
        events = Any[]
        record_event(event, state) = push!(events, event)
        observed = sample_with_callback(; callback = record_event)
        expected = sample_with_callback()
        @test events == [Val(:mcmc_burnin)]
        @test observed.v == expected.v
        @test observed.weight == expected.weight
        @test observed.logd == expected.logd
        @test observed.info == expected.info
    end

    @testset "Without stored burn-in the user callback is disabled" begin
        unexpected_event(args...) = error("Unexpected user callback without stored burn-in")
        samples = sample_with_callback(; callback = unexpected_event, store_burnin = false)
        @test !isempty(samples)
        @test sum(samples.weight) == 40
    end

    @testset "Cycle callback errors propagate unchanged" begin
        # Include MethodError: dispatch guards must not swallow errors raised
        # inside an otherwise valid callback.
        for failure in (ErrorException("Intentional cycle failure"), MethodError(sin, (nothing,)))
            fail_cycle(::Val{:mcmc_burnin}, ::AbstractVector) = throw(failure)
            caught = try
                sample_with_callback(; callback = fail_cycle)
                nothing
            catch err
                err
            end
            @test caught === failure
        end
    end
end
