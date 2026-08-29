# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using Distributions, Logging, Random123


struct PointNonfiniteDistribution <: ContinuousUnivariateDistribution
    value::Float64
end

Distributions.logpdf(d::PointNonfiniteDistribution, x::Real) =
    x == 0.5 ? d.value : (insupport(d, x) ? 0.0 : -Inf)
Distributions.minimum(::PointNonfiniteDistribution) = 0.0
Distributions.maximum(::PointNonfiniteDistribution) = 1.0
Distributions.insupport(::PointNonfiniteDistribution, x::Real) = 0 <= x <= 1


mutable struct RetryInitFixture <: InitvalAlgorithm
    initial::Vector{Float64}
    draws::Vector{Float64}
    ncalls::Int
    threshold_retry::Bool
end

function BAT.bat_initval_impl(target, algorithm::RetryInitFixture, context::BATContext)
    algorithm.ncalls += 1
    if algorithm.ncalls <= length(algorithm.initial)
        return (result = [algorithm.initial[algorithm.ncalls]],)
    end

    u = rand(context.rng)
    push!(algorithm.draws, u)
    value = algorithm.threshold_retry && u > 0.5 ? 10.5 : 0.5
    (result = [value],)
end


function retry_sampling_algorithm(
    initval_alg; nwalkers = 1, max_init_tries = 3, init_strict = true
)
    TransformedMCMC(
        proposal = RandomWalk(proposaldist = Normal(0, 1e-12)),
        pretransform = DoNotTransform(),
        adaptive_transform = BAT.NoAdaptiveTransform(),
        init = MCMCRetryInit(
            max_init_tries = max_init_tries,
            nsteps_init = 1,
            initval_alg = initval_alg,
            strict = init_strict,
        ),
        burnin = MCMCMultiCycleBurnin(max_ncycles = 0, nsteps_final = 0),
        convergence = AssumeConvergence(),
        nchains = 1,
        nwalkers = nwalkers,
        nsteps = 2,
        strict = false,
        store_burnin = true,
    )
end

function run_retry(
    initval_alg; target = Uniform(0, 1), seed = (564, 83), kwargs...
)
    with_logger(NullLogger()) do
        bat_sample(
            target,
            retry_sampling_algorithm(initval_alg; kwargs...),
            BATContext(rng = Philox4x(seed)),
        )
    end
end


@testset "MCMCRetryInit" begin
    @testset "invalid initial target values fail" begin
        for logd in (NaN, Inf), strict in (true, false)
            target = PointNonfiniteDistribution(logd)
            @test_throws BAT.EvalException run_retry(
                ExplicitInit([0.5]); target, init_strict = strict
            )
        end
    end

    @testset "invalid retry target values fail" begin
        for logd in (NaN, Inf), strict in (true, false)
            initval_alg = RetryInitFixture([10.5], Float64[], 0, false)
            @test_throws BAT.EvalException run_retry(
                initval_alg;
                target = PointNonfiniteDistribution(logd),
                init_strict = strict,
            )
        end
    end

    @testset "rejects zero initialization tries" begin
        @test_throws ArgumentError MCMCRetryInit(max_init_tries = 0)
    end

    @testset "max_init_tries counts tested attempts" begin
        for ntries in (1, 3)
            initval_alg = RetryInitFixture(fill(10.5, ntries + 1), Float64[], 0, false)
            @test_throws ErrorException run_retry(
                initval_alg; max_init_tries = ntries
            )
            @test initval_alg.ncalls == ntries
        end

        last_success = RetryInitFixture([10.5], Float64[], 0, false)
        result = run_retry(last_success; max_init_tries = 2).result
        @test last_success.ncalls == 2
        @test all(isfinite, result.logd)
    end

    @testset "retry streams are fresh and reproducible" begin
        first_init = RetryInitFixture(Float64[], Float64[], 0, true)
        replay_init = RetryInitFixture(Float64[], Float64[], 0, true)
        first_result = run_retry(first_init; seed = (564, 1)).result
        replay_result = run_retry(replay_init; seed = (564, 1)).result

        @test first_init.draws == replay_init.draws
        @test length(first_init.draws) == 3
        @test allunique(first_init.draws)
        @test first_result == replay_result
        @test all(isfinite, first_result.logd)
    end

    @testset "retry stream follows logical walker" begin
        walker_two = RetryInitFixture([0.5, 10.5], Float64[], 0, false)
        both_walkers = RetryInitFixture([10.5, 10.5], Float64[], 0, false)
        run_retry(walker_two; nwalkers = 2)
        run_retry(both_walkers; nwalkers = 2)

        @test length(walker_two.draws) == 1
        @test length(both_walkers.draws) == 2
        @test walker_two.draws[1] == both_walkers.draws[2]
        @test both_walkers.draws[1] != both_walkers.draws[2]
    end
end
