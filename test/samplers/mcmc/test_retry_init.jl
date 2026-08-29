# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using Distributions, Logging, Random123
import ForwardDiff


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

    @testset "retry stream follows original walker index" begin
        walker_two = RetryInitFixture([0.5, 10.5], Float64[], 0, false)
        both_walkers = RetryInitFixture([10.5, 10.5], Float64[], 0, false)
        run_retry(walker_two; nwalkers = 2)
        run_retry(both_walkers; nwalkers = 2)

        @test length(walker_two.draws) == 1
        @test length(both_walkers.draws) == 2
        @test walker_two.draws[1] == both_walkers.draws[2]
        @test both_walkers.draws[1] != both_walkers.draws[2]
    end

    @testset "multi-proposal retry invalidates MALA gradients" begin
        initval_alg = RetryInitFixture([0.25, 10.5], Float64[], 0, false)
        init_alg = MCMCRetryInit(
            max_init_tries = 2,
            nsteps_init = 1,
            initval_alg = initval_alg,
            strict = true,
        )
        proposals = BAT.MCMCProposal[
            MALAProposal(τ_base = 1e-4),
            RandomWalk(proposaldist = Normal(0, 1e-12)),
        ]
        algorithm = TransformedMCMC(
            proposal = MCMCMultiProposal(proposals, [10, 10]),
            proposal_tuning = MultiProposalTuning(BAT.MCMCProposalTuning[
                BAT.NoMCMCProposalTuning(), BAT.NoMCMCProposalTuning(),
            ]),
            pretransform = DoNotTransform(),
            adaptive_transform = BAT.NoAdaptiveTransform(),
            transform_tuning = BAT.NoMCMCTransformTuning(),
            tempering = BAT.NoMCMCTempering(),
            init = init_alg,
            nchains = 1,
            nwalkers = 2,
            nonzero_weights = true,
        )
        target = batmeasure(product_distribution(fill(truncated(Normal(), -2, 2), 1)))
        result = BAT.mcmc_init!(
            algorithm,
            target,
            init_alg,
            (args...) -> nothing,
            BATContext(rng = Philox4x((0x0564, 84)), ad = ForwardDiff),
        )
        state = only(result.mcmc_states)
        chain_state = state.chain_state
        rerolled_idx = only(findall(
            ==(Int32(2)), getproperty.(chain_state.current.x.info, :walkerid),
        ))

        @test initval_alg.ncalls == 3
        @test chain_state.current.x.v[rerolled_idx] != [10.5]
        @test chain_state.current.x.v[rerolled_idx] ==
            chain_state.current.z.v[rerolled_idx]
        @test chain_state.current.x.logd[rerolled_idx] ==
            BAT.checked_logdensityof(target, chain_state.current.x.v[rerolled_idx])
        @test chain_state.current.z.logd[rerolled_idx] ==
            chain_state.current.x.logd[rerolled_idx]

        mala_cache = chain_state.proposal.proposal_states[1].grad_cache
        @test isempty(mala_cache.grads_curr)
        @test isempty(mala_cache.grads_prop)

        # Mark MALA active without going through proposal re-entry, which would
        # itself invalidate the cache and mask a missing retry invalidation.
        multi_proposal = chain_state.proposal
        chain_state.proposal = BAT.MultiProposalState(
            multi_proposal.proposal_states,
            multi_proposal.picking_rule,
            1,
        )
        candidate = deepcopy(state)
        oracle = deepcopy(state)
        oracle_cache = oracle.chain_state.proposal.proposal_states[1].grad_cache
        empty!(oracle_cache.grads_curr)
        empty!(oracle_cache.grads_prop)
        candidate = BAT.mcmc_step!!(candidate)
        oracle = BAT.mcmc_step!!(oracle)

        @test candidate.chain_state.proposal.active_idx == 1
        @test oracle.chain_state.proposal.active_idx == 1
        @test candidate.chain_state.proposed.x.v == oracle.chain_state.proposed.x.v
        @test candidate.chain_state.proposed.z.v == oracle.chain_state.proposed.z.v
        @test candidate.chain_state.proposed.x.logd == oracle.chain_state.proposed.x.logd
        @test candidate.chain_state.proposed.z.logd == oracle.chain_state.proposed.z.logd
        @test candidate.chain_state.accepted == oracle.chain_state.accepted
        @test candidate.chain_state.current.x == oracle.chain_state.current.x
        @test candidate.chain_state.current.z == oracle.chain_state.current.z
    end
end
