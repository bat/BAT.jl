# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions
using Logging
using ValueShapes


mutable struct RetryInitSequence <: BAT.InitvalAlgorithm
    values::Vector{Vector{Float64}}
    next::Int
end

RetryInitSequence(values) = RetryInitSequence([[value] for value in values], 1)

function BAT.bat_initval_impl(target, algorithm::RetryInitSequence, context::BATContext)
    result = algorithm.values[algorithm.next]
    algorithm.next += 1
    return (result = copy(result),)
end

struct ZeroMCMCWeighting <: BAT.AbstractMCMCWeightingScheme{Int} end

BAT.mcmc_weight_type(::ZeroMCMCWeighting) = Int

function BAT.mcmc_weight_values(::ZeroMCMCWeighting, p_accept, accepted)
    return (zeros(Int, length(p_accept)), zeros(Int, length(p_accept)))
end

function retry_init_result(initial_values, rerolled_values)
    target = unshaped(batmeasure(Normal()))
    initval_alg = RetryInitSequence([initial_values; rerolled_values])
    samplingalg = TransformedMCMC(
        adaptive_transform = BAT.NoAdaptiveTransform(),
        nchains = 1,
        nwalkers = length(initial_values),
        nonzero_weights = true,
        sample_weighting = ZeroMCMCWeighting(),
    )
    init_alg = MCMCRetryInit(
        max_init_tries = 1,
        nsteps_init = 1,
        initval_alg = initval_alg,
        strict = false,
    )

    result = nothing
    @test_logs min_level=Logging.Debug match_mode=:any (:debug, r"Rerolling starting positions for \d+ walkers") begin
        result = BAT.mcmc_init!(samplingalg, target, init_alg, (_...) -> nothing, BATContext())
    end

    return (; result..., target)
end

function test_reset_walker(state, walkerid, rerolled_value)
    chain_state = state.chain_state
    expected_v = [rerolled_value]
    expected_logd = logpdf(Normal(), rerolled_value)

    @test chain_state.current.x.v[walkerid] == expected_v
    @test chain_state.current.z.v[walkerid] == expected_v
    @test chain_state.proposed.x.v[walkerid] == expected_v
    @test chain_state.proposed.z.v[walkerid] == expected_v
    @test chain_state.output.v[walkerid] == expected_v

    @test chain_state.current.x.logd[walkerid] == expected_logd
    @test chain_state.current.z.logd[walkerid] == expected_logd
    @test chain_state.proposed.x.logd[walkerid] == expected_logd
    @test chain_state.proposed.z.logd[walkerid] == expected_logd
    @test chain_state.output.logd[walkerid] == expected_logd

    @test iszero(chain_state.current.x.weight[walkerid])
    @test iszero(chain_state.current.z.weight[walkerid])
    @test iszero(chain_state.proposed.x.weight[walkerid])
    @test iszero(chain_state.proposed.z.weight[walkerid])
    @test iszero(chain_state.output.weight[walkerid])
    @test isnothing(chain_state.current.x.aux[walkerid])
    @test isnothing(chain_state.current.z.aux[walkerid])
    @test isnothing(chain_state.proposed.x.aux[walkerid])
    @test isnothing(chain_state.proposed.z.aux[walkerid])
    @test isnothing(chain_state.output.aux[walkerid])
    @test !chain_state.accepted[walkerid]

    current_info = chain_state.current.x.info[walkerid]
    proposed_info = chain_state.proposed.x.info[walkerid]
    @test current_info == chain_state.current.z.info[walkerid]
    @test current_info == chain_state.output.info[walkerid]
    @test proposed_info == chain_state.proposed.z.info[walkerid]
    @test current_info.walkerid == walkerid
    @test proposed_info.walkerid == walkerid
    @test current_info.chaincycle == chain_state.info.cycle
    @test proposed_info.chaincycle == chain_state.info.cycle
    @test current_info.stepno == 0
    @test proposed_info.stepno == 0
    @test current_info.sampletype
    @test !proposed_info.sampletype
end

@testset "MCMCRetryInit" begin
    @testset "empty output rerolls one walker" begin
        result = retry_init_result([-8.0], [0.25])
        @test all(isempty, only(result.outputs))
        test_reset_walker(only(result.mcmc_states), 1, 0.25)
    end

    @testset "empty output rerolls and counts multiple walkers" begin
        initial_values = [-8.0, 8.0]
        rerolled_values = [-0.5, 0.5]
        target = unshaped(batmeasure(Normal()))
        initval_alg = RetryInitSequence([initial_values; rerolled_values])
        samplingalg = TransformedMCMC(
            adaptive_transform = BAT.NoAdaptiveTransform(),
            nchains = 1,
            nwalkers = 2,
            nonzero_weights = true,
            sample_weighting = ZeroMCMCWeighting(),
        )
        init_alg = MCMCRetryInit(
            max_init_tries = 1,
            nsteps_init = 1,
            initval_alg = initval_alg,
            strict = false,
        )

        result = nothing
        @test_logs min_level=Logging.Debug match_mode=:any (:debug, "Rerolling starting positions for 2 walkers in chain 1.") begin
            result = BAT.mcmc_init!(samplingalg, target, init_alg, (_...) -> nothing, BATContext())
        end

        @test all(isempty, only(result.outputs))
        for walkerid in eachindex(rerolled_values)
            test_reset_walker(only(result.mcmc_states), walkerid, rerolled_values[walkerid])
        end
    end


    @testset "reset only changes selected walkers" begin
        target = unshaped(batmeasure(Normal()))
        samplingalg = TransformedMCMC(
            adaptive_transform = BAT.NoAdaptiveTransform(),
            nchains = 1,
            nwalkers = 2,
        )
        state = MCMCState(samplingalg, target, 1, [[-0.25], [8.0]], BATContext())
        BAT.next_cycle!(state)
        state.chain_state.accepted .= true
        first_walker_before = deepcopy((
            current_x = state.chain_state.current.x[1],
            current_z = state.chain_state.current.z[1],
            proposed_x = state.chain_state.proposed.x[1],
            proposed_z = state.chain_state.proposed.z[1],
            output = state.chain_state.output[1],
            accepted = state.chain_state.accepted[1],
        ))

        state = BAT._reset_mcmc_walkers!!(state, [2], [[0.5]])

        first_walker_after = (
            current_x = state.chain_state.current.x[1],
            current_z = state.chain_state.current.z[1],
            proposed_x = state.chain_state.proposed.x[1],
            proposed_z = state.chain_state.proposed.z[1],
            output = state.chain_state.output[1],
            accepted = state.chain_state.accepted[1],
        )
        @test first_walker_after == first_walker_before
        test_reset_walker(state, 2, 0.5)
    end
end
