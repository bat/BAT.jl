# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using ArraysOfArrays, BAT, DensityInterface, Distributions, Random123, Test, ValueShapes
import EnsembleMCMC

@testset "shaped posterior" begin
    prior = NamedTupleDist(a=Uniform(-3.0, 4.0), b=LogNormal(0.2, 0.4))
    posterior = PosteriorMeasure(logfuncdensity(x -> -0.5 * (x.a - log(x.b))^2), prior)
    algorithm = TransformedMCMC(
        proposal=EnsembleProposal(EnsembleMCMC.StretchMove()), nwalkers=8,
        nchains=2, nsteps=64, convergence=AssumeConvergence(),
        init=MCMCRetryInit(nsteps_init=30), burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=30, max_ncycles=1),
    )
    em = evalmeasure(posterior, algorithm, BATContext(rng=Philox4x((573, 2026))))
    @test BAT.validate_evalmeasure(em) === em
    @test logdensityof(posterior).(samplesof(em).v) ≈ samplesof(em).logd
end

@testset "Mixture history and diagnostics" begin
    moves = EnsembleMCMC.MoveMixture((EnsembleMCMC.StretchMove(), EnsembleMCMC.DEMove()),
        [1, 1]; schedule=:cycle)
    mixed(executor) = MCMCMultiProposal(
        proposals=BAT.MCMCProposal[EnsembleProposal(moves; executor), RandomWalk()],
        picking_rule=[1, 1])
    target = batmeasure(MvNormal(zeros(2), 1.0))
    initial = [[cospi(k / 4), sinpi(k / 4)] for k in 0:7]
    states = map([EnsembleMCMC.SerialExecutor(), EnsembleMCMC.ThreadedExecutor()]) do executor
        algorithm = TransformedMCMC(proposal=mixed(executor), nwalkers=8,
            proposal_tuning=NoMCMCProposalTuning(), convergence=AssumeConvergence())
        BAT.MCMCState(algorithm, target, 1, initial, BATContext(rng=Philox4x((573, 401))))
    end
    histories = [BAT._empty_chain_outputs(state) for state in states]
    for cycle in 1:2
        cycle > 1 && foreach(BAT.next_cycle!, states)
        for i in eachindex(states)
            states[i] = BAT.mcmc_iterate!!(histories[i], states[i]; max_nsteps=8, nonzero_weights=false)
        end
    end
    samples = [BAT._merge_chain_outputs(state, [output]) for (state, output) in zip(states, histories)]
    @test first(samples) == last(samples)
    cs = first(states).chain_state
    component = first(BAT._proposal_diagnostics(cs.proposal, cs).components)
    @test component.cycle_n_attempts == 32
    @test component.diagnostics.cumulative_move_attempts == [32, 32]
    @test any(iszero, first(samples).weight) && sum(first(samples).weight) == 128
    @test first(samples).logd ≈ logdensityof(target).(first(samples).v)
end

@testset "Initial density cache is shared across ensemble proposals" begin
    calls = Ref(0)
    target = batmeasure(PosteriorMeasure(
        logfuncdensity(x -> (calls[] += 1; -sum(abs2, x) / 2)), MvNormal(zeros(2), 1.0)))
    initial = [[cospi(k / 4), sinpi(k / 4)] for k in 0:7]
    for proposal in (EnsembleProposal(EnsembleMCMC.StretchMove()),
        MCMCMultiProposal(proposals=BAT.MCMCProposal[
            EnsembleProposal(EnsembleMCMC.StretchMove()), EnsembleProposal(EnsembleMCMC.DEMove())],
            picking_rule=[1, 1]))
        algorithm = TransformedMCMC(; proposal, nwalkers=8, pretransform=DoNotTransform(),
            proposal_tuning=NoMCMCProposalTuning(), convergence=AssumeConvergence())
        calls[] = 0
        state = BAT.MCMCState(algorithm, target, 1, initial, BATContext(rng=Philox4x((573, 91))))
        @test calls[] == length(initial)
        components = proposal isa MCMCMultiProposal ? state.chain_state.proposal.proposal_states :
            (state.chain_state.proposal,)
        @test all(p -> EnsembleMCMC.current_state(p.ensemble).logdensities == state.chain_state.current.z.logd,
            components)
    end
end

@testset "CPU batches preserve transformed posterior sampling" begin
    batches = Ref(0)
    function batch!(values, target, positions)
        batches[] += 1
        values .= BAT.checked_logdensityof.(Ref(target), eachcol(positions))
        return nothing
    end
    prior = NamedTupleDist(a=Uniform(-3.0, 4.0), b=LogNormal(0.2, 0.4))
    posterior = PosteriorMeasure(logfuncdensity(x -> -0.5 * (x.a - log(x.b))^2), prior)
    results = map((nothing, batch!)) do batch
        algorithm = TransformedMCMC(
            proposal=EnsembleProposal(EnsembleMCMC.StretchMove();
                batch_logdensity! = batch, record_transitions=!isnothing(batch)),
            nwalkers=8, nchains=1, nsteps=16, convergence=AssumeConvergence(),
            init=MCMCRetryInit(nsteps_init=10),
            burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=10, max_ncycles=1))
        evalmeasure(posterior, algorithm, BATContext(rng=Philox4x((573, 92))))
    end
    @test batches[] > 0
    @test samplesof(first(results)) == samplesof(last(results))
    @test logdensityof(posterior).(samplesof(last(results)).v) ≈ samplesof(last(results)).logd
    diagnostics = only(evalinfo(last(results)).result.chain_diagnostics)
    @test length(diagnostics.transitions) > 16
    @test length(unique(r.cycle for r in diagnostics.transitions)) > 1
    @test sum(diagnostics.cumulative_move_attempts) == 8 * length(diagnostics.transitions)
end

@testset "Owned inner transition records retain rejected sweeps" begin
    moves = EnsembleMCMC.MoveMixture((EnsembleMCMC.StretchMove(), EnsembleMCMC.DEMove()),
        [1, 1]; schedule=:cycle)
    target = batmeasure(MvNormal(zeros(2), 1.0))
    initial = [[cospi(k / 4), sinpi(k / 4)] for k in 0:7]
    states = map([false, true]) do record_transitions
        proposal = MCMCMultiProposal(proposals=BAT.MCMCProposal[
            EnsembleProposal(moves; record_transitions), RandomWalk()], picking_rule=[1, 1])
        algorithm = TransformedMCMC(; proposal, nwalkers=8,
            proposal_tuning=NoMCMCProposalTuning(), convergence=AssumeConvergence())
        BAT.MCMCState(algorithm, target, 1, initial, BATContext(rng=Philox4x((573, 401))))
    end
    histories = [BAT._empty_chain_outputs(state) for state in states]
    for cycle in 1:2
        cycle > 1 && foreach(BAT.next_cycle!, states)
        for i in eachindex(states)
            states[i] = BAT.mcmc_iterate!!(histories[i], states[i]; max_nsteps=8, nonzero_weights=false)
        end
    end
    samples = [BAT._merge_chain_outputs(state, [output]) for (state, output) in zip(states, histories)]
    @test first(samples) == last(samples)
    plain = first(states).chain_state.proposal.proposal_states[1]
    recorded = last(states).chain_state.proposal.proposal_states[1]
    @test !hasproperty(BAT._proposal_diagnostics(plain), :transitions)
    diagnostics = BAT._proposal_diagnostics(recorded)
    @test diagnostics.walker_ids == collect(1:8)
    @test [(r.cycle, r.step, r.move_index) for r in diagnostics.transitions] ==
        [(cycle, step, mod1(i, 2)) for cycle in 0:1 for (i, step) in enumerate(1:2:7)]
    @test any(r -> !all(r.accepted), diagnostics.transitions)
    @test [sum(count(r.accepted) for r in diagnostics.transitions if r.move_index == i) for i in 1:2] ==
        diagnostics.cumulative_move_acceptances
    latest = last(diagnostics.transitions)
    transition = EnsembleMCMC.current_state(recorded.ensemble)
    @test latest.accepted == transition.accepted
    @test latest.acceptance_probabilities == transition.acceptance_probabilities
    saved = deepcopy(diagnostics)
    BAT.next_cycle!(last(states))
    states[2] = BAT.mcmc_iterate!!(histories[2], states[2]; max_nsteps=2, nonzero_weights=false)
    @test diagnostics == saved
    latest.accepted .= false
    latest.acceptance_probabilities .= -1
    diagnostics.walker_ids .= 0
    @test BAT._proposal_diagnostics(recorded).transitions[8] == last(saved.transitions)
    @test BAT._proposal_diagnostics(recorded).walker_ids == collect(1:8)
end
mutable struct EnsembleRetryFixture <: InitvalAlgorithm
    calls::Int
end
function BAT.bat_initval_impl(::BAT.MeasureLike, init::EnsembleRetryFixture, ::BATContext)
    init.calls += 1
    return (result=[init.calls <= 4 ? 10 + init.calls / 10 : (init.calls - 4) / 5],)
end

@testset "Retry initialization synchronizes the ensemble" begin
    initval = EnsembleRetryFixture(0)
    algorithm = TransformedMCMC(
        proposal=EnsembleProposal(EnsembleMCMC.DEMove(gamma0=1e-12, sigma=0)),
        pretransform=DoNotTransform(),
        init=MCMCRetryInit(max_init_tries=2, nsteps_init=1, initval_alg=initval),
        burnin=MCMCMultiCycleBurnin(max_ncycles=0, nsteps_final=0),
        convergence=AssumeConvergence(), nchains=1, nwalkers=4, nsteps=2,
        nonzero_weights=false, strict=false)
    samples = bat_sample(Uniform(0, 1), algorithm, BATContext(rng=Philox4x((573, 83)))).result
    @test all(isfinite, samples.logd)
    @test initval.calls == 8
end

function ensemble_walker_output(path, walkerid)
    info = [
        BAT.MCMCSampleID(Int32(1), Int32(walkerid), Int32(1), Int64(step), Int32(1), true)
        for step in axes(path, 2)
    ]
    return DensitySampleVector(
        v = VectorOfSimilarVectors(path), logd = zeros(eltype(path), size(path, 2)), info = info,
    )
end

@testset "coupled ensemble ESS" begin
    paths = [
        reshape([1.0, -1, 1, 1, -1, -1, 1, -1], 1, :),
        reshape([0.0, 1, -1, 1, 1, -1, -1, 1], 1, :),
    ]
    outputs = [[ensemble_walker_output(path, walkerid)
                for (walkerid, path) in pairs(paths)]]
    expected = 2 * only(bat_eff_sample_size(
        VectorOfSimilarVectors((paths[1] + paths[2]) / 2),
        EffSampleSizeFromAC(),
        BATContext(),
    ).result)

    @test BAT._mcmc_ess(
        outputs, reduce(vcat, only(outputs)), EnsembleProposal(EnsembleMCMC.StretchMove()),
        RepetitionWeighting(), false, BATContext(),
    ) ≈ expected
end

@testset "coupled ensemble ESS avoids Float32 overflow" begin
    paths = [fill(2.0f38, 1, 8), fill(2.5f38, 1, 8)]
    outputs = [[ensemble_walker_output(path, walkerid)
                for (walkerid, path) in pairs(paths)]]

    @test BAT._mcmc_ess(
        outputs, reduce(vcat, only(outputs)), EnsembleProposal(EnsembleMCMC.StretchMove()),
        RepetitionWeighting(), false, BATContext(),
    ) == 16
end
