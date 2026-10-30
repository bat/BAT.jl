# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestMCMCInitialState

using BAT, Test, Random, Random123, LinearAlgebra, Distributions, ValueShapes, ArraysOfArrays
using AffineMaps: MulAdd
import ForwardDiff

const CHAIN_ID = 7
const POSITIONS = [0.2 -0.8 1.4; -0.4 0.6 -1.2]
const DISTRIBUTION = MvNormal([0.5, -0.7], [1.3 0.2; 0.2 0.8])
const AFFINE = MulAdd(Diagonal([2.0, 3.0]), [1.0, -2.0])
const PROPOSALS = (RandomWalk(proposaldist=Normal()), HamiltonianMC(step_size=0.2, max_depth=2))

new_context(seed=1819) = BATContext(rng=Philox4x((seed, 564)), ad=ForwardDiff)

function fixed_algorithm(proposal, nwalkers, affine)
    return TransformedMCMC(proposal=proposal, nchains=1, nwalkers=nwalkers,
        adaptive_transform=affine ? BAT.CustomTransform(AFFINE) : BAT.NoAdaptiveTransform(),
        proposal_tuning=NoMCMCProposalTuning(), transform_tuning=NoMCMCTransformTuning(),
        pretransform=DoNotTransform(), nonzero_weights=false)
end

buffers(chain) = (chain.current.x, chain.current.z, chain.proposed.x, chain.proposed.z, chain.output)

function check_spaces(chain, distribution)
    f = chain.f_transform
    # An analytic affine inverse and determinant, independently of BAT's inverse
    # transform/pullback implementation. The explicit transform has det(A)=6.
    A, b, ladj = if f === identity
        (Matrix{Float64}(I, length(first(chain.current.x.v)), length(first(chain.current.x.v))),
            zeros(length(first(chain.current.x.v))), 0.0)
    else
        @test f isa MulAdd
        matrix = Matrix(f.A)
        (matrix, Vector(f.b), first(logabsdet(matrix)))
    end
    for pair in (chain.current, chain.proposed), i in eachindex(pair.x)
        x, z = Vector(pair.x.v[i]), Vector(pair.z.v[i])
        @test z ≈ A \ (x - b) rtol=1e-12 atol=1e-12
        @test x ≈ A * z + b rtol=1e-12 atol=1e-12
        expected_logd = logpdf(distribution, x)
        @test pair.x.logd[i] ≈ expected_logd rtol=1e-12 atol=1e-12
        @test pair.z.logd[i] ≈ expected_logd + ladj rtol=1e-12 atol=1e-12
        @test pair.x.weight[i] == pair.z.weight[i]
        @test pair.x.info[i] == pair.z.info[i]
    end
    @test typeof(chain.current.x) === typeof(chain.proposed.x)
    @test typeof(chain.current.z) === typeof(chain.proposed.z)
    @test typeof(chain.output) === typeof(chain.current.x)
end

function check_initial(chain, distribution, initial_positions)
    check_spaces(chain, distribution)
    n = length(initial_positions)
    @test chain.current.x.v == initial_positions
    @test chain.proposed.x == chain.current.x
    @test chain.proposed.z == chain.current.z
    @test chain.output == chain.current.x
    @test chain.accepted == falses(n)
    @test chain.walker_order == collect(1:n)
    @test chain.stepno == 0 && chain.info.cycle == 0
    @test all(iszero, chain.nattempts) && all(iszero, chain.nsamples)
    for samples in buffers(chain)
        @test all(iszero, samples.weight)
        @test all(isnothing, samples.aux)
        @test samples.info == [BAT.MCMCSampleID(Int32(CHAIN_ID), Int32(i), Int32(1),
            Int64(0), Int32(1), true) for i in 1:n]
    end
end

function check_writable_isolation!(chain)
    all_buffers = buffers(chain)
    snapshots = map(deepcopy, all_buffers)
    # Do not assert caller/current detachment: current can intentionally retain
    # canonical input storage. Proposed x/z and output must be independent of
    # every other buffer, down to writable coordinates and component arrays.
    for index in (3, 4, 5)
        destination = all_buffers[index]
        before = snapshots[index]
        others = [j for j in eachindex(all_buffers) if j != index]
        for field in (:v, :logd, :weight, :info, :aux), j in others
            @test getproperty(destination, field) !== getproperty(all_buffers[j], field)
        end
        for field in (:v, :logd, :weight, :info)
            if field === :v
                destination.v[1][1] += 7
            elseif field === :logd
                destination.logd[1] += 7
            elseif field === :weight
                destination.weight[1] += 1
            else
                old = destination.info[1]
                destination.info[1] = BAT.MCMCSampleID(old.chainid, old.walkerid,
                    old.chaincycle, old.stepno + 1, old.proposalid, !old.sampletype)
            end
            for j in others
                @test all_buffers[j] == snapshots[j]
            end
            @test destination[2:end] == before[2:end]
            if field === :v
                destination.v[1][1] = before.v[1][1]
            else
                getproperty(destination, field)[1] = getproperty(before, field)[1]
            end
            @test destination == before
        end
    end
end

function short_run(state, distribution; nsteps=4)
    outputs = BAT._empty_chain_outputs(state)
    @test length(outputs) == BAT.nwalkers(state)
    @test all(i == j || outputs[i] !== outputs[j] for i in eachindex(outputs), j in eachindex(outputs))
    first_step = state.chain_state.stepno
    state = BAT.mcmc_iterate!!(outputs, state; max_nsteps=nsteps, nonzero_weights=false)
    @test state.chain_state.stepno == first_step + nsteps
    @test sum(sum(output.weight) for output in outputs) == nsteps * length(outputs)
    check_spaces(state.chain_state, distribution)
    check_writable_isolation!(state.chain_state)
    for (i, output) in enumerate(outputs)
        @test !isempty(output)
        @test all(info -> info.walkerid == i && info.chainid == CHAIN_ID, output.info)
        @test output.logd ≈ [logpdf(distribution, Vector(x)) for x in output.v]
    end
    return state, outputs
end

function rng_continuation(state)
    chain = state.chain_state
    return (rand(copy(BAT.get_rng(chain.context)), UInt64, 8),
        [rand(copy(BAT.get_rng(context)), UInt64, 8) for context in chain.walker_genctxs])
end

function nested_positions(data)
    return [collect(column) for column in eachcol(data)]
end

function view_positions(data)
    d, n = size(data)
    parent = fill(-99.0, d+2, n+2)
    parent[2:d+1, 2:n+1] .= data
    return VectorOfSimilarVectors(view(parent, 2:d+1, 2:n+1))
end

function reshaped_positions(data)
    storage = vcat(-99.0, vec(data), -99.0)
    # Reshaping a SubArray keeps a distinct view-backed representation instead
    # of relying on Array reshape's implementation-dependent concrete type.
    return VectorOfSimilarVectors(reshape(view(storage, 2:length(storage)-1), size(data)))
end

@testset "MCMC corresponding initial sample storage" begin
    @testset "Real unshaped NamedTupleDist initialization, MH/HMC and walkers" begin
        objective = NamedTupleDist(a=Normal(1, 1.5),
            b=MvNormal([-1.0, 2.0], [2.0 1.5; 1.5 3.0]))
        target, distribution = unshaped(batmeasure(objective)), unshaped(objective)
        for n in (1, 3), proposal in PROPOSALS
            @testset "$(nameof(typeof(proposal))) with $n walkers" begin
                context = new_context(1800 + n)
                initial = BAT.bat_ensemble_initvals(target, InitFromTarget(), n, context)
                positions = unshaped.(initial)
                expected = nested_positions(hcat(positions...))
                # The original HMC regression uses StanLikeTuning; MH uses RAM.
                tuning = proposal isa HamiltonianMC ? BAT.StanLikeTuning() : RAMTuning()
                algorithm = TransformedMCMC(proposal=proposal, nchains=1, nwalkers=n,
                    adaptive_transform=BAT.TriangularAffineTransform(), transform_tuning=tuning,
                    pretransform=DoNotTransform(), nonzero_weights=false)
                state = BAT.MCMCState(algorithm, target, CHAIN_ID, positions, context)
                check_initial(state.chain_state, distribution, expected)
                check_writable_isolation!(state.chain_state)
                # Exercise actual pre-proposal tuning hooks, not only the bare
                # chain constructor. None should need incorrect x-space z data.
                BAT.mcmc_tuning_init!!(state, 4)
                BAT.mcmc_tuning_reinit!!(state, 4)
                check_spaces(state.chain_state, distribution)
                short_run(state, distribution)
            end
        end
    end

    @testset "Representations preserve corresponding buffers and fixed-work results" begin
        representations = (
            :nested => nested_positions,
            :matrix => (data -> VectorOfSimilarVectors(copy(data))),
            :view => view_positions,
            :reshaped => reshaped_positions,
            :eachcol => (data -> eachcol(copy(data))),
        )
        for (r, (name, represent)) in enumerate(representations), (p, proposal) in enumerate(PROPOSALS)
            # Both proposals and both transforms occur for every representation,
            # without a redundant full representation × proposal × transform grid.
            affine = isodd(r + p)
            @testset "$name $(nameof(typeof(proposal))) affine=$affine" begin
                algorithm = fixed_algorithm(proposal, size(POSITIONS, 2), affine)
                target = batmeasure(DISTRIBUTION)
                state = BAT.MCMCState(algorithm, target, CHAIN_ID, represent(POSITIONS), new_context())
                reference = BAT.MCMCState(algorithm, target, CHAIN_ID, nested_positions(POSITIONS), new_context())
                check_initial(state.chain_state, DISTRIBUTION, nested_positions(POSITIONS))
                check_writable_isolation!(state.chain_state)
                @test rng_continuation(state) == rng_continuation(reference)
                state, outputs = short_run(state, DISTRIBUTION)
                reference, expected_outputs = short_run(reference, DISTRIBUTION)
                @test outputs == expected_outputs
                @test buffers(state.chain_state) == buffers(reference.chain_state)
                @test state.chain_state.accepted == reference.chain_state.accepted
                @test state.chain_state.nattempts == reference.chain_state.nattempts
                @test state.chain_state.nsamples == reference.chain_state.nsamples
                @test rng_continuation(state) == rng_continuation(reference)
            end
        end
    end

    @testset "Remapping transformed storage remains writable and type-compatible" begin
        for proposal in PROPOSALS
            algorithm = fixed_algorithm(proposal, size(POSITIONS, 2), true)
            state = BAT.MCMCState(algorithm, batmeasure(DISTRIBUTION), CHAIN_ID,
                reshaped_positions(POSITIONS), new_context(1818))
            state, _ = short_run(state, DISTRIBUTION)
            old_type = typeof(state)
            x_before = deepcopy((state.chain_state.current.x, state.chain_state.proposed.x))
            z_types = (typeof(state.chain_state.current.z), typeof(state.chain_state.proposed.z))
            state.chain_state.f_transform = MulAdd(Diagonal([1.5, 0.75]), [-0.25, 0.5])
            state.chain_state.proposal = BAT.set_proposal_transform!!(state.chain_state.proposal, state.chain_state)
            state = BAT.mcmc_update_z_position!!(state)
            @test typeof(state) === old_type
            @test (state.chain_state.current.x, state.chain_state.proposed.x) == x_before
            @test typeof(state.chain_state.current.z) === z_types[1]
            @test typeof(state.chain_state.proposed.z) === z_types[2]
            check_spaces(state.chain_state, DISTRIBUTION)
            check_writable_isolation!(state.chain_state)
            short_run(state, DISTRIBUTION; nsteps=2)
        end
    end
end

end
