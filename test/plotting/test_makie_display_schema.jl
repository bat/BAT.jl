# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieDisplaySchema
using BAT, Test, Makie, Distributions, ValueShapes, InverseFunctions, Random, Statistics

const EXT = Base.get_extension(BAT, :BATMakieExt)
const SEQUENTIAL = BAT.SequentialExec()

sample_id(chain, walker, cycle, step) = BAT.MCMCSampleID(
    Int32(chain), Int32(walker), Int32(cycle), Int64(step), Int32(1), true)
span(row, cycle, first, last) = EXT.SampleStepSpan(row, Int32(cycle), first, last)
empty_time(outputs) = [EXT._empty_walker_spans(chain) for chain in outputs]
fields(samples) = map(k -> deepcopy(getproperty(samples, k)), (:v, :logd, :weight, :info, :aux))

function fixture(target, intent, original_values; chain=1)
    context = BATContext(rng=BAT.Philox4x((chain, 217)))
    transformed, f = BAT.transform_and_unshape(intent, target, context)
    points = [collect(f(x)) for x in original_values]
    alg = TransformedMCMC(nchains=1, nwalkers=length(points), pretransform=intent,
        adaptive_transform=BAT.NoAdaptiveTransform(), transform_tuning=NoMCMCTransformTuning())
    state = BAT.MCMCState(alg, transformed, Int32(chain), points, context)
    return state, f
end

function state_snapshot(state)
    chain = state.chain_state
    return (fields(chain.current.x), fields(chain.current.z), fields(chain.output),
        copy(BAT.get_rng(chain.context)), [copy(BAT.get_rng(c)) for c in chain.walker_genctxs])
end

canonical(f, samples, shape) = EXT._canonical_display_samples(
    EXT._inverse_display_samples(f, samples, SEQUENTIAL), shape)

# An undeclared direct-caller target must not acquire a varshape requirement.
struct UndeclaredTarget end
struct MissingShapeMeasure <: BAT.BATMeasure end

to_sampler_precision(x) = Float32.(x)
to_display_precision(x) = Float64.(x)
InverseFunctions.inverse(::typeof(to_sampler_precision)) = to_display_precision
InverseFunctions.inverse(::typeof(to_display_precision)) = to_sampler_precision

struct RecordingInverse <: Function
    seen::Vector{Float64}
end
function (f::RecordingInverse)(x)
    push!(f.seen, first(x))
    first(x) == 99 && error("unexpected current-state prototype")
    return copy(x)
end
struct RecordingForward <: Function
    backward::RecordingInverse
end
InverseFunctions.inverse(f::RecordingForward) = f.backward

@testset "Original-coordinate live display schema" begin
    @testset "Declared shape and empty bootstrap" begin
        simplex = Dirichlet([2.0, 3.0, 4.0])
        nested = NamedTupleDist(a=Normal(), sub=NamedTupleDist(p=simplex, c=[4.2, 3.7]))
        cases = (
            ("simplex", simplex, NormalBased(), [[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]],
                [[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]], 2),
            ("nested constants", nested, NormalBased(),
                [(a=1.25, sub=(p=[0.2, 0.3, 0.5], c=[4.2, 3.7])),
                 (a=-0.5, sub=(p=[0.1, 0.6, 0.3], c=[4.2, 3.7]))],
                [[1.25, 0.2, 0.3, 0.5], [-0.5, 0.1, 0.6, 0.3]], 3),
            ("scalar", Normal(10.0, 2.0), NormalBased(), [9.0, 12.0], [[9.0], [12.0]], 1),
            ("shape only", nested, DoNotTransform(),
                [(a=1.25, sub=(p=[0.2, 0.3, 0.5], c=[4.2, 3.7])),
                 (a=-0.5, sub=(p=[0.1, 0.6, 0.3], c=[4.2, 3.7]))],
                [[1.25, 0.2, 0.3, 0.5], [-0.5, 0.1, 0.6, 0.3]], 4),
            ("identity", MvNormal(zeros(2), ones(2)), DoNotTransform(),
                [[1.0, 2.0], [-1.0, 0.5]], [[1.0, 2.0], [-1.0, 0.5]], 2),
        )
        for (name, target, intent, values, expected, sampler_dim) in cases
            @testset "$name" begin
                state, f = fixture(target, intent, values)
                outputs = [BAT._empty_chain_outputs(state)]
                shape = EXT._live_display_shape(target, f, state)
                before = state_snapshot(state)
                task_rng = copy(Random.default_rng())
                @test shape == varshape(target)
                @test totalndof(varshape(BAT.current_sample(state.chain_state))) == sampler_dim
                @test all(iszero, state.chain_state.current.x.weight)
                history, resolved = EXT._initial_display_outputs([state], outputs, f, shape)
                @test resolved == shape
                @test all(isempty, only(history))
                @test all(s -> totalndof(varshape(s)) == length(first(expected)), only(history))
                @test all(s -> all(isempty, fields(s)), only(history))
                @test all(isempty, only(empty_time(history)))
                @test isequal(state_snapshot(state), before)
                @test Random.default_rng() == task_rng

                displayed = canonical(f, deepcopy(BAT.current_sample(state.chain_state)), shape)
                @test all(isapprox(a, b; rtol=1e-10, atol=1e-12) for (a, b) in zip(displayed.v, expected))
                @test displayed.logd ≈ logpdf.(Ref(target), values)
                @test displayed.weight == [0, 0]
                @test displayed.info == state.chain_state.current.x.info
                @test displayed.aux == state.chain_state.current.x.aux
                push!(history[1][1], displayed[1])
                @test isempty(history[1][2])
                @test isempty(outputs[1][1])
                history[1][1].v[1][1] = -123.0
                history[1][1].weight[1] = 7
                @test isequal(state_snapshot(state), before)
                @test displayed.v[1] ≈ expected[1]

                # A valid coordinate prototype does not require finite density.
                state.chain_state.current.x.logd .= -Inf
                nonfinite_before = state_snapshot(state)
                empty_history, _ = EXT._initial_display_outputs([state], outputs, f, shape)
                @test all(isempty, only(empty_history))
                @test isequal(state_snapshot(state), nonfinite_before)

                fresh_empty = EXT._transform_display_outputs(f, outputs, empty_history, shape; executor=SEQUENTIAL)
                @test all(s -> totalndof(varshape(s)) == length(first(expected)), only(fresh_empty))
                push!(fresh_empty[1][2], displayed[2])
                @test isempty(fresh_empty[1][1]) && all(isempty, only(empty_history))
            end
        end
    end

    @testset "Shape metadata fallbacks and explicit nested rows" begin
        target = NamedTupleDist(a=Normal(), sub=NamedTupleDist(p=Dirichlet([2.0, 3.0, 4.0]), c=4.2))
        value = (a=1.25, sub=(p=[0.2, 0.3, 0.5], c=4.2))
        state, f = fixture(target, DoNotTransform(), [value])
        shape = varshape(target)
        shape_forward = inverse(shape)
        @test EXT._live_display_shape(nothing, shape_forward, state) == shape
        @test EXT._live_display_shape(UndeclaredTarget(), shape_forward, state) == shape
        @test EXT._live_display_shape(MissingShapeMeasure(), shape_forward, state) == shape

        raw = DensitySampleVector(([value], [-2.0], Int32[0], [:source], [nothing]))
        flat = EXT._canonical_display_samples(raw, shape)
        @test only(flat.v) == [1.25, 0.2, 0.3, 0.5]
        @test (flat.logd, flat.weight, flat.info, flat.aux) == ([-2.0], Int32[0], [:source], [nothing])

        # Nested raw tuples need not implement container-level varshape. Their
        # original target already supplies the complete domain schema.
        empirical = DensitySampleVector(([value], [-2.0], [1], [:source], [nothing]))
        evaluated = EvaluatedMeasure(target; empirical)
        @test EXT._live_display_shape(evaluated, f, state) == shape
        lo, hi = EXT._estimate_prior_domain(BAT.Philox4x((83, 2)), evaluated, 4;
            n_prior_samples=4, display_shape=shape)
        @test lo == hi == [1.25, 0.2, 0.3, 0.5]

        ordinary, _ = fixture(MvNormal(zeros(2), ones(2)), DoNotTransform(), [[1.0, 2.0]])
        @test ismissing(EXT._live_display_shape(UndeclaredTarget(), to_sampler_precision, ordinary))
        inferred, inferred_shape = EXT._initial_display_outputs([ordinary],
            [BAT._empty_chain_outputs(ordinary)], to_sampler_precision, missing)
        @test totalndof(inferred_shape) == 2
        @test isempty(only(only(inferred)))
        @test totalndof(varshape(only(only(inferred)))) == 2
    end

    @testset "Composed uniform pushforward preserves declared constants" begin
        source = BAT.StandardMvUniform(3)
        distribution = NamedTupleDist(a=Normal(), p=Dirichlet([2.0, 3.0, 4.0]), c=4.2)
        target = BAT.MeasureBase.pushfwd(BAT.DistributionTransform(distribution, source), batmeasure(source))
        values = [(a=1.25, p=[0.2, 0.3, 0.5], c=4.2), (a=-0.5, p=[0.1, 0.6, 0.3], c=4.2)]
        state, f = fixture(target, NormalBased(), values)
        current = deepcopy(BAT.current_sample(state.chain_state))
        current.weight .= [2, 3]
        converted = EXT._inverse_display_samples(f, current, SEQUENTIAL)
        shape = EXT._live_display_shape(target, f, state)
        actual = EXT._canonical_display_samples(converted, shape)
        @test totalndof(shape) == 4
        @test totalndof(varshape(actual)) == 4
        @test all(v -> v.c == 4.2, converted.v)
        @test actual.v[1] ≈ [1.25, 0.2, 0.3, 0.5]
        @test actual.v[2] ≈ [-0.5, 0.1, 0.6, 0.3]
        @test all(v -> sum(v[2:4]) ≈ 1 && all(>(0), v[2:4]), actual.v)
        @test actual.logd ≈ logpdf.(Ref(distribution), values)
        @test (actual.weight, actual.info, actual.aux) == (current.weight, current.info, current.aux)
        @test isequal(actual.logd, converted.logd)

        evaluated = EvaluatedMeasure(target; empirical=converted)
        @test EXT._live_display_shape(evaluated, f, state) == shape
        # Empirical initialization is permitted, but its inferred shape must not
        # turn the constant into a displayed coordinate.
        lo, hi = EXT._estimate_prior_domain(BAT.Philox4x((84, 2)), evaluated, 4;
            n_prior_samples=16, display_shape=shape)
        @test length(lo) == length(hi) == 4
        @test all(lo .>= [-0.5, 0.1, 0.3, 0.3] .- 1e-12)
        @test all(hi .<= [1.25, 0.2, 0.6, 0.5] .+ 1e-12)
    end

    @testset "Retained chains precede current-only prototypes" begin
        target = MvNormal(zeros(2), ones(2))
        retained, _ = fixture(target, DoNotTransform(), [[99.0, 0.0], [99.0, 0.0]])
        current_only, _ = fixture(target, DoNotTransform(), [[7.0, 8.0], [9.0, 10.0]]; chain=2)
        # Adaptive coordinates must not substitute for current sampler x.
        foreach(v -> fill!(v, 99.0), current_only.chain_state.current.z.v)
        states = [current_only, retained]
        outputs = BAT._empty_chain_outputs.(states)
        push!(outputs[2][1], BAT.DensitySample([1.0, 2.0], -1.0, 0,
            sample_id(1, 1, 1, 1), nothing))
        push!(outputs[2][1], BAT.DensitySample([3.0, 4.0], -2.0, 2,
            sample_id(1, 1, 1, 2), nothing))
        before = state_snapshot.(states)
        seen = Float64[]
        f = RecordingForward(RecordingInverse(seen))
        history, shape = EXT._initial_display_outputs(states, outputs, f, varshape(target))
        @test findlast(==(3.0), seen) < findfirst(==(7.0), seen)
        @test !(99.0 in seen)
        @test history[2][1].v == [[1.0, 2.0], [3.0, 4.0]]
        @test history[2][1].weight == [0, 2]
        @test all(isnan, history[2][1].logd)
        @test length.(history[2]) == [2, 0]
        @test all(isempty, history[1])
        @test isequal(state_snapshot.(states), before)
        @test all(s -> totalndof(varshape(s)) == 2, Iterators.flatten(history))
        push!(history[1][1], history[2][1][1])
        @test isempty(history[1][2]) && isempty(history[2][2])
        @test isequal(state_snapshot.(states), before)
    end

    @testset "Display scalar types and appendable missing-Jacobian columns" begin
        rows = DensitySampleVector(v=[Float32[1, 2], Float32[3, 4]],
            logd=Float32[-1, -2], weight=Int32[1, 2], info=[:first, :second], aux=[nothing, nothing])
        identity_rows = canonical(identity, rows, ArrayShape{Real}(2))
        identity_empty = EXT._empty_display_samples(identity_rows)
        @test eltype(eltype(identity_empty.v)) == Float32
        @test eltype(identity_empty.logd) == Float32
        @test eltype(identity_empty.weight) == Int32
        append!(identity_empty, identity_rows)
        @test isequal(fields(identity_empty), fields(rows))

        wide = canonical(to_sampler_precision, rows, ArrayShape{Real}(2))
        @test eltype(eltype(wide.v)) == Float64
        @test all(isnan, wide.logd)
        @test wide.v == [[1.0, 2.0], [3.0, 4.0]]
        fresh = EXT._empty_display_samples(wide)
        append!(fresh, wide)
        @test isequal(fields(fresh), fields(wide))
        @test fresh.logd isa Vector
        @test eltype(fresh.weight) == Int32
        fresh.logd[1] = 123.0
        @test isnan(wide.logd[1])
        @test rows.logd == Float32[-1, -2]
        combined, times = EXT._append_timed_walkers([EXT._empty_display_samples(wide)],
            [wide], [EXT.SampleStepSpan[]], [EXT.SampleStepSpan[]])
        @test isequal(fields(only(combined)), fields(wide))
        @test isempty(only(times))

        target = Dirichlet([2.0, 3.0, 4.0])
        _, f = BAT.transform_and_unshape(NormalBased(), target, BATContext())
        simplex32 = canonical(f, DensitySampleVector(v=[Float32[0, 0]],
            logd=Float32[-1], weight=Int32[0]), varshape(target))
        storage = EXT._empty_display_samples(simplex32)
        @test totalndof(varshape(storage)) == 3
        @test eltype(eltype(storage.v)) == eltype(eltype(simplex32.v))
        @test eltype(storage.logd) == eltype(simplex32.logd)
        @test eltype(storage.weight) == Int32
        push!(storage, simplex32[1])
        @test isequal(fields(storage), fields(simplex32))
    end

    @testset "Domain draws are transformed before quantiles" begin
        original = Normal(100.0, 2.0)
        context = BATContext(rng=BAT.Philox4x((71, 22)))
        sampler, f = BAT.transform_and_unshape(NormalBased(), original, context)
        saved_rng = copy(BAT.get_rng(context))
        saved_task_rng = copy(Random.default_rng())
        expected_rng = copy(saved_rng)
        source = BAT.get_initsrc_from_target(sampler)
        z = [only(unshaped(rand(expected_rng, source), varshape(source))) for _ in 1:32]
        lo, hi = EXT._estimate_prior_domain(copy(saved_rng), sampler, 1;
            n_prior_samples=32, display_shape=varshape(original), f_display=inverse(f))
        @test only(lo) ≈ 100 + 2 * quantile(z, 0.0015)
        @test only(hi) ≈ 100 + 2 * quantile(z, 0.9985)
        @test BAT.get_rng(context) == saved_rng
        @test Random.default_rng() == saved_task_rng
        @test_throws ArgumentError EXT._estimate_prior_domain(copy(saved_rng), sampler, 2;
            n_prior_samples=4, display_shape=varshape(original), f_display=inverse(f))
    end

    @testset "Inferred Float32 schema accepts ordinary domain draws" begin
        context = BATContext(precision=Float32, rng=BAT.Philox4x((1, 2)))
        target = batmeasure(MvNormal(zeros(2), ones(2)))
        alg = TransformedMCMC(adaptive_transform=BAT.NoAdaptiveTransform(),
            transform_tuning=NoMCMCTransformTuning())
        state = BAT.MCMCState(alg, target, Int32(1), [Float32[0, 0]], context)
        saved = state_snapshot(state)
        history, shape = EXT._initial_display_outputs([state], [BAT._empty_chain_outputs(state)],
            identity, EXT._live_display_shape(nothing, identity, state))
        storage = only(only(history))
        @test isempty(storage)
        @test totalndof(shape) == 2
        @test eltype(eltype(storage.v)) == Float32
        draw_rng = copy(BAT.get_rng(context))
        source = BAT.get_initsrc_from_target(target)
        draws = reduce(hcat, [rand(draw_rng, source) for _ in 1:8])
        @test eltype(draws) == Float64
        lo, hi = EXT._estimate_prior_domain(copy(BAT.get_rng(context)), target, 2;
            display_shape=shape, n_prior_samples=8)
        @test lo ≈ [quantile(view(draws, i, :), 0.0015) for i in 1:2]
        @test hi ≈ [quantile(view(draws, i, :), 0.9985) for i in 1:2]
        push!(storage, BAT.current_sample(state.chain_state)[1])
        @test eltype(eltype(storage.v)) == Float32
        @test isequal(state_snapshot(state), saved)
    end
end

function timed_rows(values, weights, chain, walker, cycles, steps)
    DensitySampleVector(v=values, logd=[-sum(abs2, v) for v in values], weight=weights,
        info=[sample_id(chain, walker, cycles[i], steps[i]) for i in eachindex(values)])
end

# Interpret explicit provenance into individual observations. This oracle does
# not infer cycles from the surviving sample IDs or reproduce row compression.
function expanded_records(samples, timing)
    records = [(i, j, Int(s.cycle), Int(step)) => Tuple(samples[i][j].v[s.row])
        for i in eachindex(samples) for j in eachindex(samples[i])
        for s in timing[i][j] for step in s.first:s.last]
    @test length(unique(first.(records))) == length(records)
    return Dict(records)
end

@testset "Display-owned timed and untimed append" begin
    a, b, c, d = [0.2, 0.3, 0.5], [0.1, 0.5, 0.4], [0.6, 0.2, 0.2], [0.1, 0.2, 0.7]
    old = [[timed_rows([a], [2], 1, 1, [1], [1]), timed_rows([c], [1], 1, 2, [1], [1])],
        [timed_rows([b], [2], 2, 1, [1], [3])]]
    fresh = [[timed_rows([a, b], [3, 1], 1, 1, [2, 2], [1, 4]),
              timed_rows([c, d], [1, 2], 1, 2, [1, 2], [2, 1])],
        [timed_rows([b], [1], 2, 1, [2], [5])]]
    old_time = [[[span(1, 1, 1, 2)], [span(1, 1, 1, 1)]], [[span(1, 1, 3, 4)]]]
    fresh_time = [[[span(1, 2, 1, 3), span(2, 2, 4, 4)],
                   [span(1, 1, 2, 2), span(2, 2, 1, 2)]], [[span(1, 2, 5, 5)]]]
    expected = Dict(
        (1, 1, 1, 1)=>Tuple(a), (1, 1, 1, 2)=>Tuple(a),
        (1, 1, 2, 1)=>Tuple(a), (1, 1, 2, 2)=>Tuple(a), (1, 1, 2, 3)=>Tuple(a), (1, 1, 2, 4)=>Tuple(b),
        (1, 2, 1, 1)=>Tuple(c), (1, 2, 1, 2)=>Tuple(c), (1, 2, 2, 1)=>Tuple(d), (1, 2, 2, 2)=>Tuple(d),
        (2, 1, 1, 3)=>Tuple(b), (2, 1, 1, 4)=>Tuple(b), (2, 1, 2, 5)=>Tuple(b))
    before = ([fields.(chain) for chain in old], [fields.(chain) for chain in fresh])
    merged, timing = EXT._append_timed_chains(old, fresh, old_time, fresh_time)
    @test expanded_records(merged, timing) == expected
    @test merged[1][1].weight == [5, 1]
    @test merged[1][2].weight == [2, 2]
    @test merged[2][1].weight == [3]
    @test [(s.row, s.cycle, s.first, s.last) for s in timing[1][1]] ==
        [(1, 1, 1, 2), (1, 2, 1, 3), (2, 2, 4, 4)]
    @test merged[1][1].info[1] == old[1][1].info[1]
    @test sum(sum(s.weight) for chain in merged for s in chain) == length(expected)
    @test isequal(([fields.(chain) for chain in old], [fields.(chain) for chain in fresh]), before)

    data = EXT._sample_window_data(merged, [length.(chain) for chain in merged], timing)
    for window in ((1, 1), (2, 2), (3, 4), (5, 5), (6, 8))
        selected, _ = EXT._window_selection(data, window)
        expected_values = [value for (key, value) in expected if window[1] <= key[4] <= window[2]]
        @test sum(selected.weight) == length(expected_values)
        for value in unique(values(expected))
            @test sum((selected.weight[i] for i in eachindex(selected) if Tuple(selected.v[i]) == value); init=0) ==
                count(==(value), expected_values)
        end
    end

    @testset "Pending replacements start from committed history" begin
        base = [old[1][1]]
        base_time = [old_time[1][1]]
        pending1 = [timed_rows([b], [1], 1, 1, [1], [3])]
        pending3 = [timed_rows([b], [3], 1, 1, [1], [3])]
        frame1, time1 = EXT._append_timed_walkers(base, pending1, base_time, [[span(1, 1, 3, 3)]])
        held_frame = fields(only(frame1))
        frame3, time3 = EXT._append_timed_walkers(base, pending3, base_time, [[span(1, 1, 3, 5)]])
        @test only(frame1).weight == [2, 1]
        @test only(frame3).weight == [2, 3]
        @test only(base).weight == [2]
        @test isequal(fields(only(frame1)), held_frame)
        @test length(expanded_records([frame1], [time1])) == 3
        @test length(expanded_records([frame3], [time3])) == 5
        terminal, terminal_time = EXT._append_timed_walkers(frame3,
            [EXT._empty_display_samples(only(frame3))], time3, [EXT.SampleStepSpan[]])
        @test isequal(fields(only(terminal)), fields(only(frame3)))
        @test expanded_records([terminal], [terminal_time]) == expanded_records([frame3], [time3])
        only(terminal).v[1][1] = -5.0
        @test only(frame3).v[1] == a
        @test only(base).v[1] == a
    end

    @testset "Untimed fractional mass and retained zero rows" begin
        fractional = timed_rows([a, a, b, c], [0.1, 0.2, 0.0, 1.75], 1, 1, [1, 1, 1, 1], [1, 2, 3, 4])
        # The binary input weights define the reference mass. Floating addition
        # may regroup across batches, so require a bounded error, not equality.
        mass = sum(BigFloat.(fractional.weight))
        tolerance = 8 * eps(Float64) * max(abs(mass), one(mass))
        for cuts in ((4,), (1, 4), (2, 3, 4))
            accumulator = [EXT._empty_display_samples(fractional)]
            times = [EXT.SampleStepSpan[]]
            start = 1
            for stop in cuts
                accumulator, times = EXT._append_timed_walkers(accumulator,
                    [fractional[start:stop]], times, [EXT.SampleStepSpan[]])
                start = stop + 1
            end
            result = only(accumulator)
            @test result.v == [a, b, c]
            @test abs(sum(BigFloat.(result.weight)) - mass) <= tolerance
            @test result.weight[2:3] == [0.0, 1.75]
            @test isempty(only(times))
            data = EXT._sample_window_data([accumulator], [length.(accumulator)], nothing)
            selected, groups = EXT._window_selection(data, (2, 3))
            @test selected.weight == [0.0, 1.75]
            @test isempty(groups)
        end
        @test fractional.weight == [0.1, 0.2, 0.0, 1.75]
    end

    @testset "Dimension and topology failures" begin
        two = DensitySampleVector(v=[[1.0, 2.0]])
        three = DensitySampleVector(v=[[1.0, 2.0, 3.0]])
        none = [EXT.SampleStepSpan[]]
        @test_throws ArgumentError EXT._canonical_display_samples(two, ArrayShape{Real}(3))
        @test_throws ArgumentError EXT._inverse_display_samples(identity, EXT._empty_display_samples(two), SEQUENTIAL)
        @test_throws ArgumentError EXT._append_timed_walkers([two], [three], none, none)
        @test_throws ArgumentError EXT._append_timed_walkers([three], [three, three], none, none)
        @test_throws ArgumentError EXT._append_timed_walkers([three], [three], none, Vector{EXT.SampleStepSpan}[])
        @test_throws ArgumentError EXT._append_timed_chains(old, fresh[1:1], old_time, fresh_time)
        @test_throws ArgumentError EXT._transform_display_outputs(identity, [[three]], [[two, two]],
            ArrayShape{Real}(3); executor=SEQUENTIAL)
        @test_throws ArgumentError EXT._transform_display_outputs(identity, [[three]], [[two], [two]],
            ArrayShape{Real}(3); executor=SEQUENTIAL)
        @test_throws ArgumentError EXT._transform_display_outputs(identity, [[two]], [[three]],
            ArrayShape{Real}(3); executor=SEQUENTIAL)
        state, f = fixture(MvNormal(zeros(2), ones(2)), DoNotTransform(), [[0.0, 0.0]])
        outputs = BAT._empty_chain_outputs(state)
        @test_throws ArgumentError EXT._initial_display_outputs([state, state], [outputs], f, ArrayShape{Real}(2))
        @test_throws ArgumentError EXT._initial_display_outputs([state], [[outputs[1], outputs[1]]], f, ArrayShape{Real}(2))
    end
end
end # module
