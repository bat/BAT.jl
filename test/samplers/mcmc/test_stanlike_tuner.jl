# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestStanLikeTuner

using BAT, Test, LinearAlgebra, Random, Random123, Distributions, AffineMaps

# Closed-form calendar in unbounded arithmetic, independent of the production
# remaining-distance loop. This preserves BAT's retained split locations.
function exact_calendar(initial, terminal, width, horizon)
    initial, terminal, width, horizon = BigInt.((initial, terminal, width, horizon))
    stop = horizon - terminal
    splits = BigInt[]
    for k in 0:(8sizeof(Int))
        boundary = initial + width * (big(2)^(k + 1) - 1)
        boundary <= stop || break
        if boundary + 2width * big(2)^k > stop
            push!(splits, stop)
            break
        end
        push!(splits, boundary)
    end
    !isempty(splits) && last(splits) == horizon && pop!(splits)
    isempty(splits) && return (1, 0, Int[])
    return Int(initial + 1), Int(last(splits)), Int.(splits)
end

function check_calendar(initial, terminal, width, horizon)
    actual = BAT._stan_adaptation_windows(initial, terminal, width, horizon)
    @test actual == exact_calendar(initial, terminal, width, horizon)
    start, stop, splits = actual
    @test isempty(splits) ? (start, stop) == (1, 0) :
        (start <= first(splits) <= last(splits) == stop < horizon && issorted(splits) && allunique(splits))
    @test length(splits) <= 8sizeof(Int)
end

@testset "StanLike integer calendar, including overflow boundaries" begin
    for initial in 0:6, terminal in 0:6, width in 1:8, horizon in 0:40
        check_calendar(initial, terminal, width, horizon)
    end
    m = typemax(Int)
    edges = [0, 1, 2, 3, m ÷ 4, m ÷ 2, m ÷ 2 + 1, m - 2, m - 1, m]
    for initial in edges, terminal in edges, width in edges[2:end], horizon in edges
        check_calendar(initial, terminal, width, horizon)
    end
    rng = MersenneTwister(2800)
    for _ in 1:1000
        initial, terminal, horizon = rand(rng, 0:m, 3)
        check_calendar(initial, terminal, rand(rng, 1:m), horizon)
    end
    for bad in (-1, typemin(Int), big(m) + 1, typemax(UInt)), position in 1:4
        args = Any[0, 0, 1, 10]
        args[position] = bad
        @test_throws ArgumentError BAT._stan_adaptation_windows(args...)
    end
    @test_throws ArgumentError BAT._stan_adaptation_windows(0, 0, 0, 0)
    @test BAT._stan_adaptation_windows(UInt8(0), big(1), Int128(2), UInt64(7)) == (1, 6, [2, 6])
    @test BAT._stan_adaptation_windows(0, 1, 25, 7) == (1, 0, Int[])
    @test BAT._stan_adaptation_windows(0, 1, 25, 31) == (1, 30, [30])
    @test BAT._stan_adaptation_windows(0, 0, 2, 7) == (1, 2, [2])
    @test BAT._stan_adaptation_windows(0, 0, 1, 7) == (1, 3, [1, 3])
    @test BAT._stan_adaptation_windows(75, 50, 25, 999) == (76, 949, [100, 150, 250, 450, 949])
    # Equality in the lookahead must retain the first split.
    @test BAT._stan_adaptation_windows(0, 1, 2, 7) == (1, 6, [2, 6])
    # Float64 division would incorrectly miss absorption at this boundary.
    width = Int(2)^(8sizeof(Int) - 4)
    @test BAT._stan_adaptation_windows(0, 1, width, 3width) == (1, 3width - 1, [3width - 1])
end

function fixture(; initial=0, terminal=1, width=25, dims=1, walkers=1)
    tuning = BAT.StanLikeTuning(init_buffer=initial, term_buffer=terminal, window_size=width)
    algorithm = TransformedMCMC(proposal=RandomWalk(), transform_tuning=tuning,
        pretransform=DoNotTransform(), nchains=1, nwalkers=walkers, nsteps=8)
    target = BAT.batmeasure(MvNormal(zeros(dims), Matrix{Float64}(I, dims, dims)))
    chain = BAT.MCMCChainState(algorithm, target, 1, [zeros(dims) for _ in 1:walkers],
        BATContext(rng=Philox4x((0x281, 0x282))))
    return chain, BAT.create_trafo_tuner_state(tuning, chain, 8)
end

function sample_state(rows; weights=zeros(length(rows)))
    d = DensitySampleVector(v=[Float64.(row) for row in rows],
        logd=fill(-1.0, length(rows)), weight=Float64.(weights),
        info=fill(nothing, length(rows)), aux=fill(nothing, length(rows)))
    return (x=d, z=deepcopy(d))
end

function stats_snapshot(stats)
    p, l = stats.param_stats, stats.logtf_stats
    return (p.m, p.mean.sum_w, copy(p.mean.S), copy(p.mean.C),
        p.cov.m, p.cov.n, p.cov.sum_w, p.cov.sum_w2,
        copy(p.cov.Mean_X), copy(p.cov.New_Mean_X), copy(p.cov.S),
        copy(p.minimum), copy(p.maximum), l.mean.sum_w, l.mean.sum_v,
        l.var.n, l.var.sum_w, l.var.sum_w2, l.var.mean_x, l.var.s,
        l.minimum, l.maximum, copy(stats.mode))
end
function storage(stats)
    p, l = stats.param_stats, stats.logtf_stats
    return (stats, p, l, p.mean, p.cov, l.mean, l.var, p.mean.S, p.mean.C,
        p.cov.Mean_X, p.cov.New_Mean_X, p.cov.S, p.minimum, p.maximum, stats.mode)
end
snapshot(tuner) = (tuner.tuning, tuner.i, tuner.window_start, tuner.window_end,
    copy(tuner.window_splits), stats_snapshot(tuner.stats))
function assert_fresh(stats)
    @test isequal(stats_snapshot(stats), stats_snapshot(typeof(stats)(length(stats.mode))))
end
function assert_storage(stats, before)
    @test all(a === b for (a, b) in zip(storage(stats), before))
end

function supplied_step(f, tuner, chain, current, proposed;
    accepted=trues(length(current.x)), order=collect(eachindex(accepted)))
    chain.accepted .= accepted
    before_current, before_proposed = deepcopy(current), deepcopy(proposed)
    rng = copy(BAT.get_rng(chain.context))
    info = BAT.MCMCStepInfo(ones(length(accepted)), nothing, nothing, nothing, nothing, order)
    next, returned_tuner, returned_chain = BAT.mcmc_tune_trafo_post_step!!(
        f, tuner, chain, chain.proposal, current, proposed, info)
    @test returned_tuner === tuner && returned_chain === chain
    @test current == before_current && proposed == before_proposed
    @test BAT.get_rng(chain.context) == rng
    return next
end
step(f, tuner, chain, row) = supplied_step(f, tuner, chain,
    sample_state([zeros(length(row))]), sample_state([row]))
metric(f) = Matrix(f.A * f.A')

function exact_covariance(rows)
    xs = hcat((Rational{BigInt}.(row) for row in rows)...)
    center = sum(xs; dims=2) / length(rows)
    centered = xs .- center
    return centered * centered' / (length(rows) - 1)
end

@testset "Explicit init and reinit clear all prior statistics in place" begin
    for reset! in (BAT.mcmc_trafo_tuning_init!!, BAT.mcmc_trafo_tuning_reinit!!),
        old_value in (10.0, 1e16)
        chain, tuner = fixture()
        BAT.mcmc_trafo_tuning_init!!(tuner, chain, 32)
        f = chain.f_transform
        for _ in 1:6
            f = step(f, tuner, chain, [old_value])
        end
        @test tuner.stats.param_stats.cov.n == 6
        owned, rng = storage(tuner.stats), copy(BAT.get_rng(chain.context))
        proposal = chain.proposal
        current, proposed = deepcopy(chain.current), deepcopy(chain.proposed)
        @test reset!(tuner, chain, 32) === nothing
        @test (tuner.i, tuner.window_start, tuner.window_end, tuner.window_splits) == (0, 1, 30, [30])
        assert_fresh(tuner.stats)
        assert_storage(tuner.stats, owned)
        @test chain.f_transform === f && chain.proposal === proposal
        @test chain.current == current && chain.proposed == proposed
        @test BAT.get_rng(chain.context) == rng
        rows = [[x] for x in repeat([0, 2], 15)]
        for row in rows
            f = step(f, tuner, chain, row)
        end
        @test only(exact_covariance(rows)) == 30//29
        @test metric(f) ≈ Float64.(exact_covariance(rows))
        assert_fresh(tuner.stats)
        assert_storage(tuner.stats, owned)
    end
    # Cross-covariance and centers must also exclude an interrupted window.
    for reset! in (BAT.mcmc_trafo_tuning_init!!, BAT.mcmc_trafo_tuning_reinit!!)
        chain, tuner = fixture(width=4, dims=2)
        BAT.mcmc_trafo_tuning_init!!(tuner, chain, 6)
        f = step(chain.f_transform, tuner, chain, [1e16, -1e16])
        reset!(tuner, chain, 6)
        rows = [[-1, -3], [-1, 1], [1, -1], [1, 3]]
        for row in rows
            f = step(f, tuner, chain, row)
        end
        @test exact_covariance(rows) == [4//3 4//3; 4//3 20//3]
        @test metric(f) ≈ Float64.(exact_covariance(rows))
        assert_fresh(tuner.stats)
    end
end

@testset "Inactive schedules, omitted tails and completed counters stay inactive" begin
    for horizon in (0, 1, UInt(0), big(1), 8)
        chain, tuner = fixture()
        # Dirty statistics from an interrupted eligible window.
        BAT.mcmc_trafo_tuning_init!!(tuner, chain, 32)
        step(chain.f_transform, tuner, chain, [10.0])
        BAT.mcmc_trafo_tuning_reinit!!(tuner, chain, horizon)
        @test (tuner.i, tuner.window_start, tuner.window_end, tuner.window_splits) == (0, 1, 0, Int[])
        fresh = snapshot(tuner)
        f = chain.f_transform
        for _ in 1:12
            @test step(f, tuner, chain, [10.0]) === f
        end
        @test isequal(snapshot(tuner), fresh)
        assert_fresh(tuner.stats)
        # Original no-split H28 reproduction now enters the next window cleanly.
        BAT.mcmc_trafo_tuning_reinit!!(tuner, chain, 32)
        for x in repeat([0.0, 2.0], 15)
            f = step(f, tuner, chain, [x])
        end
        @test only(metric(f)) ≈ 30 / 29
    end
    chain, tuner = fixture(terminal=0, width=2)
    BAT.mcmc_trafo_tuning_init!!(tuner, chain, 8)
    @test (tuner.window_start, tuner.window_end, tuner.window_splits) == (1, 2, [2])
    f = step(chain.f_transform, tuner, chain, [0.0])
    f = step(f, tuner, chain, [2.0])
    @test only(metric(f)) ≈ 2.0
    finished = snapshot(tuner)
    for _ in 1:10
        @test step(f, tuner, chain, [1e308]) === f
    end
    @test isequal(snapshot(tuner), finished)
    assert_fresh(tuner.stats)
    BAT.mcmc_trafo_tuning_reinit!!(tuner, chain, 8)
    f = step(f, tuner, chain, [0.0]); f = step(f, tuner, chain, [2.0])
    @test only(metric(f)) ≈ 2.0

    # Exercise the final safe increment without iterating an enormous horizon.
    chain, tuner = fixture(initial=typemax(Int)-4, terminal=0, width=1)
    BAT.mcmc_trafo_tuning_init!!(tuner, chain, typemax(Int))
    @test tuner.window_end == typemax(Int)-3
    tuner.i = tuner.window_end - 1
    f = chain.f_transform
    @test step(f, tuner, chain, [0.0]) === f # singleton at last retained split
    @test tuner.i == tuner.window_end
    finished = snapshot(tuner)
    for _ in 1:3
        @test step(f, tuner, chain, [0.0]) === f
    end
    @test isequal(snapshot(tuner), finished)
end

@testset "StanLike validation fails before changing any tuner state" begin
    for reset! in (BAT.mcmc_trafo_tuning_init!!, BAT.mcmc_trafo_tuning_reinit!!)
        chain, tuner = fixture()
        BAT.mcmc_trafo_tuning_init!!(tuner, chain, 32)
        step(chain.f_transform, tuner, chain, [10.0])
        owned, splits = storage(tuner.stats), tuner.window_splits
        for bad in (-1, typemin(Int), big(typemax(Int)) + 1, typemax(UInt))
            before = snapshot(tuner)
            @test_throws ArgumentError reset!(tuner, chain, bad)
            @test isequal(snapshot(tuner), before)
            assert_storage(tuner.stats, owned)
            @test tuner.window_splits === splits
        end
        for config in ((-1, 0, 1), (0, -1, 1), (0, 0, 0), (0, 0, -1)), horizon in (0, 32)
            tuner.tuning = BAT.StanLikeTuning(init_buffer=config[1], term_buffer=config[2], window_size=config[3])
            before = snapshot(tuner)
            @test_throws ArgumentError reset!(tuner, chain, horizon)
            @test isequal(snapshot(tuner), before)
            assert_storage(tuner.stats, owned)
            @test tuner.window_splits === splits
        end
    end
end

@testset "Window row count, finite degeneracy and metric admission" begin
    chain, tuner = fixture(width=1)
    BAT.mcmc_trafo_tuning_init!!(tuner, chain, 8)
    f = MulAdd(fill(3.0, 1, 1), [7.0])
    @test step(f, tuner, chain, [10.0]) === f
    @test (f.A, f.b) == (fill(3.0, 1, 1), [7.0])
    assert_fresh(tuner.stats)
    # The following window gets only its own five rows (no singleton ten).
    rows = [[0], [2], [0], [2], [0]]
    for row in rows
        f = step(f, tuner, chain, row)
    end
    @test metric(f) ≈ Float64.(exact_covariance(rows))
    @test f.b == [0.0]
    assert_fresh(tuner.stats)

    chain, tuner = fixture(width=1, walkers=2)
    BAT.mcmc_trafo_tuning_init!!(tuner, chain, 8)
    f = MulAdd(fill(3.0, 1, 1), [7.0])
    next = supplied_step(f, tuner, chain,
        sample_state([[99.0], [2.0]]; weights=[1e200, 0]),
        sample_state([[0.0], [99.0]]; weights=[0, 1e200]); accepted=[true, false], order=[2, 1])
    @test next !== f && only(metric(next)) ≈ 2.0
    @test next.b == [0.0]
    assert_fresh(tuner.stats)

    for rows in ([[3.0], [3.0]], [[-1.0, -1.0], [1.0, 1.0]])
        chain, tuner = fixture(width=2, dims=length(first(rows)))
        BAT.mcmc_trafo_tuning_init!!(tuner, chain, 8)
        f = chain.f_transform
        for row in rows
            f = step(f, tuner, chain, row)
        end
        # Fixed expected Positive policy: zero scalar pivot becomes one;
        # a singular 2D covariance [2 2; 2 2] becomes metric [2 2; 2 3].
        expected = length(first(rows)) == 1 ? ones(1, 1) : [2.0 2.0; 2.0 3.0]
        @test metric(f) ≈ expected
        assert_fresh(tuner.stats)
    end

    for (magnitude, T, message) in ((1e308, Float64, "window covariance"),
        (1e40, Float32, "covariance factor"), (1e-50, Float32, "covariance factor"))
        chain, tuner = fixture(width=2)
        BAT.mcmc_trafo_tuning_init!!(tuner, chain, 8)
        f = MulAdd(fill(T(3), 1, 1), [T(7)])
        f = step(f, tuner, chain, [-magnitude])
        proposed = sample_state([[magnitude]])
        before = deepcopy(proposed)
        owned, rng = storage(tuner.stats), copy(BAT.get_rng(chain.context))
        error = try
            BAT.mcmc_tune_trafo_post_step!!(f, tuner, chain, chain.proposal,
                sample_state([[0.0]]), proposed, BAT.MCMCStepInfo([1.0]))
        catch err
            err
        end
        @test error isa ArgumentError
        @test occursin(message, sprint(showerror, error))
        @test (f.A, f.b) == (fill(T(3), 1, 1), [T(7)])
        @test tuner.i == 2 && tuner.stats.param_stats.cov.n == 2
        @test proposed == before && BAT.get_rng(chain.context) == rng
        assert_storage(tuner.stats, owned)
        if magnitude == 1e308
            @test !all(isfinite, Matrix(tuner.stats.param_stats.cov))
        else
            @test only(Matrix(tuner.stats.param_stats.cov)) ≈ 2magnitude^2
        end
        # Failed post-step state is deliberately not resumed: explicit reset
        # discards its evidence and restores a usable campaign.
        BAT.mcmc_trafo_tuning_reinit!!(tuner, chain, 8)
        assert_fresh(tuner.stats)
        f = step(f, tuner, chain, [0.0]); f = step(f, tuner, chain, [2.0])
        @test only(metric(f)) ≈ 2.0
    end
end

end # module
