# This file is a part of BAT.jl, licensed under the MIT License (MIT).
# Supplied-state tuner checks and public sampling regressions.
module TestReweightTuners

using BAT, Test, LinearAlgebra, Statistics, StatsBase, Distributions, Random123
using DensityInterface, Logging

function exact_frequency(rows, weights)
    x = [Rational{BigInt}.(row) for row in rows]
    w = Rational{BigInt}.(weights)
    total = sum(w)
    mu = [sum(w[i] * x[i][j] for i in eachindex(w)) / total for j in eachindex(first(x))]
    scatter = zeros(Rational{BigInt}, length(mu), length(mu))
    for k in eachindex(w), j in eachindex(mu), i in eachindex(mu)
        scatter[i, j] += w[k] * (x[k][i] - mu[i]) * (x[k][j] - mu[j])
    end
    return (total=total, squared=sum(abs2, w), mean=mu, cov=scatter / (total - 1))
end

function samples(rows; weights=ones(length(rows)), scores=[-sum(abs2, x) / 2 for x in rows])
    DensitySampleVector(v=[Float64.(x) for x in rows], logd=Float64.(scores),
        weight=Float64.(weights), info=fill(nothing, length(rows)), aux=fill(nothing, length(rows)))
end
sample_state(rows; kwargs...) = let x = samples(rows; kwargs...)
    (x=x, z=deepcopy(x))
end

function chain_fixture(tuning; dims=1, nwalkers=1)
    target = BAT.batmeasure(MvNormal(zeros(dims), Matrix{Float64}(I, dims, dims)))
    algorithm = TransformedMCMC(proposal=RandomWalk(), transform_tuning=tuning,
        pretransform=DoNotTransform(), nchains=1, nwalkers=nwalkers, nsteps=8)
    context = BATContext(rng=Philox4x((0x211, 0x212)))
    chain = BAT.MCMCChainState(algorithm, target, 1, [zeros(dims) for _ in 1:nwalkers], context)
    # Finite acceptance bookkeeping; β=1 additionally isolates covariance
    # adaptation from changes in a proposal's target acceptance interval.
    chain.stepno = 4
    chain.nattempts .= 4
    chain.nsamples .= 1
    tuner = BAT.create_trafo_tuner_state(tuning, chain, 8)
    return chain, tuner
end

function snapshot(stats)
    p, l = stats.param_stats, stats.logtf_stats
    (parameter_mean=(sum_w=p.mean.sum_w, S=copy(p.mean.S), C=copy(p.mean.C)),
        covariance=(n=p.cov.n, sum_w=p.cov.sum_w, sum_w2=p.cov.sum_w2,
            center=copy(p.cov.Mean_X), next_center=copy(p.cov.New_Mean_X), S=copy(p.cov.S)),
        parameter_extrema=(minimum=copy(p.minimum), maximum=copy(p.maximum)),
        score_mean=(sum_w=l.mean.sum_w, sum_v=l.mean.sum_v),
        score_variance=(n=l.var.n, sum_w=l.var.sum_w, sum_w2=l.var.sum_w2,
            center=l.var.mean_x, S=l.var.s),
        score_extrema=(minimum=l.minimum, maximum=l.maximum), mode=copy(stats.mode))
end

function storage(stats)
    p, l = stats.param_stats, stats.logtf_stats
    (p, l, p.mean, p.cov, l.mean, l.var, p.mean.S, p.mean.C,
        p.cov.Mean_X, p.cov.New_Mean_X, p.cov.S, p.minimum, p.maximum, stats.mode)
end

function assert_storage(stats, before)
    @test all(pair -> pair[1] === pair[2], zip(storage(stats), before))
    return nothing
end

function assert_fresh(stats)
    fresh = typeof(stats)(length(stats.mode))
    @test isequal(snapshot(stats), snapshot(fresh))
    @test isnan(stats.logtf_stats.mean[])
    @test all(isnan, stats.param_stats.mean)
    @test all(isnan, stats.param_stats.cov)
    return nothing
end

metric(f) = Matrix(f.A * f.A')

function supplied_step(f, tuner, chain, current, proposed;
    accepted=trues(length(current.x)), order=collect(eachindex(accepted)))
    chain.accepted .= accepted
    before_current, before_proposed = deepcopy(current), deepcopy(proposed)
    rng = copy(BAT.get_rng(chain.context))
    info = BAT.MCMCStepInfo(ones(length(accepted)), nothing, nothing, nothing, nothing, order)
    f_new, tuner_new, chain_new = BAT.mcmc_tune_trafo_post_step!!(
        f, tuner, chain, chain.proposal, current, proposed, info)
    @test tuner_new === tuner
    @test chain_new === chain
    @test current == before_current
    @test proposed == before_proposed
    @test BAT.get_rng(chain.context) == rng
    return f_new, tuner_new, chain_new
end

@testset "StanLike real two-window reset uses only the next window" begin
    tuning = BAT.StanLikeTuning(init_buffer=0, term_buffer=1, window_size=2)
    chain, tuner = chain_fixture(tuning)
    BAT.mcmc_trafo_tuning_init!!(tuner, chain, 8)
    @test tuner.window_splits == [2, 6]
    @test (tuner.window_start, tuner.window_end) == (1, 6)
    owned = storage(tuner.stats)
    stats = tuner.stats
    f = chain.f_transform
    windows = ([[1e16 - 2], [1e16 + 2]], [[1.0], [3.0], [1.0], [3.0]])
    for rows in windows
        expected = exact_frequency(rows, ones(length(rows)))
        for (i, row) in pairs(rows)
            previous = f
            f, tuner, chain = supplied_step(f, tuner, chain,
                sample_state([[0.0]]; weights=[0.0]), sample_state([row]; weights=[0.0]))
            if i < length(rows)
                @test f === previous
            else
                @test f !== previous
            end
        end
        @test metric(f) ≈ Float64.(expected.cov) rtol=1e-12 atol=1e-12
        @test f.b == [0.0] # StanLike deliberately does not shift by the mean.
        @test tuner.stats === stats
        assert_storage(tuner.stats, owned)
        assert_fresh(tuner.stats)
    end
    @test tuner.i == 6
    @test only(metric(f)) ≈ 4 / 3
end

@testset "AdaptiveAffine r=0 forgets seeded initialization statistics" begin
    populations = ((rows=[[1.0], [3.0]], scores=[-1.0, -3.0]),
        (rows=[[1.0, 1.0], [3.0, 1.0], [1.0, 3.0], [3.0, 5.0]],
            scores=[-1.0, -2.0, -3.0, -4.0]))
    for population in populations
        rows, scores = population.rows, population.scores
        dims = length(first(rows))
        tuning = AdaptiveAffineTuning(r=0, β=1)
        chain, tuner = chain_fixture(tuning; dims=dims)
        prefix = samples([fill(1e16, dims)]; scores=[10.0])
        BAT.mcmc_trafo_tuning_postinit!!(tuner, chain, [prefix])
        @test tuner.stats.param_stats.cov.Mean_X == fill(1e16, dims)
        @test tuner.stats.mode == fill(1e16, dims)
        owned, stats = storage(tuner.stats), tuner.stats
        suffix = samples(rows; scores=scores)
        before_prefix, before_suffix = deepcopy(prefix), deepcopy(suffix)
        rng = copy(BAT.get_rng(chain.context))
        f, returned_tuner, returned_chain = BAT.mcmc_tune_trafo_post_cycle!!(
            chain.f_transform, tuner, chain, chain.proposal, [suffix])
        expected = exact_frequency(rows, ones(length(rows)))
        @test returned_tuner === tuner && returned_chain === chain
        @test tuner.stats === stats
        assert_storage(stats, owned)
        @test prefix == before_prefix && suffix == before_suffix
        @test BAT.get_rng(chain.context) == rng
        @test metric(f) ≈ Float64.(expected.cov) rtol=1e-12 atol=1e-12
        @test f.b ≈ Float64.(expected.mean)
        @test stats.param_stats.mean ≈ Float64.(expected.mean)
        @test stats.param_stats.cov.Mean_X ≈ Float64.(expected.mean)
        @test Matrix(stats.param_stats.cov) ≈ Float64.(expected.cov)
        @test stats.param_stats.cov.n == length(rows)
        @test stats.logtf_stats.var.n == length(rows)
        @test stats.logtf_stats.mean[] ≈ mean(scores)
        @test stats.mode == rows[argmax(scores)]
        @test stats.param_stats.maximum == [maximum(x[j] for x in rows) for j in 1:dims]
        @test stats.param_stats.minimum == [minimum(x[j] for x in rows) for j in 1:dims]
        @test stats.logtf_stats.maximum == maximum(scores)
        @test stats.logtf_stats.minimum == minimum(scores)
        @test tuner.iteration == 2
        @test tuner.scale == 1
    end
end

@testset "AdaptiveAffine positive forgetting retains frequency correction and blending" begin
    tuning = AdaptiveAffineTuning(r=0.5, λ=1, β=1)
    chain, tuner = chain_fixture(tuning)
    prefix = samples([[0.0], [4.0]]; weights=[2, 1], scores=[-10, -2])
    suffix = samples([[1.0], [3.0]]; weights=[1, 2], scores=[-1, -3])
    before_prefix, before_suffix = deepcopy(prefix), deepcopy(suffix)
    owned, stats = storage(tuner.stats), tuner.stats
    rng = copy(BAT.get_rng(chain.context))
    f, tuner, chain = BAT.mcmc_tune_trafo_post_cycle!!(
        chain.f_transform, tuner, chain, chain.proposal, [prefix])
    first = exact_frequency([[0], [4]], [2, 1])
    @test metric(f) ≈ Float64.(first.cov)
    @test f.b ≈ Float64.(first.mean)
    previous_metric, previous_center = metric(f), copy(f.b)
    f, tuner, chain = BAT.mcmc_tune_trafo_post_cycle!!(
        f, tuner, chain, chain.proposal, [suffix])
    expected = exact_frequency([[0], [4], [1], [3]], [1, 1//2, 1, 2])
    @test expected.total == 9//2
    @test expected.squared == 25//4
    @test expected.mean == [2]
    @test only(expected.cov) == 18//7
    @test stats.param_stats.cov.sum_w == 4.5
    @test stats.param_stats.cov.sum_w2 == 6.25
    @test stats.param_stats.cov.n == 4
    @test stats.logtf_stats.var.n == 4
    @test stats.param_stats.mean ≈ Float64.(expected.mean)
    @test Matrix(stats.param_stats.cov) ≈ Float64.(expected.cov)
    @test stats.logtf_stats.mean[] ≈ -4
    @test stats.mode == [1.0]
    @test metric(f) ≈ (previous_metric + Float64.(expected.cov)) / 2
    @test only(metric(f)) ≈ 83 / 21
    @test f.b ≈ (previous_center + Float64.(expected.mean)) / 2
    @test only(f.b) ≈ 5 / 3
    @test tuner.iteration == 3 && tuner.scale == 1
    @test tuner.stats === stats
    assert_storage(stats, owned)
    @test prefix == before_prefix && suffix == before_suffix
    @test BAT.get_rng(chain.context) == rng
end

@testset "StanLike uses selected states and unit weights in logical order" begin
    tuning = BAT.StanLikeTuning(init_buffer=0, term_buffer=1, window_size=2)
    chain, tuner = chain_fixture(tuning; nwalkers=2)
    BAT.mcmc_trafo_tuning_init!!(tuner, chain, 8)
    f = chain.f_transform
    current = sample_state([[10.0], [20.0]]; weights=[0, 1e200])
    proposed = sample_state([[1.0], [99.0]]; weights=[1e200, 0])
    f, tuner, chain = supplied_step(f, tuner, chain, current, proposed;
        accepted=[true, false], order=[2, 1])
    @test tuner.stats.param_stats.cov.n == 2
    @test tuner.stats.param_stats.cov.sum_w == 2
    @test only(tuner.stats.param_stats.mean) ≈ 10.5
    current = sample_state([[2.0], [3.0]]; weights=[0, 1e200])
    proposed = sample_state([[99.0], [4.0]]; weights=[1e200, 0])
    f, tuner, chain = supplied_step(f, tuner, chain, current, proposed;
        accepted=[false, true], order=[2, 1])
    expected = exact_frequency([[1], [20], [2], [4]], ones(Int, 4))
    @test only(expected.cov) == 955//12
    @test metric(f) ≈ Float64.(expected.cov)
    assert_fresh(tuner.stats)
end

@testset "invalid reweight factors leave the actual tuner untouched" begin
    for factor in (-1.0, NaN, Inf)
        tuning = AdaptiveAffineTuning(r=factor, β=1)
        chain, tuner = chain_fixture(tuning)
        prefix = samples([[1.0], [3.0]])
        BAT.mcmc_trafo_tuning_postinit!!(tuner, chain, [prefix])
        before = snapshot(tuner.stats)
        original = deepcopy(prefix)
        f = chain.f_transform
        @test_throws ArgumentError BAT.mcmc_tune_trafo_post_cycle!!(
            f, tuner, chain, chain.proposal, [prefix])
        @test isequal(snapshot(tuner.stats), before)
        @test tuner.iteration == 1 && tuner.scale == 1
        @test chain.f_transform === f
        @test prefix == original
    end
end

function affine_sample(target; nonzero_weights = true, proposal_scale = 1.0)
    algorithm = TransformedMCMC(
        proposal = RandomWalk(proposaldist = Normal(0, proposal_scale)),
        transform_tuning = AdaptiveAffineTuning(), pretransform = DoNotTransform(),
        nchains = 1, nwalkers = 1, nsteps = 16, nonzero_weights = nonzero_weights,
        init = MCMCRetryInit(nsteps_init = 40),
        burnin = MCMCMultiCycleBurnin(nsteps_per_cycle = 20, max_ncycles = 1, nsteps_final = 0),
        convergence = AssumeConvergence(), strict = false
    )
    with_logger(NullLogger()) do
        bat_sample(target, algorithm, BATContext(rng = Philox4x((0x123, 0x456)))).result
    end
end

@testset "AdaptiveAffine recording zero-weight rejections preserves sampling" begin
    target = product_distribution([Uniform(0, 1)])
    filtered = affine_sample(target)
    retained = affine_sample(target; nonzero_weights = false)
    expanded(samples) = [only(samples.v[i]) for i in eachindex(samples) for _ in 1:samples.weight[i]]
    @test expanded(filtered) == expanded(retained)
    @test sum(filtered.weight) == sum(retained.weight) == 16
    @test all(>(0), filtered.weight)
    @test any(iszero, retained.weight)
    @test any(==(-Inf), retained.logd)
    @test all(i -> isfinite(retained.logd[i]) || iszero(retained.weight[i]), eachindex(retained))
end

@testset "AdaptiveAffine is invariant to representable constant score offsets" begin
    offset = -2.0^-1040
    @test Rational{BigInt}(10offset * 0.5) == 5Rational{BigInt}(offset)
    results = map((0.0, -1.0, offset)) do value
        target = PosteriorMeasure(logfuncdensity(x -> value), product_distribution([Uniform(0, 1)]))
        affine_sample(target; proposal_scale = 0.1)
    end
    for result in results
        @test result.v == first(results).v
        @test result.weight == first(results).weight
        @test result.info == first(results).info
        @test sum(result.weight) == 16
    end
end

end # module
