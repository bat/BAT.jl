# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT, Test, StatsBase, LinearAlgebra

# Independent exact batch formulas; no online accumulator or StatsBase moment
# routine is used to construct expected results.
function reweight_batch_oracle(data, weights, W)
    x = Rational{BigInt}.(data)
    w = Rational{BigInt}.(weights)
    total = sum(w)
    mu = x * w / total
    scatter = sum(w[i] * (x[:,i] - mu) * (x[:,i] - mu)' for i in eachindex(w))
    n = count(!iszero, w)
    denominator = W === Weights ? total : W === AnalyticWeights ?
        total - sum(abs2, w) / total : W === FrequencyWeights ?
        total - 1 : total * (n - 1) / n
    return Float64.(mu), Float64.(scatter / denominator)
end

# Inspect storage, including counters/scratch/metadata, rather than only the
# exposed mean or covariance. Arrays in these structures have plain storage.
reweight_storage(x::Number) = x
reweight_storage(x::Array) = map(reweight_storage, x)
reweight_storage(x) = map(n -> reweight_storage(getfield(x, n)), fieldnames(typeof(x)))

@testset "online reweighting" begin
    x = [1 3 -2 4 7; 5 -1 2 8 -3]
    w = [2, 3, 1, 4, 2]
    @testset "batch and incremental oracles $T $W f=$f" for T in (Float32, Float64),
        W in (Weights, AnalyticWeights, FrequencyWeights, ProbabilityWeights), f in (0.5, 1.0, 2.0)
        uv = BAT.BasicUvStatistics{T,W}()
        mv = BAT.BasicMvStatistics{T,W}(2)
        for i in 1:3
            push!(uv, x[1,i], w[i])
            push!(mv, x[:,i], w[i])
        end
        uv_before, mv_before = reweight_storage.((uv, mv))
        mean_alias, cov_alias, S_alias = mv.mean, mv.cov, mv.cov.S
        @test BAT.reweight_relative!(uv, f) === uv
        @test BAT.reweight_relative!(mv, f) === mv
        @test mv.mean === mean_alias && mv.cov === cov_alias && mv.cov.S === S_alias
        f == 1 && @test isequal((reweight_storage(uv), reweight_storage(mv)), (uv_before, mv_before))
        expected_mean, expected_cov = reweight_batch_oracle(x[:,1:3], f .* w[1:3], W)
        @test uv.mean[] ≈ expected_mean[1]
        @test uv.var[] ≈ expected_cov[1,1]
        @test Vector(mv.mean) ≈ expected_mean
        @test Matrix(mv.cov) ≈ expected_cov
        @test mv.cov.n == uv.var.n == 3
        @test Float64(mv.cov.sum_w2) ≈ sum(abs2, f .* w[1:3])
        @test (uv.minimum, uv.maximum) == (-2, 3)
        @test mv.minimum == [-2, -1] && mv.maximum == [3, 5]

        tail_uv, tail_mv = BAT.BasicUvStatistics{T,W}(), BAT.BasicMvStatistics{T,W}(2)
        for i in 4:5
            push!(tail_uv, x[1,i], w[i])
            push!(tail_mv, x[:,i], w[i])
        end
        merged_uv, merged_mv = merge(uv, tail_uv), merge(mv, tail_mv)
        for i in 4:5
            push!(uv, x[1,i], w[i])
            push!(mv, x[:,i], w[i])
        end
        expected_mean, expected_cov = reweight_batch_oracle(x, [f .* w[1:3]; w[4:5]], W)
        for (a,b) in ((uv,mv), (merged_uv,merged_mv))
            @test a.mean[] ≈ expected_mean[1]
            @test a.var[] ≈ expected_cov[1,1]
            @test Vector(b.mean) ≈ expected_mean
            @test Matrix(b.cov) ≈ expected_cov
        end
    end

    @testset "fresh reset and invalid factors $T $W" for T in (Float32, Float64),
        W in (Weights, AnalyticWeights, FrequencyWeights, ProbabilityWeights)
        constructors = (() -> BAT.OnlineUvMean{T}(), () -> BAT.OnlineUvVar{T,W}(),
            () -> BAT.BasicUvStatistics{T,W}(), () -> BAT.OnlineMvMean{T}(2),
            () -> BAT.OnlineMvCov{T,W}(2), () -> BAT.BasicMvStatistics{T,W}(2))
        for (i, make) in enumerate(constructors)
            stats = make()
            scalar = i <= 3
            for value in (T(1e16), T(1e16))
                push!(stats, scalar ? value : [value, -value])
            end
            for bad in (-1, NaN, Inf, -Inf, big"1e1000", big"1e-1000")
                before = deepcopy(reweight_storage(stats))
                @test_throws ArgumentError BAT.reweight_relative!(stats, bad)
                @test isequal(reweight_storage(stats), before)
            end
            for factor in (0, -0.0)
                @test BAT.reweight_relative!(stats, factor) === stats
                @test isequal(reweight_storage(stats), reweight_storage(make()))
            end
            fresh = make()
            for value in (1, 3, 1, 3)
                datum = scalar ? value : [value, -value]
                push!(stats, datum)
                push!(fresh, datum)
            end
            @test isequal(reweight_storage(stats), reweight_storage(fresh))
            empty!(stats)
            @test isequal(reweight_storage(stats), reweight_storage(make()))
        end
    end

    @testset "MCMC aggregate reset and failure atomicity" begin
        stats = BAT.MCMCBasicStats{Float64,Float64}(2)
        for (v,l) in (([1e16,-1e16], 20.0), ([1e16+2,-1e16+2], 10.0))
            push!(stats, DensitySample(v, l, 1, nothing, nothing))
        end
        aliases = (stats.param_stats, stats.logtf_stats, stats.mode, stats.param_stats.mean.S,
            stats.param_stats.cov.S, stats.param_stats.maximum)
        @test BAT.reweight_relative!(stats, .5) === stats
        @test stats.logtf_stats.mean[] == 15
        @test stats.mode == [1e16,-1e16]
        before = deepcopy(reweight_storage(stats))
        @test BAT.reweight_relative!(stats, 1) === stats
        @test isequal(reweight_storage(stats), before)
        # A late failing component must not partially reweight parameter stats.
        stats.logtf_stats.var.s = floatmax(Float64) / 2
        before = deepcopy(reweight_storage(stats))
        @test_throws ArgumentError BAT.reweight_relative!(stats, 4)
        @test isequal(reweight_storage(stats), before)
        @test BAT.reweight_relative!(stats, 0) === stats
        @test isequal(reweight_storage(stats), reweight_storage(BAT.MCMCBasicStats{Float64,Float64}(2)))
        @test all(a === b for (a,b) in zip(aliases, (stats.param_stats, stats.logtf_stats,
            stats.mode, stats.param_stats.mean.S, stats.param_stats.cov.S, stats.param_stats.maximum)))
        fresh = BAT.MCMCBasicStats{Float64,Float64}(2)
        for value in (1.0, 3.0)
            s = DensitySample([value, -value], -value, 1, nothing, nothing)
            push!(stats, s); push!(fresh, s)
        end
        @test isequal(reweight_storage(stats), reweight_storage(fresh))
        @test Matrix(stats.param_stats.cov) ≈ [2 -2; -2 2]
        @test stats.mode == [1,-1]
    end

    @testset "zero-weight scores preserve moments and diagnostic metadata" begin
        samples = DensitySampleVector([[1.0], [9.0], [8.0]], [-1.0, -Inf, 20.0]; weight = [2, 0, 0])
        for accumulate! in ((s, xs) -> foreach(x -> push!(s, x), xs), append!)
            stats = BAT.MCMCBasicStats{Float64,Float64}(1)
            accumulate!(stats, samples)
            @test stats.logtf_stats.mean[] == -1
            @test stats.logtf_stats.var.n == 1
            @test stats.logtf_stats.var.s == 0
            @test (stats.logtf_stats.minimum, stats.logtf_stats.maximum) == (-Inf, 20)
            @test stats.mode == [8.0]
            @test stats.param_stats.mean[1] == 1
            @test stats.param_stats.minimum == [1.0]
            @test stats.param_stats.maximum == [9.0]
            @test BAT.reweight_relative!(stats, 0.5) === stats
            @test stats.logtf_stats.mean[] == -1
            @test stats.logtf_stats.mean.sum_w == 1
            @test (stats.logtf_stats.minimum, stats.logtf_stats.maximum) == (-Inf, 20)
            @test stats.mode == [8.0]
        end
    end

    @testset "subnormal MCMC score moments $T" for T in (Float32, Float64)
        offset = -ldexp(one(T), T === Float32 ? -135 : -1040)
        stats = BAT.MCMCBasicStats{T,T}(1)
        for i in 1:10
            push!(stats, DensitySample(T[i], offset, 1, nothing, nothing))
        end
        original = reweight_storage(stats)
        # The exception is specific to MCMC's auxiliary moments. A standalone
        # statistic retains its existing normal-range policy.
        @test_throws ArgumentError BAT.reweight_relative!(stats.logtf_stats, 0.5)
        @test isequal(reweight_storage(stats), original)
        @test BAT.reweight_relative!(stats, 0.5) === stats
        @test stats.logtf_stats.mean.sum_v == 5offset
        @test stats.logtf_stats.mean[] == offset
        @test stats.logtf_stats.mean.sum_w == stats.logtf_stats.var.sum_w == 5
        @test stats.logtf_stats.var.sum_w2 == 2.5
        @test stats.param_stats.mean[1] == T(5.5)

        # Nonconstant auxiliary scores can also have representable subnormal
        # scatter; parameter geometry is at an ordinary scale.
        amplitude = ldexp(one(T), T === Float32 ? -67 : -520)
        scatter_stats = BAT.MCMCBasicStats{T,T}(1)
        for (i, score) in enumerate(T[-amplitude, amplitude, -amplitude, amplitude])
            push!(scatter_stats, DensitySample(T[i], score, 1, nothing, nothing))
        end
        scatter = scatter_stats.logtf_stats.var.s
        @test 0 < scatter < floatmin(T)
        @test Rational{BigInt}(scatter * T(0.5)) == Rational{BigInt}(scatter) / 2
        @test BAT.reweight_relative!(scatter_stats, 0.5) === scatter_stats
        @test scatter_stats.logtf_stats.var.s == scatter * T(0.5)
    end

    @testset "MCMC score admission preserves weight and parameter guards" begin
        for corrupt! in (
            s -> (s.param_stats.mean.S[1] = floatmin(Float64)),
            s -> (s.param_stats.cov.S[1, 1] = floatmin(Float64)),
            s -> (s.logtf_stats.mean.sum_w = floatmin(Float64)),
            s -> (s.logtf_stats.var.sum_w = floatmin(Float64)),
            s -> (s.logtf_stats.var.sum_w2 = floatmin(Float64)),
            s -> (s.logtf_stats.mean.sum_v = nextfloat(0.0)),
            s -> (s.logtf_stats.var.s = nextfloat(0.0)),
            s -> (s.logtf_stats.mean.sum_v = NaN),
            s -> (s.logtf_stats.var.s = Inf),
        )
            stats = BAT.MCMCBasicStats{Float64,Float64}(1)
            push!(stats, DensitySample([1.0], -1.0, 2, nothing, nothing))
            corrupt!(stats)
            before = deepcopy(reweight_storage(stats))
            @test_throws ArgumentError BAT.reweight_relative!(stats, 0.5)
            @test isequal(reweight_storage(stats), before)
        end
    end

    @testset "range boundaries and compensated moments" begin
        previous_mode = get_zero_subnormals()
        guarded = push!(BAT.OnlineUvMean(), 2.0)
        before_guard = reweight_storage(guarded)
        try
            set_zero_subnormals(true)
            if get_zero_subnormals()
                @test_throws ArgumentError BAT.reweight_relative!(guarded, nextfloat(0.0))
                @test isequal(reweight_storage(guarded), before_guard)
            end
        finally
            set_zero_subnormals(previous_mode)
        end
        for T in (Float32, Float64)
            stats = BAT.OnlineMvMean{T}(1)
            push!(stats, [one(T)])
            stats.C[1] = nextfloat(zero(T))
            BAT.reweight_relative!(stats, .5)
            @test stats[1] == 1 # negligible compensation may round away
            @test iszero(stats.C[1])
            stats.S[1], stats.C[1] = zero(T), T(2)
            BAT.reweight_relative!(stats, .5)
            @test stats[1] == 4 # correction-only first moment survives
            stats.S[1], stats.C[1] = nextfloat(zero(T)), zero(T)
            before = deepcopy(reweight_storage(stats))
            @test_throws ArgumentError BAT.reweight_relative!(stats, .5)
            @test isequal(reweight_storage(stats), before)
            stats.S[1] = floatmax(T)
            before = deepcopy(reweight_storage(stats))
            @test_throws ArgumentError BAT.reweight_relative!(stats, 2)
            @test isequal(reweight_storage(stats), before)
        end
        # A nonzero subnormal Q can lose substantial relative accuracy: the
        # old finite/nonzero-only check admitted variance 1.8 instead of 2.
        for make in (() -> BAT.OnlineUvVar{Float64,AnalyticWeights}(),
                     () -> BAT.OnlineMvCov{Float64,AnalyticWeights}(1))
            stats = make()
            scalar = stats isa BAT.OnlineUvVar
            for value in (0.0, 2.0)
                push!(stats, scalar ? value : [value])
            end
            before = deepcopy(reweight_storage(stats))
            @test_throws ArgumentError BAT.reweight_relative!(stats, 3 * 2.0^-539)
            @test isequal(reweight_storage(stats), before)
        end
        # A subnormal intermediate Q*f may lose bits which a large second
        # multiplication would amplify, even if the final Q is normal.
        partial = BAT.OnlineUvVar{Float64,AnalyticWeights}()
        partial.sum_w, partial.sum_w2 = 1.0, nextfloat(0.0)
        before = deepcopy(reweight_storage(partial))
        @test_throws ArgumentError BAT.reweight_relative!(partial, 2.0^26 + 0.5)
        @test isequal(reweight_storage(partial), before)
        mixed = BAT.MCMCBasicStats{Float32,Float64}(1)
        push!(mixed, DensitySample([1.0], 1.0f0, 1, nothing, nothing))
        before = deepcopy(reweight_storage(mixed))
        @test_throws ArgumentError BAT.reweight_relative!(mixed, 1e100)
        @test isequal(reweight_storage(mixed), before)
        # Both final Q values are representable although forming f² is not.
        for (weight, square, factor, expected) in ((1e-150,1e-300,1e200,1e100),
                                                  (1e150,1e300,1e-200,1e-100))
            stats = BAT.OnlineUvVar{Float64,AnalyticWeights}()
            stats.sum_w, stats.sum_w2 = weight, square
            BAT.reweight_relative!(stats, factor)
            @test Float64(stats.sum_w2) ≈ expected
        end
        stats = BAT.BasicUvStatistics{Float64,AnalyticWeights}()
        push!(stats, 1.0)
        stats.var.sum_w2 = nextfloat(0.0)
        before = deepcopy(reweight_storage(stats))
        @test_throws ArgumentError BAT.reweight_relative!(stats, .5)
        @test isequal(reweight_storage(stats), before)
    end
end
