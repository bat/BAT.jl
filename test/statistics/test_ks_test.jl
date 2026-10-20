# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestWeightedKS

using BAT, Test, Random, Statistics, LogarithmicNumbers

# Exact CDF evaluation at every original coordinate, independent of sorting,
# accumulation, tie grouping and weight normalization in the implementation.
function exact_ks(x, y, wx, wy)
    rx, ry = Rational{BigInt}.(x), Rational{BigInt}.(y)
    a, b = Rational{BigInt}.(wx), Rational{BigInt}.(wy)
    differences = [sum(a[i] for i in eachindex(a) if rx[i] <= t; init=0//big(1)) / sum(a) -
        sum(b[i] for i in eachindex(b) if ry[i] <= t; init=0//big(1)) / sum(b) for t in [rx;ry]]
    dp = max(0, maximum(differences))
    dn = max(0, -minimum(differences))
    Float64.((max(dp,dn), dp, dn))
end

function check_distance(x, y, wx, wy)
    before = deepcopy((x,y,wx,wy))
    observed = BAT.ksstats(x,y,wx,wy)
    expected = exact_ks(x,y,wx,wy)
    @test all(isapprox(a,b; atol=2e-15, rtol=2e-14) for (a,b) in zip(observed,expected))
    @test all(v -> 0 <= v <= 1, observed)
    @test observed[1] == max(observed[2], observed[3])
    swapped = BAT.ksstats(y,x,wy,wx)
    @test all(isapprox(a,b; atol=2e-15, rtol=2e-14) for (a,b) in zip(swapped, (observed[1],observed[3],observed[2])))
    @test isequal((x,y,wx,wy), before)
    observed
end

function samples(x, w; dims=2)
    DensitySampleVector(v=[[Float64(v+j) for j in 1:dims] for v in x],
        logd=zeros(length(x)), weight=w, info=fill(nothing,length(x)), aux=fill(nothing,length(x)))
end

# Independent high-precision Kolmogorov survival series. The positive CDF
# series is stable near zero; the alternating survival series is stable above 1.
function exact_tail(delta, nx, ny)
    setprecision(BigFloat, 256) do
        z = BigFloat(delta) * sqrt(BigFloat(nx) * BigFloat(ny) / (BigFloat(nx)+BigFloat(ny)))
        iszero(z) && return 1.0
        if z < 1
            cdf = sqrt(2*big(pi)) / z * sum(exp(-((2*k-1)^2)*big(pi)^2/(8*z^2)) for k in 1:80)
            Float64(1-cdf)
        else
            Float64(2*sum((-1)^(k-1)*exp(-2*k^2*z^2) for k in 1:80))
        end
    end
end

@testset "Weighted empirical KS distance" begin
    @test BAT.ksstats([0],[0],[1],[1]) == (0,0,0)
    @test BAT.ksstats([0,0],[0],[1,1],[2]) == (0,0,0)
    @test BAT.ksstats([-0.0],[0.0],[1],[1]) == (0,0,0)
    @test BAT.ksstats([0,1],[2,3],fill(1e308,2),fill(1e308,2)) == (1,1,0)
    @test BAT.ksstats([0,1],[2,3],fill(typemax(Int),2),fill(typemax(Int),2)) == (1,1,0)
    @test BAT.ksstats([Int64(2)^53+1],[Float64(2)^53],[1],[1]) == (1,0,1)
    @test BAT.ksstats([big(2)^100+1],[BigFloat(big(2)^100)],[1],[1]) == (1,0,1)

    rng = MersenneTwister(0x2021)
    for _ in 1:100
        x, y = rand(rng,-3:3,rand(rng,1:20)), rand(rng,-3:3,rand(rng,1:20))
        wx, wy = rand(rng,0:9,length(x)), rand(rng,0:9,length(y))
        wx[1] += 1; wy[1] += 1
        expected = check_distance(x,y,wx,wy)
        px, py = randperm(rng,length(x)), randperm(rng,length(y))
        permuted = BAT.ksstats(x[px],y[py],wx[px],wy[py])
        @test all(isapprox(a,b;atol=2e-15,rtol=2e-14) for (a,b) in zip(permuted,expected))
        scaled = BAT.ksstats(x,y,wx .* 2.0^500,wy .* 2.0^-500)
        @test all(isapprox(a,b;atol=2e-15,rtol=2e-14) for (a,b) in zip(scaled,expected))
    end
    for T in (Float16,Float32,Float64)
        check_distance([0,1,1,3],[1,2,3],T[1,2,0,3],T[2,1,4])
        T !== Float64 && check_distance([0,1],[0,2],T[floatmin(T),floatmax(T)],T[1,2])
    end
    original = check_distance([0,1,3],[0,2],[3,2,4],[2,5])
    split = BAT.ksstats([0,0,1,3],[0,2],[1,2,2,4],[2,5])
    @test all(isapprox(a,b;atol=2e-15) for (a,b) in zip(original,split))
    check_distance([0,1],[0,2],BigFloat[1,2],BigFloat[3,4])
    q = typemax(Int)
    rational_weights = [(q-1)//q, 1//q, (q-2)//(q-1), 1//(q-1)]
    check_distance([0,1,2,3],[0,2],rational_weights,[1,1])
    rational_samples = samples([0,1,2,3],rational_weights)
    @test bat_compare(rational_samples,rational_samples;nsamples=:weights).result.nsamples_1 == [2,2]
    setprecision(BigFloat,2) do
        # Literal Float64 oracle: converting a Rational{BigInt} here would
        # itself depend on the low MPFR precision under test.
        @test BAT.ksstats([0,1],[1],[1//3,1//1],[1]) == (.25,.25,0)
        @test_throws ArgumentError BAT.ksstats([0],[1],BigFloat[1],BigFloat[1])
    end
    log_weights = exp.(ULogarithmic, [1000.0,1001.0])
    @test BAT.ksstats([0,1],[2,3],log_weights,log_weights) == (1,1,0)
    # Ordinary no-tie cases agree with the existing dependency's public test.
    x,y = [-3.0,0,2], [-2.0,1,3,5]
    @test first(BAT.ksstats(x,y,ones(3),ones(4))) ≈ BAT.HypothesisTests.ApproximateTwoSampleKSTest(x,y).δ
end

@testset "KS input and numerical admission" begin
    @test_throws DimensionMismatch BAT.ksstats([1],[2],[1,2],[1])
    for bad in (Float64[], [0.0,0.0], [-1.0,2.0], [NaN,1.0], [Inf,1.0])
        @test_throws ArgumentError BAT.ksstats(collect(eachindex(bad)),[2],bad,[1])
    end
    @test_throws ArgumentError BAT.ksstats([NaN,1],[2],[0,1],[1])
    @test_throws ArgumentError BAT.ksstats([Inf],[2],[1],[1])
    @test_throws ArgumentError BAT.ksstats([0,1],[2],[1e-308,1e308],[1])
    setrounding(BigFloat,RoundUp) do
        @test_throws ArgumentError BAT.ksstats([0],[1],BigFloat[1],BigFloat[1])
    end
    previous = get_zero_subnormals()
    try
        set_zero_subnormals(true)
        if get_zero_subnormals()
            @test_throws ArgumentError BAT.ksstats([0],[1],[nextfloat(0.0)],[1])
        end
    finally
        set_zero_subnormals(previous)
    end
end

@testset "Public comparison count policies and tails" begin
    a,b = samples([0,1],ones(2)),samples([2,3,4],ones(3))
    before = deepcopy((a,b))
    t = bat_compare(a,b;nsamples=:length).result
    @test propertynames(t) == (:dim_ind,:nsamples_1,:nsamples_2,:ks_p_values)
    @test t.nsamples_1 == [2,2] && t.nsamples_2 == [3,3]
    @test t.ks_p_values ≈ fill(exact_tail(1,2,3),2)
    reversed = bat_compare(b,a;nsamples=:length).result
    @test reversed.nsamples_1 == t.nsamples_2 && reversed.nsamples_2 == t.nsamples_1
    @test reversed.ks_p_values == t.ks_p_values
    @test isequal((a,b),before)
    for counts in ((7,11),(4_000_000_000,4_000_000_000),(typemax(Int),typemax(Int)))
        for delta in (0.0,1e-6,0.1,1.0)
            @test BAT._ks_comparison_pvalue(delta,counts...) ≈ exact_tail(delta,counts...) atol=2e-15 rtol=2e-13
        end
        ta = samples([0,1],[counts[1],0])
        tb = samples([2,3],[counts[2],0])
        result = bat_compare(ta,tb;nsamples=:weights).result
        @test result.nsamples_1 == fill(counts[1],2)
        @test result.nsamples_2 == fill(counts[2],2)
        @test result.ks_p_values ≈ fill(exact_tail(1,counts...),2)
    end
    # Common weight scaling changes absolute frequency counts, not the ECDF.
    ta,tb = samples([0,1],[1,1]), samples([1,2],[1,1])
    low = bat_compare(ta,tb;nsamples=:weights).result
    high = bat_compare(samples([0,1],[10,10]),samples([1,2],[10,10]);nsamples=:weights).result
    @test low.nsamples_1 == [2,2] && high.nsamples_1 == [20,20]
    @test all(high.ks_p_values .< low.ks_p_values)
    @test high.ks_p_values ≈ fill(exact_tail(.5,20,20),2)

    for (weights,count) in (([.25,.75],1),([2.0^52,.5,.5],2^52+1),
                            ([1//3,2//3],1),(Float32[.5,1.5],2),(BigFloat[.5,.5],1),(BigFloat[big(2)^53+1,.5,.5],2^53+2))
        s = samples(collect(eachindex(weights)),weights)
        result = bat_compare(s,s;nsamples=:weights).result
        @test result.nsamples_1 == fill(count,2)
        @test result.ks_p_values == ones(2)
    end
    for weights in ([1.0,eps()/4],[2.0^52,.5],[.25,.5],fill(1e308,2),
                    [typemax(Int),1],BigFloat[big"0.1",big"0.9"])
        s = samples(collect(eachindex(weights)),weights)
        @test_throws ArgumentError bat_compare(s,s;nsamples=:weights)
    end
    s = samples([0,1],exp.(ULogarithmic,[0.0,1.0]))
    @test_throws ArgumentError bat_compare(s,s;nsamples=:weights)
    @test bat_compare(s,s;nsamples=:length).result.ks_p_values == ones(2)
    huge = samples([0,1],fill(1e308,2))
    @test bat_compare(huge,huge;nsamples=:length).result.ks_p_values == ones(2)

    # Nonuniform, provenance-free weights select scalar Kish ESS.
    s = samples([0,1,2],[1.0,2.0,3.0])
    ess = bat_eff_sample_size(s).result
    @test ess isa Real
    @test bat_compare(s,s).result.nsamples_1 == fill(round(Int,ess),2)
    @test bat_compare(s,s).result.ks_p_values == ones(2)
    nested = bat_compare(s,s;nsamples=:nested,nested_eff_ss=10).result
    @test nested.nsamples_2 == [5,5]
    # Uniform weights retain the existing per-coordinate autocorrelation ESS.
    s = samples([0,2,1,5,3,8,4,2],ones(8))
    ess = bat_eff_sample_size(s).result
    @test ess isa AbstractVector
    @test bat_compare(s,s).result.nsamples_1 == round.(Int,ess)

    @test_throws ArgumentError bat_compare(a,b;nsamples=:unknown)
    for count in (0,-1,NaN,Inf,.1)
        @test_throws ArgumentError bat_compare(a,b;nsamples=:nested,nested_eff_ss=count)
    end
    @test_throws ArgumentError bat_compare(a,b;nsamples=:nested)
    @test_throws ArgumentError bat_compare(a,samples([1,2],[1,1];dims=1))
    @test_throws ArgumentError bat_compare(a[1:0],b)
    zero_dims = samples([1,2],[1,1];dims=0)
    @test_throws ArgumentError bat_compare(zero_dims,zero_dims)
    invalid = samples([1,2],[0,0])
    @test_throws ArgumentError bat_compare(invalid,invalid)
    @test_throws ArgumentError BAT._ks_effective_counts([1.0],2)
    @test_throws ArgumentError BAT._ks_effective_counts([1.0,NaN],2)
    @test_throws ArgumentError BAT._ks_effective_counts(.1,2)
end

end # module
