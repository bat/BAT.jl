# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestBinnedRNG
using BAT, Test, Random, StableRNGs, Distributions, EmpiricalDistributions
using StatsBase, ValueShapes

# Independently prescribe bin selection and the coordinate within that bin.
# This characterizes the current sampler, not a cross-version draw-count promise.
mutable struct ScriptRNG <: AbstractRNG
    values::Vector{Float64}
    index::Int
end

function Random.rand(rng::ScriptRNG, ::Type{T}) where {T<:AbstractFloat}
    value = rng.values[rng.index]
    rng.index += 1
    return T(value)
end

function fixture(T, closed)
    source = DensitySampleVector([[T(0.25)], [T(1.25)]], zeros(2); weight=[1., 3.])
    return BAT.MarginalDist(source, 1; bins=range(T(0), T(2); length=3), closed).dist
end

function scripted_contract(T, closed, kind)
    dist = fixture(T, closed)
    @test dist isa UvBinnedDist
    stream = ScriptRNG([0.125, 0.25, 0.875, 0.75], 1)
    default_before = copy(Random.default_rng())
    result = if kind == :scalar
        [rand(stream, dist), rand(stream, dist)]
    elseif kind == :bulk
        rand(stream, dist, 2)
    else
        rand!(stream, dist, Vector{T}(undef, 2))
    end
    @test stream.index == 5
    @test result == T[0.25, 1.75]
    @test copy(Random.default_rng()) == default_before
end

function observe(f, make_rng, default_seed)
    Random.seed!(default_seed)
    rng = make_rng(817)
    default_before = copy(Random.default_rng())
    value = f(rng)
    return (value=value, default_unchanged=copy(Random.default_rng()) == default_before,
        continuation=rand(copy(rng), UInt64, 8))
end

function invariance(f, make_rng)
    a = observe(f, make_rng, 311)
    b = observe(f, make_rng, 911)
    c = observe(f, make_rng, 311)
    @test a.default_unchanged && b.default_unchanged && c.default_unchanged
    @test a.value == b.value == c.value
    @test a.continuation == b.continuation == c.continuation
end

@testset "Public binned sampling RNG contract" begin
    saved_default = copy(Random.default_rng())
    try
        @testset "Scripted $T $closed $kind" for T in (Float32, Float64),
                closed in (:left, :right), kind in (:scalar, :bulk, :inplace)
            scripted_contract(T, closed, kind)
        end

        @testset "Public paths and ownership $make_rng $closed" for
                make_rng in (Xoshiro, MersenneTwister, StableRNG), closed in (:left, :right)
            uv = fixture(Float64, closed)
            hist_before = deepcopy(convert(Histogram, uv))
            invariance(rng -> rand(rng, uv), make_rng)
            invariance(rng -> rand(rng, uv, 64), make_rng)
            invariance(rng -> rand!(rng, uv, zeros(64)), make_rng)

            rng = make_rng(817)
            explicit_before = copy(rng)
            default_before = copy(Random.default_rng())
            @test isempty(rand(rng, uv, 0))
            @test isempty(rand!(rng, uv, Float64[]))
            @test rand(copy(rng), UInt64, 8) == rand(explicit_before, UInt64, 8)
            @test copy(Random.default_rng()) == default_before

            # Exercise BAT's actual IID path, including values and log densities.
            invariance(rng -> bat_sample(uv, IIDSampling(nsamples=32), BATContext(rng=rng)).result, make_rng)
            @test convert(Histogram, uv) == hist_before

            source = DensitySampleVector([[0.25, 0.25], [1.25, 1.25]], zeros(2); weight=[1., 3.])
            source_before = deepcopy(source)
            mv = BAT.MarginalDist(source, (1, 2); bins=(0.:1.:2., 0.:1.:2.), closed).dist
            mv_before = deepcopy(convert(Histogram, unshaped(mv)))
            invariance(rng -> rand(rng, mv, 32), make_rng)
            @test convert(Histogram, unshaped(mv)) == mv_before

            shape = NamedTupleShape(x=ScalarShape{Real}(), y=ScalarShape{Real}())
            shaped_source = shape.(source)
            shaped_before = deepcopy(shaped_source)
            shaped = BAT.MarginalDist(shaped_source, (:y, :x); bins=(0.:1.:2., 0.:1.:2.), closed).dist
            shaped_hist_before = deepcopy(convert(Histogram, unshaped(shaped)))
            invariance(rng -> rand(rng, shaped, 32), make_rng)
            @test convert(Histogram, unshaped(shaped)) == shaped_hist_before
            @test shaped_source == shaped_before
            @test source == source_before
        end

        @testset "Implicit RNG remains the default stream" begin
            dist = fixture(Float64, :left)
            Random.seed!(421)
            before = copy(Random.default_rng())
            values = rand(dist, 16)
            @test copy(Random.default_rng()) != before
            Random.seed!(421)
            @test values == rand(dist, 16)
        end
    finally
        copy!(Random.default_rng(), saved_default)
    end
    @test copy(Random.default_rng()) == saved_default
end

end # module
