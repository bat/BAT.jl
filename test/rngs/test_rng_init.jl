# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random, Random123


@testset "rng_init" begin
    @testset "BAT.RNGPartition" begin
        @test @inferred(BAT.RNGPartition(Philox4x(), Base.OneTo(42))) isa BAT.RNGPartition

        _getfields(x::Any) = map(n -> getfield(x, n), fieldnames(typeof(x)))

        rng1 = Philox4x()
        @test BAT.rngpart_getpartctrs(rng1) == (partctrs = (0, 0, 0, 0, 0, 0), depth = 1)
        @test BAT.rngpart_getpartctrs(rng1) == (partctrs = (0, 0, 0, 0, 0, 0), depth = 1)
        prng1 = @inferred BAT.RNGPartition(rng1, 3:3207)
        @test prng1 isa BAT.RNGPartition{<:typeof(rng1)}
        @test _getfields(prng1) == (BAT.rngpart_getseed(rng1), (1, 0, 0, 0, 0, 0), 1, 3:3207)
        @test BAT.rngpart_getpartctrs(rng1) == (partctrs = (3206, 0, 0, 0, 0, 0), depth = 1)

        @test BAT.rngpart_getpartctrs(@inferred AbstractRNG(prng1, 3)) == (partctrs = (1, 0, 0, 0, 0, 0), depth = 2)
        @test BAT.rngpart_getpartctrs(@inferred AbstractRNG(prng1, 3207)) == (partctrs = (3205, 0, 0, 0, 0, 0), depth = 2)
        @test_throws ArgumentError AbstractRNG(prng1, 2)
        @test_throws ArgumentError AbstractRNG(prng1, 3208)
        rng2 = AbstractRNG(prng1, 6)
        @test BAT.rngpart_getpartctrs(rng2) == (partctrs = (4, 0, 0, 0, 0, 0), depth = 2)
        @test all(x -> x != 0, rand(rng1, UInt64, 16) .- rand(rng2, UInt64, 16))
        @test BAT.rngpart_getpartctrs(rng2) == (partctrs = (4, 0, 0, 0, 0, 0), depth = 2)
        @test_throws InexactError BAT.RNGPartition(deepcopy(rng2), -5:800000000000000000)
        @test_throws ArgumentError BAT.RNGPartition(deepcopy(rng2), -5:2147483641)
        prng2 = BAT.RNGPartition(rng2, -5:2147483640)

        rng3 = AbstractRNG(prng2, 2147483640)
        @test BAT.rngpart_getpartctrs(rng3) == (partctrs = (4, 2147483646, 0, 0, 0, 0), depth = 3)
        prng3a = BAT.RNGPartition(rng3, 1:1073741823)
        @test BAT.rngpart_getpartctrs(rng3) == (partctrs = (4, 2147483646, 1073741823+1, 0, 0, 0), depth = 3)
        prng3b = BAT.RNGPartition(rng3, 1:1073741822)
        @test _getfields(prng3b) == (BAT.rngpart_getseed(rng1), (4, 2147483646, 1073741823+2, 0, 0, 0), 3, 1:1073741822)
        @test BAT.rngpart_getpartctrs(rng3) == (partctrs = (4, 2147483646, 1073741823+1073741822+2, 0, 0, 0), depth = 3)
        @test_throws ArgumentError BAT.RNGPartition(deepcopy(rng3), 1:1)

        rng4 = AbstractRNG(prng3b, 1073741822)
        BAT.rngpart_getpartctrs(rng4) == (partctrs = (4, 2147483646, 1073741823+1073741822+1, 0, 0, 0), depth = 4)
        prng4 = BAT.RNGPartition(rng4, 1:1000)

        rng5 = AbstractRNG(prng4, 500)
        prng5 = BAT.RNGPartition(rng5, 1:1000)

        rng6 = AbstractRNG(prng5, 500)
        @test BAT.rngpart_getpartctrs(rng6) == (partctrs = (4, 2147483646, 1073741823+1073741822+1, 500, 500, 0), depth = 6)
        prng6 = BAT.RNGPartition(rng6, 1:1000)
        @test BAT.rngpart_getpartctrs(rng6) == (partctrs = (4, 2147483646, 1073741823+1073741822+1, 500, 500, 1001), depth = 6)

        @test_throws ArgumentError AbstractRNG(prng6, 500)
    end
end
