# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Compat.Test

@testset "rng" begin
    @testset "Philox4xSeed" begin
        philox = @inferred Philox4xSeed()
        @test typeof(philox) <: AbstractRNGSeed
        @test typeof(philox.seed) <: NTuple{2, UInt}
        philox = @inferred Philox4xSeed{UInt32}(tuple(ones(UInt32,2)...))
        @test typeof(philox.seed) <: NTuple{2, UInt32}
        philox = @inferred Philox4xSeed{UInt64}(tuple(UInt64(1),UInt64(2)))
        res = @inferred philox()
        @test res.key1 == UInt64(1) && res.key2 == UInt64(2)
    end

    @testset "Threefry4xSeed" begin
        tf4x = @inferred Threefry4xSeed()
        @test typeof(tf4x) <: AbstractRNGSeed
        @test typeof(tf4x.seed) <: NTuple{4, UInt}
        tf4x = @inferred Threefry4xSeed{UInt32}(tuple(ones(UInt32,4)...))
        @test typeof(tf4x.seed) <: NTuple{4, UInt32}
        tf4x = @inferred Threefry4xSeed{UInt64}(tuple(UInt64(2):UInt64(5)...))
        res = @inferred tf4x()
        @test res.key1 == UInt64(2) && res.key2 == UInt64(3) &&
              res.key3 == UInt64(4) && res.key4 == UInt64(5)
    end
end
