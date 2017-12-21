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

end
