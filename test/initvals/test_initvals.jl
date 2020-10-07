# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random
using Distributions, PDMats, StatsBase, ValueShapes, ArraysOfArrays


@testset "initvals" begin
    @testset "bat_initval" begin
        du = MvNormal(Diagonal(fill(1.0, 2)))
        @test @inferred(bat_default(bat_initval, Val(:algorithm), du)) == InitFromTarget()
        @test @inferred(bat_initval(du)).result isa Vector{<:AbstractFloat}
        @test bat_initval(du).optargs.algorithm == InitFromTarget()
        @test pdf(du, bat_initval(du).result) > 0
        @test @inferred(bat_initval(du, 10)).result isa VectorOfSimilarVectors{<:AbstractFloat}
        @test all(x -> x > 0, pdf.(Ref(du), bat_initval(du, 10).result))

        ds = NamedTupleDist(a = Normal(), b = Exponential(), c = Uniform(-1, 2))
        @test @inferred(bat_default(bat_initval, Val(:algorithm), du)) == InitFromTarget()
        @test @inferred(bat_initval(ds)).result isa ShapedAsNT
        @test bat_initval(ds).optargs.algorithm == InitFromTarget()
        @test pdf(ds, bat_initval(ds).result) > 0
        @test @inferred(bat_initval(ds, 10)).result isa ShapedAsNTArray
        @test all(x -> x > 0, pdf.(Ref(ds), bat_initval(ds, 10).result))
    end
end
