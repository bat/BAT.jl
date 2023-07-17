# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random
using Distributions, PDMats, StatsBase, ValueShapes, ArraysOfArrays


@testset "initvals" begin
    context = BATContext()
    @testset "bat_initval" begin
        du = MvNormal(Diagonal(fill(1.0, 2)))
        @test @inferred(bat_default(bat_initval, Val(:algorithm), du)) == InitFromTarget()
        @test @inferred(bat_initval(du, context)).result isa Vector{<:AbstractFloat}
        @test bat_initval(du, context).optargs.algorithm == InitFromTarget()
        @test pdf(du, bat_initval(du, context).result) > 0
        @test @inferred(bat_initval(du, 10, context)).result isa VectorOfSimilarVectors{<:AbstractFloat}
        @test all(x -> x > 0, pdf.(Ref(du), bat_initval(du, 10, context).result))
        v_u = BAT.bat_sample(du, BAT.IIDSampling(nsamples=1)).result
        @test BAT.bat_initval(du, BAT.InitFromSamples(v_u), context).result == first(v_u.v)

        ds = NamedTupleDist(a = Normal(), b = Exponential(), c = Uniform(-1, 2))
        @test @inferred(bat_default(bat_initval, Val(:algorithm), du)) == InitFromTarget()
        @test @inferred(bat_initval(ds, context)).result isa NamedTuple
        @test bat_initval(ds, context).optargs.algorithm == InitFromTarget()
        @test pdf(ds, bat_initval(ds, context).result) > 0
        @test @inferred(bat_initval(ds, 10, context)).result isa ShapedAsNTArray
        @test all(x -> x > 0, pdf.(Ref(ds), bat_initval(ds, 10, context).result))
        v_s = BAT.bat_sample(du, BAT.IIDSampling(nsamples=1)).result
        @test BAT.bat_initval(du, BAT.InitFromSamples(v_s), context).result == first(v_s.v)
    end
end
