# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random
using Distributions, PDMats, StatsBase, ValueShapes, ArraysOfArrays, StructArrays
using DensityInterface


@testset "initvals" begin
    context = BATContext()
    @testset "bat_initval" begin
        du = batmeasure(MvNormal(Diagonal(fill(1.0, 2))))
        @test @inferred(bat_default(bat_initval, Val(:algorithm), du)) == InitFromTarget()
        @test @inferred(bat_initval(du, context)).result isa Vector{<:AbstractFloat}
        @test bat_initval(du, context).optargs.algorithm == InitFromTarget()
        @test logdensityof(du, bat_initval(du, context).result) > -Inf
        @test @inferred(bat_initval(du, 10, context)).result isa VectorOfSimilarVectors{<:AbstractFloat}
        @test all(x -> x > -Inf, logdensityof.(Ref(du), bat_initval(du, 10, context).result))
        v_u = BAT.bat_sample(du, BAT.IIDSampling(nsamples=1)).result
        @test BAT.bat_initval(du, BAT.InitFromSamples(v_u), context).result == first(v_u.v)

        ds = batmeasure(NamedTupleDist(a = Normal(), b = Exponential(), c = Uniform(-1, 2)))
        @test @inferred(bat_default(bat_initval, Val(:algorithm), du)) == InitFromTarget()
        @test @inferred(bat_initval(ds, context)).result isa NamedTuple
        @test bat_initval(ds, context).optargs.algorithm == InitFromTarget()
        @test logdensityof(ds, bat_initval(ds, context).result) > -Inf
        @test @inferred(bat_initval(ds, 10, context)).result isa StructVector
        @test all(x -> x > -Inf, logdensityof.(Ref(ds), bat_initval(ds, 10, context).result))
        v_s = BAT.bat_sample(ds, BAT.IIDSampling(nsamples=1)).result
        @test BAT.bat_initval(ds, BAT.InitFromSamples(v_s), context).result == first(v_s.v)

        # Initial values from IID sources are transformed along with the target:
        src = batmeasure(NamedTupleDist(a = Exponential(5.0), b = [Exponential(5.0), Exponential(5.0)]))
        f = BAT.transport_to(BAT.StdNormal()^3, src)
        iid_alg = BAT.apply_trafo_to_init(f, InitFromIID(src))
        xs = bat_initval(BAT.StdNormal()^3, 1000, iid_alg, context).result
        @test xs isa VectorOfSimilarVectors
        @test abs(mean(flatview(xs))) < 0.2 && abs(std(flatview(xs)) - 1) < 0.2

        # Variate types without a dedicated _maycopy_val method (like
        # ShapedAsNT) must pass through instead of raising a MethodError:
        vs_nt = NamedTupleShape(ShapedAsNT, a = ScalarShape{Real}(), b = ArrayShape{Real}(2))
        x_nt = vs_nt(rand(3))
        @test BAT._maycopy_val(x_nt) === x_nt
    end
end
