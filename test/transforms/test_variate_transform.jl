# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using ValueShapes, Distributions

@testset "variate_transform" begin
    @testset "identity_vt" begin
        @test @inferred(BAT.IdentityVT(BAT.InfiniteSpace(), ScalarShape{Real}())) isa BAT.VariateTransform{Univariate,BAT.InfiniteSpace,BAT.InfiniteSpace}
        @test @inferred(BAT.IdentityVT(BAT.UnitSpace(), ArrayShape{Int}(2))) isa BAT.VariateTransform{Multivariate,BAT.UnitSpace,BAT.UnitSpace}
        @test @inferred(BAT.IdentityVT(BAT.MixedSpace(), ArrayShape{Float32}(2, 3))) isa BAT.VariateTransform{Matrixvariate,BAT.MixedSpace,BAT.MixedSpace}

        ntshape = NamedTupleShape(a = ScalarShape{Real}(), b = ArrayShape{Int}(2))
        @test @inferred(BAT.IdentityVT(BAT.MixedSpace(), ntshape)) isa BAT.VariateTransform{<:ValueShapes.NamedTupleVariate{(:a, :b)},BAT.MixedSpace,BAT.MixedSpace}

        nttrafo = BAT.IdentityVT(BAT.MixedSpace(), ntshape)
        ntvalue = (a = 4.2, b = [5, 7])

        @test @inferred(varshape(nttrafo)) == ntshape
        @test @inferred(inv(nttrafo)) === nttrafo

        @test @inferred(BAT.apply_vartrafo(nttrafo, ntvalue, 0)).v === ntvalue
        @test BAT.apply_vartrafo(nttrafo, ntvalue, 0.79).ladj == 0.79
        @test isnan(BAT.apply_vartrafo(nttrafo, ntvalue, NaN).ladj)
        @test @inferred(BAT.apply_vartrafo(inv(nttrafo), ntvalue, 0.79)).v === ntvalue
        @test BAT.apply_vartrafo(inv(nttrafo), ntvalue, 0.79).ladj == 0.79
        @test isnan(BAT.apply_vartrafo(inv(nttrafo), ntvalue, NaN).ladj)

        @test @inferred(nttrafo ∘ nttrafo) === nttrafo
    end
end
