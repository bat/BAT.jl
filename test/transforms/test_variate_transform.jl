# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using ValueShapes, Distributions

@testset "variate_transform" begin
    @testset "identity_vt" begin
        @test @inferred(BAT.IdentityVT(ScalarShape{Real}())) isa BAT.VariateTransform{<:ScalarShape,<:ScalarShape}
        @test @inferred(BAT.IdentityVT(ArrayShape{Int}(2))) isa BAT.VariateTransform{<:ArrayShape,<:ArrayShape}
        @test @inferred(BAT.IdentityVT(ArrayShape{Float32}(2, 3))) isa BAT.VariateTransform{<:ArrayShape,<:ArrayShape}

        ntshape = NamedTupleShape(a = ScalarShape{Real}(), b = ArrayShape{Int}(2))
        @test @inferred(BAT.IdentityVT(ntshape)) isa BAT.VariateTransform{<:ValueShapes.NamedTupleShape{(:a, :b)},<:ValueShapes.NamedTupleShape{(:a, :b)}}

        nttrafo = BAT.IdentityVT(ntshape)
        ntvalue = (a = 4.2, b = [5, 7])

        @test @inferred(varshape(nttrafo)) == ntshape
        @test @inferred(inv(nttrafo)) === nttrafo

        @test @inferred(BAT.apply_vartrafo(nttrafo, ntvalue, 0)).v === ntvalue
        @test BAT.apply_vartrafo(nttrafo, ntvalue, 0.79).ladj == 0.79
        @test ismissing(BAT.apply_vartrafo(nttrafo, ntvalue, missing).ladj)
        @test @inferred(BAT.apply_vartrafo(inv(nttrafo), ntvalue, 0.79)).v === ntvalue
        @test BAT.apply_vartrafo(inv(nttrafo), ntvalue, 0.79).ladj == 0.79
        @test ismissing(BAT.apply_vartrafo(inv(nttrafo), ntvalue, missing).ladj)

        @test @inferred(nttrafo ∘ nttrafo) === nttrafo

        arrayshape_trafo = @inferred(BAT.IdentityVT(ArrayShape{Real}(1)))
        hoeldertable_trafo = @inferred((arrayshape_trafo)(HoelderTable(), volcorr=Val(false)))

        @test hoeldertable_trafo.volcorr == @inferred(BAT.TDNoCorr())
        @test hoeldertable_trafo.trafo isa BAT.IdentityVT
    end
end
