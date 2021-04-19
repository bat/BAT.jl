# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using StaticArrays, ForwardDiff

@testset "forwarddiff_utils" begin
    @test @inferred(ForwardDiff.value(BAT.forwarddiff_eval(sin, 0.5))) == sin(0.5)
    @test @inferred(ForwardDiff.partials(BAT.forwarddiff_eval(sin, 0.5))[1]) == cos(0.5)
    @test length(ForwardDiff.partials(BAT.forwarddiff_eval(sin, 0.5))) == 1

    @test @inferred(BAT.forwarddiff_vjp(0.7, BAT.forwarddiff_eval(sin, 0.5))) == 0.7 * ForwardDiff.derivative(sin, 0.5)


    f(x) = SVector(1.1 * x[1] + 2.1 * x[2] + 3.1 * x[3], -1.2 * x[1] - 2.2 * x[2] + -3.2 * x[3])

    x = SVector(0.1, 0.2, 0.3)
    ΔΩ = SVector((10, 20))

    @test @inferred(BAT.forwarddiff_vjp(ΔΩ, BAT.forwarddiff_eval(f, x))) == ForwardDiff.jacobian(f, x)' * ΔΩ 

    @test @inferred((x -> BAT.forwarddiff_pullback(f, x)[1])(x)) == f(x)
    @test @inferred(BAT.forwarddiff_pullback(f, x)[2](ΔΩ)) == ForwardDiff.jacobian(f, x)' * ΔΩ

    X = [SVector(0.1i, 0.2i, 0.3i) for i in 1:7]
    ΔΩs = [SVector(10i, 20i) for i in 1:7]
    @test @inferred((X -> BAT.forwarddiff_broadcast_pullback(Ref(f), X)[1])(X)) == broadcast(f, X)
    @test @inferred(BAT.forwarddiff_broadcast_pullback(f, X)[2](ΔΩs)) == broadcast((f, x, ΔΩ) -> ForwardDiff.jacobian(f, x)' * ΔΩ, Ref(f), X, ΔΩs)
end
