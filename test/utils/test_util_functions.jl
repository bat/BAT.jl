# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using ForwardDiff


@testset "util_functions" begin
    @test BAT.choose_something(42, 47) === 42
    @test BAT.choose_something(nothing, 47) === 47
    @test BAT.choose_something(missing, missing) === missing

    # `_logaddexp` must be smooth at equal arguments: log(2 cosh(x)) has
    # derivative 0 and second derivative 1 at x = 0:
    for f_tie in (x -> BAT._logaddexp(1 - x, 1 + x), x -> BAT._logaddexp(1 + x, 1 - x))
        @test f_tie(0.0) ≈ 1 + log(2.0)
        @test abs(ForwardDiff.derivative(f_tie, 0.0)) < 1e-14
        @test ForwardDiff.derivative(x -> ForwardDiff.derivative(f_tie, x), 0.0) ≈ 1.0
    end
    @test BAT._logaddexp(-Inf, -Inf) == -Inf
end
