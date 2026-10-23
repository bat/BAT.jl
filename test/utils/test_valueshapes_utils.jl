# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using ValueShapes

using BAT: all_active_names, getstring, asindex, _const_parameters

@testset "valueshapes_utils" begin
    vs = NamedTupleShape(
        x = NamedTupleShape(
            a = ScalarShape{Real}(),
            b = NamedTupleShape(c = ScalarShape{Real}(), d = ArrayShape{Real}(2, 2), k = ConstValueShape(7.0))
        ),
        e = ScalarShape{Real}(),
        f = ArrayShape{Real}(2),
        g = ConstValueShape([1.0, 2.0]),
        m = NamedTupleShape(n = ScalarShape{Real}())
    )

    names = ["x.a", "x.b.c", "x.b.d[1]", "x.b.d[2]", "x.b.d[3]", "x.b.d[4]", "e", "f[1]", "f[2]", "m.n"]
    @test all_active_names(vs) == names
    @test length(names) == totalndof(vs)
    @test [getstring(vs, i) for i in eachindex(names)] == names
    @test _const_parameters(vs) == ["x.b.k" => 7.0, "g" => [1.0, 2.0]]
    @test _const_parameters(ArrayShape{Real}(3)) == []

    @test asindex(vs, :e) == 7
    @test asindex(vs, :(m.n)) == 10
    @test asindex(vs, :m) == 10
    @test asindex(vs, :f) == [8, 9]
    @test asindex(vs, :(f[2])) == 9
    @test asindex(vs, :x) == 1:6
    @test asindex(vs, :(x.b.c)) == 2
    @test asindex(vs, :(x.b.d[2, 1])) == 4
    @test asindex(vs, :(x.b.d[1, :])) == [3, 5]
    @test asindex(vs, :(x.b.d[1:2])) == [3, 4]
    @test asindex(vs, :(x.b.d[1:2])) isa Vector{Int}
    @test asindex(vs, :(f[:])) isa Vector{Int}
    @test asindex(vs, :(x.b.d[1, :])) isa Vector{Int}

    @test_throws ArgumentError asindex(vs, :nope)
    @test_throws ArgumentError asindex(vs, :(x.nope))
    @test_throws ArgumentError asindex(vs, :(e.a))
    @test_throws ArgumentError asindex(vs, :(x[2]))
    @test_throws ArgumentError asindex(vs, :(f[k]))
    @test_throws ArgumentError asindex(vs, :g)
    @test_throws ArgumentError asindex(vs, :(x.b.k))
end
