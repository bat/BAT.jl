# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using ArraysOfArrays, ElasticArrays, FillArrays, StructArrays, ValueShapes
using Distributions, JLArrays
import MeasureBase
using MeasureBase: productmeasure, asmeasure, StdUniform, StdNormal


@testset "batch_layout" begin
    μ = productmeasure((
        a = asmeasure(Normal()),
        b = asmeasure(MvNormal([2.0 0.5; 0.5 1.0])),
        c = MeasureBase.Dirac(4.2),
        d = productmeasure((x = StdUniform(), y = StdNormal()^(2, 3))),
    ))
    vs = varshape(μ)
    n = 7

    @testset "structured batches" begin
        X = BAT._own_batch(rand(μ^n), μ)
        @test X.c isa Fill
        @test BAT.batch_varshape(X) == vs

        M = BAT.flatbatch(X)
        @test M == reduce(hcat, [unshaped(x, vs) for x in X])
        @test BAT.flatbatch(BAT.unflatbatch(μ, M)) == M
        @test BAT._batches_equal(==, BAT.unflatbatch(vs, M), X)
        @test_throws DimensionMismatch BAT.unflatbatch(vs, M[1:end-1, :])
    end

    @testset "flat and scalar batches" begin
        V = nestedview(randn(3, n))
        @test BAT.flatbatch(V) === flatview(V)
        @test BAT.batch_varshape(V) == ArrayShape{Real}(3)
        @test BAT.unflatbatch(ArrayShape{Real}(3), flatview(V)) == V

        x = randn(n)
        @test BAT.batch_varshape(x) == ScalarShape{Real}()
        @test BAT.flatbatch(x) == reshape(x, 1, n)
        @test BAT.unflatbatch(ScalarShape{Real}(), BAT.flatbatch(x)) == x

        @test BAT.batch_varshape(Fill([1, 2], n)) == ConstValueShape([1, 2])
        @test size(BAT.flatbatch(Fill(4.2, n))) == (0, n)
    end

    @testset "flat storage types" begin
        # Leaves are plain arrays, also for elastic and offset flat storage:
        vs_flat = NamedTupleShape(a = ScalarShape{Real}(), b = ArrayShape{Real}(2))
        M = ElasticArray(randn(3, n))
        X = BAT.unflatbatch(vs_flat, M)
        @test X.a isa Vector && flatview(X.b) isa Array
        @test X[2] == (a = M[1, 2], b = M[2:3, 2])
        M_offset = view(randn(4, n), Base.IdentityUnitRange(2:4), :)
        @test BAT.flatbatch(BAT.unflatbatch(vs_flat, M_offset)) == parent(M_offset)[2:4, :]
    end

    @testset "reshaping" begin
        X = BAT._own_batch(rand(μ^n), μ)
        V = nestedview(BAT.flatbatch(X))
        @test BAT._batch_with_shape(X, vs) === X
        @test BAT._batch_with_shape(X, ArrayShape{Real}(10)) == V
        @test BAT._batch_with_shape(V, vs) == X
        @test_throws ArgumentError BAT._batch_with_shape(X, NamedTupleShape(b = ArrayShape{Real}(2), a = ArrayShape{Real}(8)))
    end

    @testset "concatenation" begin
        X = BAT._own_batch(rand(μ^n), μ)
        Y = BAT._vcat_batches(X, X)
        @test Y.c isa Fill && length(Y) == 2n
        @test BAT.flatbatch(Y) == hcat(BAT.flatbatch(X), BAT.flatbatch(X))

        X_other = StructArray(merge(StructArrays.components(X), (c = Fill(1.0, n),)))
        Z = BAT._vcat_batches(X, X_other)
        @test !(Z.c isa Fill) && Z.c == vcat(fill(4.2, n), fill(1.0, n))
        @test_throws ArgumentError BAT._vcat_batches(X, StructArray((a = X.a,)))

        @test BAT._vcat_batches(Fill([1.0, 2.0], 2), nestedview([3.0 4.0; 5.0 6.0])) == [[1.0, 2.0], [1.0, 2.0], [3.0, 5.0], [4.0, 6.0]]

        # Differing named tuple and array constants are materialized:
        Z_nt = BAT._vcat_batches(Fill((a = 1.0, b = [1.0, 2.0]), 2), Fill((a = 2.0, b = [3.0, 4.0]), 1))
        @test Z_nt isa StructArray && Z_nt.a == [1.0, 1.0, 2.0] && Z_nt.b == [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]]
        @test BAT._vcat_batches(Fill((a = 1.0,), 2), StructArray((a = [3.0],))).a == [1.0, 1.0, 3.0]
        @test BAT._vcat_batches(Fill([1.0 2.0; 3.0 4.0], 1), Fill(zeros(2, 2), 1)) == [[1.0 2.0; 3.0 4.0], zeros(2, 2)]
        @test_throws ArgumentError BAT._vcat_batches(Fill((a = 1.0,), 2), Fill((b = 1.0,), 2))
        F = Fill([1.0, 2.0], 2)
        @test BAT._vcat_batches(F, F) == Fill([1.0, 2.0], 4) && BAT._vcat_batches(F, F)[1] !== F[1]
    end

    @testset "constants" begin
        μ_c = productmeasure((a = StdNormal(), c = MeasureBase.Dirac([1.0, 2.0])))
        atom = MeasureBase.marginals(μ_c).c.x
        X = rand(μ_c^n)
        Y = BAT._own_batch(X, μ_c)
        @test Y.c isa Fill && Y.c[1] == atom && Y.c[1] !== atom
        @test BAT.unflatbatch(μ_c, BAT.flatbatch(Y)).c[1] !== atom
        # Supplied constants must equal the point mass:
        @test_throws ArgumentError BAT._own_batch(StructArray((a = X.a, c = Fill([5.0, 6.0], n))), μ_c)
        @test_throws ArgumentError BAT._own_batch(StructArray((a = X.a, c = [[5.0, 6.0], (copy(atom) for _ in 2:n)...])), μ_c)
        # Free components that happen to be constant stay free:
        @test BAT._own_batch(StructArray((a = Fill(0.5, n), c = Fill([1.0, 2.0], n))), μ_c).a == fill(0.5, n)
    end

    @testset "growable storage" begin
        @test_throws ArgumentError BAT._check_growable(BAT._own_batch(rand(μ^n), μ))
        @test isnothing(BAT._check_growable(nestedview(ElasticArray(randn(3, n)))))
    end

    @testset "device arrays" begin
        X = StructArray((a = JLArray(randn(n)), b = nestedview(JLArray(randn(2, 3, n)), 2), c = Fill(1.5, n)))
        M = BAT.flatbatch(X)
        @test M isa JLArray && size(M) == (7, n)
        X2 = BAT.unflatbatch(BAT.batch_varshape(X), M)
        @test X2.a isa JLArray && flatview(X2.b) isa JLArray && X2.c isa Fill
        @test Array(BAT.flatbatch(X2)) == Array(M)
        @test BAT._vcat_batches(X, X2).a isa JLArray
    end
end
