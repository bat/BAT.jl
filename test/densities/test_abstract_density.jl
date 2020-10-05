# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random
using ArraysOfArrays, ValueShapes, Distributions, PDMats, StatsBase

struct _TestDensityStruct <: AbstractDensity
end

struct _UniformDensityStruct <: AbstractDensity
end

struct _DeltaDensityStruct <: AbstractDensity
end

struct _ShapeDensityStruct <: AbstractDensity
end

@testset "abstract_density" begin
    mvn = MvNormal(ones(3), PDMat(Matrix{Float64}(I,3,3)))
    ValueShapes.totalndof(td::_TestDensityStruct) = Int(3)
    BAT.sampler(td::_TestDensityStruct) = BAT.sampler(mvn)
    BAT.logvalof_unchecked(density::_TestDensityStruct, v::Any) = Distributions.logpdf(mvn, v)

    td = _TestDensityStruct()

    @test @inferred(isequal(@inferred(BAT.var_bounds(td)), missing))
    @test @inferred(isequal(@inferred(varshape(td)), missing))

    x = rand(3)
    @test @inferred(logvalof(td, x)) == @inferred(logpdf(mvn, x))
    @test_throws ErrorException logvalof(td, [Inf, Inf, Inf])
    @test_throws ErrorException logvalof(td, [NaN, NaN, NaN])
    @test @inferred(logvalgradof(td, x)).logd == @inferred(logpdf(mvn, x))

    x = [-Inf, 0, Inf]
    mvu = product_distribution([Uniform() for i in 1:3])
    ValueShapes.totalndof(ud::_UniformDensityStruct) = Int(3)
    BAT.varshape(ud::_UniformDensityStruct) = varshape(mvu)
    BAT.logvalof_unchecked(ud::_UniformDensityStruct, v::Any) = logpdf(mvu, v)
    BAT.var_bounds(ud::_UniformDensityStruct) = BAT.HyperRectBounds(BAT.HyperRectVolume(zeros(3), ones(3)), BAT.BoundsType[BAT.hard_bounds, BAT.hard_bounds, BAT.hard_bounds])
    ud = _UniformDensityStruct()
    lvdg = logvalgradof(ud)

    @test @inferred(logvalof(ud, x)) == -Inf
    @test @inferred(logvalof(ud, x, use_bounds=false)) == -Inf
    @test @inferred((lvdg)(x)) == @inferred(logvalgradof(ud, x))

    @test_throws ArgumentError logvalof(ud, vcat(x,x))

    @test_throws ArgumentError logvalof(ud, x .- eps(1.0), strict=true)

    @test_throws MethodError @inferred(logvalof(ud, [0 0 0]))

    ntshape = NamedTupleShape(a=ScalarShape{Real}(), b=ScalarShape{Real}(), c=ScalarShape{Real}())
    shapedasnt = ShapedAsNT(x, ntshape)

    @test @inferred(logvalgradof(ud, shapedasnt)).logd == @inferred(logvalgradof(ud, x)).logd
    @test @inferred(unshaped(logvalgradof(ud, shapedasnt).grad_logd)) == @inferred(logvalgradof(ud, x)).grad_logd

    cvd = ConstValueDist(0)
    ValueShapes.totalndof(dd::_DeltaDensityStruct) = Int(1)
    BAT.logvalof_unchecked(dd::_DeltaDensityStruct, v::Any) = Distributions.logpdf(cvd, v)

    dd = _DeltaDensityStruct()
    @test_throws ErrorException logvalof(dd, 0)

    ntdist = NamedTupleDist(a=mvn, b=mvu)
    ValueShapes.varshape(sd::_ShapeDensityStruct) = varshape(ntdist)
    BAT.logvalof_unchecked(sd::_ShapeDensityStruct, v) = logpdf(ntdist, v)

    x1_for_sd = rand(length(ntdist.a))
    x2_for_sd = rand(length(ntdist.b))
    x_for_sd = vcat(x1_for_sd, x2_for_sd)

    correct_shape_of_sd = NamedTupleShape(a=ArrayShape{Real}(length(ntdist.a)), b=ArrayShape{Real}(length(ntdist.b)))
    incorrect_shape_of_sd = NamedTupleShape(a=ArrayShape{Real}(length(ntdist.a)-1), b=ArrayShape{Real}(length(ntdist.b)+1))

    x_for_sd_good_shape = correct_shape_of_sd(x_for_sd)
    x_for_sd_bad_shape = incorrect_shape_of_sd(x_for_sd)

    sd = _ShapeDensityStruct()

    @test logvalof(sd, x_for_sd_good_shape) == logpdf(mvn, x1_for_sd) + logpdf(mvu, x2_for_sd)
    @test_throws ArgumentError logvalof(sd, x_for_sd_bad_shape)

    @testset "rand" begin
        td = _TestDensityStruct()
        @test rand(MersenneTwister(7002), sampler(td)) ≈ [-2.415270938, 0.7070171342, 1.0224848653]
    end
end
