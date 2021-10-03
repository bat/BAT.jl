# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random, StableRNGs
using DensityInterface, ValueShapes
using ArraysOfArrays, Distributions, PDMats, StatsBase

struct _TestDensityStruct{T} <: AbstractDensity
    mvn::T
end
DensityInterface.isdensitytype(::Type{<:_TestDensityStruct}) = true
DensityInterface.logdensityof(density::_TestDensityStruct, v::Any) = Distributions.logpdf(density.mvn, v)
ValueShapes.totalndof(td::_TestDensityStruct) = Int(3)
BAT.sampler(td::_TestDensityStruct) = BAT.sampler(td.mvn)

struct _UniformDensityStruct{T} <: AbstractDensity
    mvu::T
end
DensityInterface.isdensitytype(::Type{<:_UniformDensityStruct}) = true
DensityInterface.logdensityof(ud::_UniformDensityStruct, v::Any) = logpdf(ud.mvu, v)
ValueShapes.varshape(ud::_UniformDensityStruct) = varshape(ud.mvu)
ValueShapes.totalndof(ud::_UniformDensityStruct) = Int(3)
BAT.var_bounds(ud::_UniformDensityStruct) = BAT.HyperRectBounds(BAT.HyperRectVolume(zeros(3), ones(3)))

struct _DeltaDensityStruct{T} <: AbstractDensity
    cvd::T
end
DensityInterface.isdensitytype(::Type{<:_DeltaDensityStruct}) = true
DensityInterface.logdensityof(dd::_DeltaDensityStruct, v::Any) = Distributions.logpdf(dd.cvd, v)
ValueShapes.totalndof(dd::_DeltaDensityStruct) = Int(1)

struct _ShapeDensityStruct{T} <: AbstractDensity
    ntdist::T
end
DensityInterface.isdensitytype(::Type{<:_ShapeDensityStruct}) = true
DensityInterface.logdensityof(sd::_ShapeDensityStruct, v) = logpdf(sd.ntdist, v)
ValueShapes.varshape(sd::_ShapeDensityStruct) = varshape(sd.ntdist)


struct NonBATDensity end
DensityInterface.isdensitytype(::Type{<:NonBATDensity}) = true
DensityInterface.logdensityof(d::NonBATDensity, v) = log(norm(v)^2)
ValueShapes.varshape(d::NonBATDensity) = ArrayShape{Real}(2)


@testset "abstract_density" begin
    mvn = MvNormal(ones(3), PDMat(Matrix{Float64}(I,3,3)))
    td = _TestDensityStruct(mvn)
    tds = BAT.DensityWithShape(td, ArrayShape{Real}(3))

    @test @inferred(isequal(@inferred(BAT.var_bounds(td)), missing))
    @test @inferred(isequal(@inferred(varshape(td)), missing))

    x = rand(3)
    @test_throws ArgumentError BAT.checked_logdensityof(td, [Inf, Inf, Inf])
    @test_throws BAT.DensityEvalException BAT.checked_logdensityof(tds, [Inf, Inf, Inf])
    @test_throws BAT.DensityEvalException BAT.checked_logdensityof(tds, [Inf, Inf, Inf])
    @test_throws BAT.DensityEvalException BAT.checked_logdensityof(tds, [NaN, NaN, NaN])
    @test_throws ArgumentError BAT.checked_logdensityof(tds, rand(length(mvn)+1))
    @test_throws ArgumentError BAT.checked_logdensityof(tds, rand(length(mvn)-1))

    @test @inferred(BAT.checked_logdensityof(tds, x)) == @inferred(logpdf(mvn, x))

    x = [-Inf, 0, Inf]
    ud_shape_1 = NamedTupleShape(a=ArrayShape{Real}(1), b=ArrayShape{Real}(1), c=ArrayShape{Real}(1))
    ud_shape_2 = NamedTupleShape(a=ArrayShape{Real}(3))
    mvu = product_distribution([Uniform() for i in 1:3])
    ud = _UniformDensityStruct(mvu)

    @test @inferred(BAT.checked_logdensityof(ud, x)) == -Inf
    @test @inferred(BAT.checked_logdensityof(ud_shape_1(ud), ud_shape_1(x))) == @inferred(logpdf(mvu, x))
    @test @inferred(BAT.checked_logdensityof(ud_shape_2(ud), ud_shape_2(x))) == @inferred(logpdf(mvu, x))

    @test_throws ArgumentError BAT.checked_logdensityof(ud, vcat(x,x))

    @test BAT.checked_logdensityof(ud, x .- eps(1.0)) == -Inf

    @test_throws ArgumentError @inferred(BAT.checked_logdensityof(ud, [0 0 0]))

    ntshape = NamedTupleShape(a=ScalarShape{Real}(), b=ScalarShape{Real}(), c=ScalarShape{Real}())
    shapedasnt = ShapedAsNT(x, ntshape)

    cvd = ConstValueDist(0)
    dd = _DeltaDensityStruct(cvd)
    dds = BAT.DensityWithShape(dd, ScalarShape{Real}())
    @test_throws ArgumentError BAT.checked_logdensityof(dd, 0)
    @test_throws BAT.DensityEvalException BAT.checked_logdensityof(dds, 0)

    ntdist = NamedTupleDist(a=mvn, b=mvu)

    x1_for_sd = rand(length(ntdist.a))
    x2_for_sd = rand(length(ntdist.b))
    x_for_sd = vcat(x1_for_sd, x2_for_sd)

    correct_shape_of_sd = NamedTupleShape(a=ArrayShape{Real}(length(ntdist.a)), b=ArrayShape{Real}(length(ntdist.b)))
    incorrect_shape_of_sd = NamedTupleShape(a=ArrayShape{Real}(length(ntdist.a)-1), b=ArrayShape{Real}(length(ntdist.b)+1))

    x_for_sd_good_shape = correct_shape_of_sd(x_for_sd)
    x_for_sd_bad_shape = incorrect_shape_of_sd(x_for_sd)

    sd = _ShapeDensityStruct(ntdist)

    @test BAT.checked_logdensityof(sd, x_for_sd_good_shape) == logpdf(mvn, x1_for_sd) + logpdf(mvu, x2_for_sd)
    @test_throws ArgumentError BAT.checked_logdensityof(sd, x_for_sd_bad_shape)

    @testset "rand" begin
        td = _TestDensityStruct(mvn)
        @test rand(StableRNG(7002), BAT.bat_sampler(td)) ≈ [2.386799038, 1.072161895, 0.791486531]
    end

    @testset "non-BAT densities" begin
        d = NonBATDensity()
        x = randn(3)
        @test @inferred(convert(AbstractDensity, d)) isa BAT.WrappedNonBATDensity
        bd = convert(AbstractDensity, d)
        @test @inferred(logdensityof(bd, x)) == logdensityof(d, x)
        @test @inferred(logdensityof(bd)) == logdensityof(d)
        @test @inferred(varshape(bd)) == varshape(d)
    end
end
