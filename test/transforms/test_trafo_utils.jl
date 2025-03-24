# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using MeasureBase
using ValueShapes, Distributions, ArraysOfArrays
using ForwardDiff, Zygote, DistributionsAD
using InverseFunctions, ChangesOfVariables

using BAT: transform_samples

using BAT: _unshaped_trafo, _get_point_shape, _trafo_input_output_shape, _trafo_ladj_available,
    _trafo_output_numtype, _trafo_output_type, _trafo_create_unshaped_ys

@testset "test_trafo_utils" begin
    dist = distprod(
        a = MvNormal([0.4, 0.6], [2.0 1.2; 1.2 3.0]),
        b = Exponential(2.3),
        c = 4.2,
    )
    mu = batmeasure(dist)

    xsv = bat_sample(dist, IIDSampling(nsamples = 20)).result
    xs = xsv.v
    x = first(xs)
    logd_xs = xsv.logd

    @test @inferred(_get_point_shape(xs)) isa NamedTupleShape

    myidentity(x) = x
    InverseFunctions.inverse(f::typeof(myidentity)) = f
    ChangesOfVariables.with_logabsdet_jacobian(::typeof(myidentity), x) = x, Bool(false)

    f_dt = BAT.DistributionTransform(Normal, dist)
    f_hasladj = myidentity ∘ f_dt
    f_plain(x) = (d = sum(x.a) * x.c, e = x.b * x.a) 
    f_complex(x) = (d = sum(x.a) * x.c, e = (f = x.b * x.a,))

    f = f_dt
    @test @inferred(_unshaped_trafo(f)) isa BAT.DistributionTransform
    @test @inferred(_trafo_input_output_shape(f, xs)) isa Tuple{<:NamedTupleShape,<:ArrayShape}
    x_shape, y_shape = _trafo_input_output_shape(f, xs)
    @test @inferred(_trafo_ladj_available(f, xs)) isa Val{true}
    @test @inferred(_trafo_output_numtype(f, xs)) <: Float64
    @test @inferred(_trafo_output_type(f, xs)) <: AbstractVector{Float64}
    @test @inferred(_trafo_create_unshaped_ys(f, xs, y_shape)) isa VectorOfSimilarVectors{Float64}
    ys = @inferred(transform_samples(f, xs))
    @test ys isa VectorOfSimilarVectors{Float64}
    @test @inferred(_get_point_shape(ys)) isa ArrayShape
    ysv = @inferred(transform_samples(f, xsv))
    @test ysv.v == ys
    @test !any(isnan, ysv.logd)
    @test logdensityof(pushfwd(f, mu)).(ys) ≈ ysv.logd

    f = f_hasladj
    @test @inferred(_unshaped_trafo(f)) isa Missing
    @test @inferred(_trafo_input_output_shape(f, xs)) isa Tuple{<:NamedTupleShape,<:ArrayShape}
    x_shape, y_shape = _trafo_input_output_shape(f, xs)
    @test @inferred(_trafo_ladj_available(f, xs)) isa Val{true}
    @test @inferred(_trafo_output_numtype(f, xs)) <: Float64
    @test @inferred(_trafo_output_type(f, xs)) <: AbstractVector{Float64}
    @test @inferred(_trafo_create_unshaped_ys(f, xs, y_shape)) isa VectorOfSimilarVectors{Float64}
    ys = @inferred(transform_samples(f, xs))
    @test ys isa VectorOfSimilarVectors{Float64}
    @test @inferred(_get_point_shape(ys)) isa ArrayShape
    ysv = @inferred(transform_samples(f, xsv))
    @test ysv.v == ys
    @test !any(isnan, ysv.logd)
    @test logdensityof(pushfwd(f, mu)).(ys) ≈ ysv.logd

    f = f_plain
    @test @inferred(_unshaped_trafo(f)) isa Missing
    @test @inferred(_trafo_input_output_shape(f, xs)) isa Tuple{<:NamedTupleShape,<:NamedTupleShape}
    x_shape, y_shape = _trafo_input_output_shape(f, xs)
    @test @inferred(_trafo_ladj_available(f, xs)) isa Val{false}
    @test @inferred(_trafo_output_numtype(f, xs)) <: Float64
    @test @inferred(_trafo_output_type(f, xs)) <: NamedTuple
    @test @inferred(_trafo_create_unshaped_ys(f, xs, y_shape)) isa VectorOfSimilarVectors{Float64}
    ys = @inferred(transform_samples(f, xs))
    @test ys isa ShapedAsNTArray
    @test @inferred(_get_point_shape(ys)) isa NamedTupleShape
    ysv = @inferred(transform_samples(f, xsv))
    @test ysv.v == ys
    @test all(isnan, ysv.logd)

    f = f_complex
    @test @inferred(_unshaped_trafo(f)) isa Missing
    @test @inferred(_trafo_input_output_shape(f, xs)) isa Tuple{<:NamedTupleShape,Missing}
    @test @inferred(_trafo_ladj_available(f, xs)) isa Val{false}
    @test @inferred(_trafo_output_numtype(f, xs)) <: Float64
    @test @inferred(_trafo_output_type(f, xs)) <: NamedTuple
    ys = @inferred(transform_samples(f, xs))
    @test ys isa AbstractVector{<:NamedTuple}
    @test @inferred(_get_point_shape(ys)) isa Missing
    ysv = @inferred(transform_samples(f, xsv))
    @test ysv.v == ys
    @test all(isnan, ysv.logd)

    xs_complex = f_complex.(xs)
    _trafo_input_output_shape(identity, xs_complex)
end
