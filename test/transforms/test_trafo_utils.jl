# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using MeasureBase
using ValueShapes, Distributions, ArraysOfArrays
using ForwardDiff, Zygote
using InverseFunctions, ChangesOfVariables
using MeasureBase: StdNormal, StdUniform, TransportFunction, transport_to

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

    f_dt = transport_to(StdNormal()^BAT.some_dof(mu), mu)
    f_hasladj = myidentity ∘ f_dt
    f_plain(x) = (d = sum(x.a) * x.c, e = x.b * x.a) 
    f_complex(x) = (d = sum(x.a) * x.c, e = (f = x.b * x.a,))

    f = f_dt
    @test @inferred(_unshaped_trafo(f)) isa TransportFunction
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
    @test @inferred(_trafo_input_output_shape(f, xs)) isa Tuple{<:NamedTupleShape,<:NamedTupleShape}
    @test @inferred(_trafo_ladj_available(f, xs)) isa Val{false}
    @test @inferred(_trafo_output_numtype(f, xs)) <: Float64
    @test @inferred(_trafo_output_type(f, xs)) <: NamedTuple
    ys = @inferred(transform_samples(f, xs))
    @test ys isa ShapedAsNTArray
    @test @inferred(_get_point_shape(ys)) isa NamedTupleShape
    ysv = @inferred(transform_samples(f, xsv))
    @test ysv.v == ys
    @test all(isnan, ysv.logd)

    xs_complex = f_complex.(xs)
    _trafo_input_output_shape(identity, xs_complex)

    # A transport into a measure of higher precision promotes the number type
    # of the variates:
    xs32 = VectorOfSimilarVectors(rand(Float32, 2, 20))
    f32 = transport_to(batmeasure(MvNormal([0.4, 0.6], [2.0 1.2; 1.2 3.0])), StdUniform()^2)
    @test @inferred(_trafo_output_numtype(f32, xs32)) == Float64
    @test transform_samples(f32, xs32) isa VectorOfSimilarVectors{Float64}

    @testset "nested named tuples" begin
        nested_dist = distprod(x = (a = Uniform(0, 1), b = (c = Normal(), d = MvNormal([1.0 0.0; 0.0 1.0]))), e = Uniform(0, 1))
        nested_vs = varshape(nested_dist)
        nxs = [rand(nested_dist) for _ in 1:3]
        @test @inferred(_get_point_shape(nxs)) isa NamedTupleShape
        @test _get_point_shape(nxs) == valshape(first(nxs))
        # Unshaping explicit values, as the DoNotTransform pretransform does:
        nys = @inferred(transform_samples(Base.Fix2(unshaped, nested_vs), nxs))
        @test nys isa VectorOfSimilarVectors
        @test nys == unshaped.(nxs, Ref(nested_vs))

        snt_dist = NamedTupleDist(ShapedAsNT, x = (a = Uniform(0, 1),), e = Uniform(0, 1))
        sxs = [rand(snt_dist) for _ in 1:3]
        @test sxs isa AbstractVector{<:ShapedAsNT}
        @test @inferred(_get_point_shape(sxs)) isa NamedTupleShape
        @test transform_samples(Base.Fix2(unshaped, varshape(snt_dist)), sxs) == unshaped.(sxs)

        # Union-typed fields of scalars and arrays still carry a shape:
        uxs = NamedTuple{(:a,),Tuple{Union{Float64,Vector{Float64}}}}[(a = 1.0,), (a = 2.0,)]
        @test _get_point_shape(uxs) == NamedTupleShape(a = ScalarShape{Float64}())
        @test transform_samples(x -> (a = x.a + 1,), uxs) == [(a = 2.0,), (a = 3.0,)]

        # Shaped nested outputs from unshaped inputs:
        sys = transform_samples(s -> (x = (n = length(s),),), ["ab", "c"])
        @test sys isa ShapedAsNTArray
        @test sys == [(x = (n = 2,),), (x = (n = 1,),)]

        # Empty named tuples are shaped like any other named tuple:
        exs = transform_samples(x -> (a = NamedTuple(), b = x), [1.0, 2.0])
        @test exs isa ShapedAsNTArray
        @test exs == [(a = NamedTuple(), b = 1.0), (a = NamedTuple(), b = 2.0)]
    end
end
