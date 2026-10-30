# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using MeasureBase
using ValueShapes, Distributions, ArraysOfArrays
using ForwardDiff, Zygote, DistributionsAD
using InverseFunctions, ChangesOfVariables
using Random

using BAT: transform_samples

using BAT: _unshaped_trafo, _get_point_shape, _trafo_input_output_shape, _trafo_ladj_available,
    _trafo_output_numtype, _trafo_output_type, _trafo_create_unshaped_ys

# Fixture-owned transform: mathematical output is deterministic, but its
# ordinary caller-owned draws and forks must not be hidden by RNG restoration.
mutable struct SerialTrafoEffects <: Function
    rows::Vector{Float64}
    callers::Vector{Task}
    draws::Vector{UInt64}
    children::Vector{Task}
    child_draws::Vector{Vector{UInt64}}
    fail_at::Int
    failure::ErrorException
end
SerialTrafoEffects(fail_at=0) = SerialTrafoEffects(Float64[], Task[], UInt64[],
    Task[], Vector{UInt64}[], fail_at, ErrorException("intentional serial transform failure"))

function (f::SerialTrafoEffects)(x::AbstractVector{<:Real})
    push!(f.rows, only(x))
    push!(f.callers, current_task())
    push!(f.draws, rand(UInt64))
    task = @task rand(UInt64, 4)
    push!(f.children, task)
    schedule(task)
    push!(f.child_draws, fetch(task))
    length(f.rows) == f.fail_at && throw(f.failure)
    return 2 .* x
end

# Known shape/Jacobian avoid incidental inference probes in this fixture. No
# production method is replaced: both methods apply only to our callable type.
ValueShapes.resultshape(::SerialTrafoEffects, shape::ArrayShape) = shape
ChangesOfVariables.with_logabsdet_jacobian(f::SerialTrafoEffects, x::AbstractVector{<:Real}) =
    (f(x), length(x) * log(2.0))

function serial_trafo_effect_oracle(n)
    draws, child_draws = UInt64[], Vector{UInt64}[]
    for _ in 1:n
        push!(draws, rand(UInt64))
        task = @task rand(UInt64, 4)
        schedule(task)
        push!(child_draws, fetch(task))
    end
    return (; draws, child_draws, full=copy(Random.default_rng()))
end

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

    @testset "Serial and threaded execution preserve transform results and ownership" begin
        for (name, f) in (("unshaped with Jacobian", f_dt),
                          ("shaped with Jacobian", f_hasladj),
                          ("shaped without Jacobian", f_plain),
                          ("unknown output shape", f_complex))
            @testset "$name" begin
                # Mutable nested metadata makes deepcopy observable rather than
                # merely comparing immutable IID metadata filled with nothing.
                source = DensitySampleVector(v=deepcopy(xs), logd=copy(xsv.logd),
                    weight=Float64.(xsv.weight), info=[[i] for i in eachindex(xs)],
                    aux=[Dict(:row => i) for i in eachindex(xs)])
                default_values = transform_samples(f, source.v)
                threaded_values = transform_samples(f, source.v, BAT.MultiThreadedExec())
                default_samples = transform_samples(f, source)
                threaded_samples = transform_samples(f, source, BAT.MultiThreadedExec())
                before_serial = copy(Random.default_rng())
                serial_values = transform_samples(f, source.v, BAT.SequentialExec())
                serial_samples = transform_samples(f, source, BAT.SequentialExec())
                @test Random.default_rng() == before_serial
                @test typeof(serial_values) === typeof(default_values) === typeof(threaded_values)
                @test isequal(serial_values, default_values)
                @test isequal(serial_values, threaded_values)
                @test typeof(serial_samples) === typeof(default_samples) === typeof(threaded_samples)
                for field in (:v, :logd, :weight, :info, :aux)
                    @test isequal(getproperty(serial_samples, field), getproperty(default_samples, field))
                    @test isequal(getproperty(serial_samples, field), getproperty(threaded_samples, field))
                end
                @test isequal(serial_samples.v, serial_values)
                for field in (:weight, :info, :aux)
                    @test isequal(getproperty(serial_samples, field), getproperty(source, field))
                    @test getproperty(serial_samples, field) !== getproperty(source, field)
                end
                original_weight = source.weight[1]
                serial_samples.weight[1] += 10
                serial_samples.info[1][1] = -1
                serial_samples.aux[1][:row] = -1
                @test source.weight[1] == original_weight
                @test source.info[1] == [1]
                @test source.aux[1] == Dict(:row => 1)
                saved_values, saved_logd = deepcopy(serial_samples.v), copy(serial_samples.logd)
                source.v[1].a[1] += 100
                source.logd[1] += 100
                @test isequal(serial_samples.v, saved_values)
                @test isequal(serial_samples.logd, saved_logd)
            end
        end
    end

    @testset "Identity transforms retain their bypasses" begin
        for executor in (BAT.SequentialExec(), BAT.MultiThreadedExec())
            before = copy(Random.default_rng())
            @test transform_samples(identity, xs, executor) === xs
            @test transform_samples(identity, xsv, executor) === xsv
            empty_samples = xsv[1:0]
            @test transform_samples(identity, empty_samples, executor) === empty_samples
            @test Random.default_rng() == before
        end
        @test_throws MethodError transform_samples(f_dt, xs, nothing)
        @test_throws MethodError transform_samples(identity, xsv, nothing)
    end

    @testset "Serial transform effects belong to the caller, including on failure" begin
        for density_samples in (false, true), fail in (false, true)
            @testset "density=$density_samples failure=$fail" begin
                f = SerialTrafoEffects(fail ? 2 : 0)
                rows = [[1.0], [2.0], [3.0]]
                inputs = density_samples ? DensitySampleVector(
                    v=rows, logd=[-1.0, -2.0, -3.0], weight=[1, 2, 3]) : rows
                n = fail ? 2 : 3
                Random.seed!(0x5412)
                expected = serial_trafo_effect_oracle(n)
                Random.seed!(0x5412)
                before = copy(Random.default_rng())
                result, error = try
                    (transform_samples(f, inputs, BAT.SequentialExec()), nothing)
                catch err
                    (nothing, err)
                end
                after = copy(Random.default_rng())
                @test after == expected.full
                @test after != before
                @test f.draws == expected.draws
                @test f.child_draws == expected.child_draws
                @test f.rows == Float64.(1:n)
                @test all(task -> task === current_task(), f.callers)
                @test all(istaskdone, f.children)
                @test !any(istaskfailed, f.children)
                if fail
                    @test error === f.failure
                    @test isnothing(result)
                    # A throwing serial row cannot leave an executor writer
                    # behind to evaluate later rows after the call has failed.
                    rows_at_failure = copy(f.rows)
                    yield()
                    yield()
                    @test f.rows == rows_at_failure == [1.0, 2.0]
                else
                    @test isnothing(error)
                    values = density_samples ? result.v : result
                    @test values == [[2.0], [4.0], [6.0]]
                    if density_samples
                        @test result.logd == inputs.logd .- log(2.0)
                        @test result.weight == inputs.weight
                    end
                end
            end
        end
    end

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

        # Nested output admission must retain the branch's execution policy,
        # including when the input has no ValueShapes representation.
        for executor in (BAT.SequentialExec(), BAT.MultiThreadedExec())
            before = copy(Random.default_rng())
            @test transform_samples(s -> (x = (n = length(s),),), ["ab", "c"], executor) == sys
            @test transform_samples(x -> (a = NamedTuple(), b = x), [1.0, 2.0], executor) == exs
            @test transform_samples(Base.Fix2(unshaped, nested_vs), nxs, executor) == nys
            if executor isa BAT.SequentialExec
                @test Random.default_rng() == before
            end
        end
    end
end
