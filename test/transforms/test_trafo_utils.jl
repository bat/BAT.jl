# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using MeasureBase
using ValueShapes, Distributions, ArraysOfArrays, FillArrays, StructArrays
using InverseFunctions, ChangesOfVariables
using MeasureBase: StdNormal, StdUniform, TransportFunction, transport_to

using BAT: transform_samples

@testset "test_trafo_utils" begin
    dist = distprod(
        a = MvNormal([0.4, 0.6], [2.0 1.2; 1.2 3.0]),
        b = Exponential(2.3),
        c = 4.2,
    )
    mu = batmeasure(dist)

    xsv = bat_sample(dist, IIDSampling(nsamples = 20)).result
    xs = xsv.v
    @test xs isa StructVector && xs.c isa Fill

    myidentity(x) = x
    InverseFunctions.inverse(f::typeof(myidentity)) = f
    ChangesOfVariables.with_logabsdet_jacobian(::typeof(myidentity), x) = x, Bool(false)

    f_dt = transport_to(StdNormal()^BAT.some_dof(mu), mu)
    f_hasladj = myidentity ∘ f_dt
    f_plain(x) = (d = sum(x.a) * x.c, e = x.b * x.a)
    f_complex(x) = (d = sum(x.a) * x.c, e = (f = x.b * x.a,))

    @testset "transports" begin
        for f in (f_dt, f_hasladj)
            ys = transform_samples(f, xs)
            @test ys isa VectorOfSimilarVectors{Float64}
            @test ys ≈ [f_dt(x) for x in xs]
            ysv = transform_samples(f, xsv)
            @test ysv.v == ys
            @test logdensityof(pushfwd(f, mu)).(ys) ≈ ysv.logd
        end

        # The inverse transport restores the samples, with constants:
        ysv = transform_samples(f_dt, xsv)
        xsv2 = transform_samples(inverse(f_dt), ysv)
        @test xsv2.v.c isa Fill
        @test isapprox(xsv2, xsv)

        # Transports between equal measures don't share storage with their input:
        Z = nestedview(randn(3, 4))
        Z2 = transform_samples(transport_to(StdNormal()^3, StdNormal()^3), Z)
        @test Z2 == Z && !Base.mightalias(flatview(Z2), flatview(Z))

        # A transport into a measure of higher precision promotes the
        # number type of the variates:
        xs32 = VectorOfSimilarVectors(rand(Float32, 2, 20))
        f32 = transport_to(batmeasure(MvNormal([0.4, 0.6], [2.0 1.2; 1.2 3.0])), StdUniform()^2)
        @test transform_samples(f32, xs32) isa VectorOfSimilarVectors{Float64}
    end

    @testset "functions without log-abs-det Jacobian" begin
        ys = transform_samples(f_plain, xs)
        @test ys isa StructVector && ys.e isa ArrayOfSimilarArrays
        @test ys == f_plain.(collect(xs))
        ysv = transform_samples(f_plain, xsv)
        @test ysv.v == ys
        @test all(isnan, ysv.logd)

        ys = transform_samples(f_complex, xs)
        @test ys isa StructVector && ys.e isa StructVector
        @test ys == f_complex.(collect(xs))
        @test all(isnan, transform_samples(f_complex, xsv).logd)

        # A function with log-abs-det Jacobian after one without:
        @test all(isnan, transform_samples(myidentity, transform_samples(f_plain, xsv)).logd)
    end

    @testset "flat and structured variates" begin
        nested_dist = distprod(x = (a = Uniform(0, 1), b = (c = Normal(), d = MvNormal([1.0 0.0; 0.0 1.0]))), e = Uniform(0, 1))
        nested_vs = varshape(nested_dist)
        nxs = [rand(nested_dist) for _ in 1:3]
        # Unshaping explicit values, as the DoNotTransform pretransform does:
        nys = transform_samples(Base.Fix2(unshaped, nested_vs), nxs)
        @test nys isa VectorOfSimilarVectors
        @test nys == unshaped.(nxs, Ref(nested_vs))
        nxs2 = transform_samples(nested_vs, nys)
        @test nxs2 isa StructVector && nxs2.x.b isa StructVector
        @test nxs2 == nxs

        snt_dist = NamedTupleDist(ShapedAsNT, x = (a = Uniform(0, 1),), e = Uniform(0, 1))
        sxs = [rand(snt_dist) for _ in 1:3]
        @test sxs isa AbstractVector{<:ShapedAsNT}
        @test transform_samples(Base.Fix2(unshaped, varshape(snt_dist)), sxs) == unshaped.(sxs)

        # Union-typed fields of scalars and arrays:
        uxs = NamedTuple{(:a,),Tuple{Union{Float64,Vector{Float64}}}}[(a = 1.0,), (a = 2.0,)]
        @test transform_samples(x -> (a = x.a + 1,), uxs) == [(a = 2.0,), (a = 3.0,)]

        # Structured outputs from other inputs:
        sys = transform_samples(s -> (x = (n = length(s),),), ["ab", "c"])
        @test sys isa StructVector && sys.x isa StructVector
        @test sys == [(x = (n = 2,),), (x = (n = 1,),)]

        # Empty named tuples carry no degrees of freedom:
        exs = transform_samples(x -> (a = NamedTuple(), b = x), [1.0, 2.0])
        @test exs.a isa Fill
        @test exs == [(a = NamedTuple(), b = 1.0), (a = NamedTuple(), b = 2.0)]
    end
end
