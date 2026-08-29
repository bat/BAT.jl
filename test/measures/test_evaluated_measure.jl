# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random
using DensityInterface, MeasureBase, ValueShapes
using Distributions, StatsBase, IntervalSets
using LazyReports: LazyReport, lazyreport
using Accessors: @set
using ScopedSettings: unchanged

struct EvalmeasureDensityError <: BAT.BATMeasure end

DensityInterface.logdensityof(::EvalmeasureDensityError, ::Any) = error("evalmeasure density failure")
ValueShapes.varshape(::EvalmeasureDensityError) = ScalarShape{Float64}()
MeasureBase.getdof(::EvalmeasureDensityError) = 1

struct EvalmeasureDensityMethodError <: BAT.BATMeasure end

evalmeasure_density_methoderror(::Nothing) = nothing
DensityInterface.logdensityof(::EvalmeasureDensityMethodError, ::Any) = evalmeasure_density_methoderror(0.0)
ValueShapes.varshape(::EvalmeasureDensityMethodError) = ScalarShape{Float64}()
MeasureBase.getdof(::EvalmeasureDensityMethodError) = 1

struct EvalmeasureInterruptTarget <: MeasureBase.AbstractMeasure end

BAT.batmeasure(::EvalmeasureInterruptTarget) = throw(InterruptException())

@testset "evaluated_measure" begin
    dist = distprod(
        a = truncated(Normal(), -2, 2),
        b = Exponential(),
        c = [1 2; 3 4],
        d = [-3..3, -4..4]
    )

    m = batmeasure(dist)

    @test @inferred(BAT.unevaluated(EvaluatedMeasure(m))) === m
    @test @inferred(BAT.unevaluated(EvaluatedMeasure(dist))).dist === dist

    n = 100
    xs = rand(Xoshiro(564003), m^n)
    xs_logd = logdensityof(m).(xs)
    smpls = DensitySampleVector(v = xs, logd = xs_logd)
    empirical_m = DensitySampleMeasure(smpls, dof = getdof(m))

    em = EvaluatedMeasure(m, empirical = empirical_m, mass = 1)
    @test @inferred(BAT.unevaluated(em)) === m
    # The dual-space entries are stored as BispacedMeasure pairs:
    @test em.unevaluated === BAT.BispacedMeasure(m)
    @test em.empirical === BAT.BispacedMeasure(empirical_m)
    @test em.transform_intent === DoNotTransform()
    @test em.f_transform === identity
    @test @inferred(BAT.empiricalof(em)) === empirical_m
    @test @inferred(BAT.samplesof(em)) === BAT.samplesof(empirical_m)
    @test @inferred(getdof(em)) == getdof(m)
    @test massof(em) ≈ 1
    @test @inferred(varshape(em)) == varshape(m)
    x = first(xs)
    @test @inferred(logdensityof(em, x)) == logdensityof(m, x)
    @test DensitySampleVector(em) == smpls

    em_dsm = EvaluatedMeasure(batmeasure(smpls))
    @test BAT.empiricalof(em_dsm) === BAT.unevaluated(em_dsm)

    em_plain = EvaluatedMeasure(m)
    @test BAT.empiricalof(em_plain) === nothing

    # The approximate statistics operate on unshaped measures:
    mf = batmeasure(MvNormal([0.4, 0.6], [2.0 1.2; 1.2 3.0]))
    xs_f = rand(Xoshiro(564004), mf^n)
    smpls_f = DensitySampleVector(v = xs_f, logd = logdensityof(mf).(xs_f))
    empirical_f = DensitySampleMeasure(smpls_f, dof = getdof(mf))
    em_f = EvaluatedMeasure(mf, empirical = empirical_f)
    em_f_plain = EvaluatedMeasure(mf)
    nf = totalndof(varshape(mf))

    @testset "approximate mean and cov" begin
        @test BAT._approx_mean(em_f, nf) == mean(smpls_f)
        @test BAT._approx_cov(em_f, nf) == cov(smpls_f)

        # Without empirical content both must fall back to the underlying measure:
        @test BAT._approx_mean(em_f_plain, nf) == BAT._approx_mean(mf, nf)
        @test BAT._approx_cov(em_f_plain, nf) == BAT._approx_cov(mf, nf)
    end

    @testset "approximate max logd" begin
        @test BAT._approx_max_logd(em) == BAT._approx_max_logd(empirical_m)
        @test BAT._approx_max_logd(em_plain) === BAT._approx_max_logd(m)
    end

    @testset "knowledge update semantics" begin
        approx_a = batmeasure(distprod(
            a = truncated(Normal(0.1, 0.9), -2, 2),
            b = Exponential(2.0),
            c = [1 2; 3 4],
            d = [-3..3, -4..4]
        ))
        em_a = EvaluatedMeasure(m, empirical = empirical_m, mass = 1)

        # Identity if all entries stay unchanged:
        @test EvaluatedMeasure(em_a) === em_a
        @test EvaluatedMeasure(em_a, approx = unchanged) === em_a

        # Given values replace, everything else is kept:
        em_b = EvaluatedMeasure(em_a, approx = approx_a)
        @test em_b.approx === BAT.BispacedMeasure(approx_a)
        @test BAT.approxof(em_b) === approx_a
        @test em_b.empirical === em_a.empirical
        @test massof(em_b) == massof(em_a)

        # nothing clears, everything else is kept:
        em_c = EvaluatedMeasure(em_b, approx = nothing)
        @test isnothing(em_c.approx)
        @test em_c.empirical === em_b.empirical
        em_d = EvaluatedMeasure(em_b, empirical = nothing)
        @test isnothing(em_d.empirical)
        @test em_d.approx === em_b.approx

        # An explicit unknown mass resets the mass:
        @test massof(EvaluatedMeasure(em_a, mass = MeasureBase.UnknownMass())) isa MeasureBase.UnknownMass

        # Accessors provide direct field surgery:
        em_e = @set em_a.mass = 0.5
        @test massof(em_e) == 0.5
        em_f = @set em_a.empirical = nothing
        @test isnothing(em_f.empirical)
        @test BAT.unevaluated(em_f) === BAT.unevaluated(em_a)
    end

    @testset "unshaped transports knowledge" begin
        approx_m = batmeasure(distprod(
            a = truncated(Normal(0.1, 0.9), -2, 2),
            b = Exponential(2.0),
            c = [1 2; 3 4],
            d = [-3..3, -4..4]
        ))
        em_k = EvaluatedMeasure(m, empirical = empirical_m, approx = approx_m, modes = [x], mass = 1)
        vs = varshape(em_k)
        uem = unshaped(em_k, vs)
        @test BAT.unevaluated(uem) == unshaped(m, vs)
        @test BAT.empiricalof(uem) == unshaped(empirical_m, vs)
        @test BAT.approxof(uem) == unshaped(approx_m, vs)
        @test uem.modes == [unshaped(x, vs)]
        @test massof(uem) == massof(em_k)
        # unevaluated is the explicit knowledge strip:
        @test !(BAT.unevaluated(uem) isa EvaluatedMeasure)
    end

    @testset "report generation" begin
        @test lazyreport(em) isa LazyReport
        @test lazyreport(em_plain) isa LazyReport
    end

    @testset "validation error boundaries" begin
        error_em = EvaluatedMeasure(
            EvalmeasureDensityError(),
            empirical = DensitySampleVector(v = [0.0], logd = [0.0])
        )
        @test_throws ErrorException BAT.validate_evalmeasure(error_em)

        methoderror_em = EvaluatedMeasure(
            EvalmeasureDensityMethodError(),
            empirical = DensitySampleVector(v = [0.0], logd = [0.0])
        )
        err = try
            BAT.validate_evalmeasure(methoderror_em)
            nothing
        catch err
            err
        end
        @test err isa MethodError
        @test err.f === evalmeasure_density_methoderror

        validation_context = BATContext()
        _, f_validation = BAT.transform_and_unshape(ToRealVector(), m, validation_context)
        unsupported_approx = DensitySampleMeasure(
            DensitySampleVector(v = [f_validation(first(xs))], logd = [0.0])
        )
        unsupported_em = EvaluatedMeasure(
            m,
            transform_intent = ToRealVector(),
            f_transform = f_validation,
            empirical = empirical_m,
            approx = BAT.BispacedMeasure(m, unsupported_approx, hash(f_validation)),
        )
        @test BAT.validate_evalmeasure(unsupported_em, context = validation_context) === unsupported_em

        @test_throws InterruptException evalmeasure(EvalmeasureInterruptTarget())
    end
end
