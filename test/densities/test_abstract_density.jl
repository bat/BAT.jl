# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestAbstractDensity

using BAT
using Test

using LinearAlgebra, Random, StableRNGs
using DensityInterface, ValueShapes
using ArraysOfArrays, Distributions, PDMats, StatsBase
using AutoDiffOperators
import ForwardDiff, Zygote

# A mutable sentinel makes cause identity distinguishable from an equal value.
mutable struct DensityFailure <: Exception end
Base.showerror(io::IO, ::DensityFailure) = print(io, "original density failure")

# Model a third-party density without requiring AbstractMeasure inheritance.
struct TestDensity{F}
    f::F
end
DensityInterface.DensityKind(::TestDensity) = IsDensity()
DensityInterface.logdensityof(d::TestDensity, x) = d.f(x)
# The existing Zygote rule requires this interface. Bare LogFuncDensity shape
# discovery is a separate issue (H34); do not add a method on that foreign type.
ValueShapes.varshape(::TestDensity) = missing

struct TestMeasure{F} <: BAT.BATMeasure
    f::F
end
DensityInterface.logdensityof(m::TestMeasure, x) = m.f(x)

caught(f) = try
    f()
catch err
    err
end

function test_evalexception(f, target, v, cause)
    err = caught(f)
    @test err isa BAT.EvalException
    if err isa BAT.EvalException
        @test err.func === logdensityof
        @test err.measure === target
        @test err.v === v
        @test err.ret === cause
        @test fieldtypes(typeof(err)) ==
            (typeof(logdensityof), typeof(target), typeof(v), typeof(cause))
        message = sprint(showerror, err)
        @test occursin("Density evaluation", message)
        if cause isa DensityFailure
            @test occursin("original density failure", message)
        elseif cause isa Real
            @test occursin("must not evaluate to", message)
        end
    end
    return err
end

@testset "abstract_density" begin
    failure = DensityFailure()

    @testset "Generic and measure-owned evaluation failures" begin
        @test failure !== DensityFailure()
        for wrap in (logfuncdensity, BAT.LFDensity, TestDensity, TestMeasure)
            for v in (0.25, [0.25], (a=0.25,))
                target = wrap(_ -> throw(failure))
                @test caught(() -> logdensityof(target, v)) === failure
                test_evalexception(() -> BAT.checked_logdensityof(target, v), target, v, failure)
                test_evalexception(() -> BAT.checked_logdensityof(target)(v), target, v, failure)
            end
            for value in (NaN, Inf, Float32(NaN), Float32(Inf), big(NaN), big(Inf))
                target = wrap(_ -> value)
                test_evalexception(() -> BAT.checked_logdensityof(target, 0.25), target, 0.25, value)
            end
        end
    end

    @testset "Posterior and transformed likelihood failures" begin
        v = [0.25, -0.5]
        for cause in (failure, NaN, Inf)
            likelihood = logfuncdensity(_ -> cause isa Exception ? throw(cause) : cause)
            posterior = PosteriorMeasure(likelihood, MvNormal(zeros(2), ones(2)))
            test_evalexception(() -> BAT.checked_logdensityof(posterior, v), likelihood, v, cause)

            transformed = bat_transform(NormalBased(), posterior).result
            test_evalexception(() -> BAT.checked_logdensityof(transformed, v),
                BAT.getlikelihood(transformed), v, cause)

            joint = joint_likelihood(logfuncdensity(_ -> -1.0), likelihood)
            test_evalexception(() -> BAT.checked_logdensityof(joint, v), joint, v, cause)
        end
        shaped = PosteriorMeasure(logfuncdensity(_ -> throw(failure)), distprod(a=Normal(), b=Normal()))
        flat = unshaped(shaped)
        test_evalexception(() -> BAT.checked_logdensityof(flat, v), BAT.getlikelihood(flat), v, failure)

        # Independent Gaussian value and support controls.
        posterior = PosteriorMeasure(logfuncdensity(x -> -sum(abs2, x)/2), MvNormal(zeros(2), ones(2)))
        @test BAT.checked_logdensityof(posterior, v) ≈ -sum(abs2, v) - log(2pi)
        blocked = PosteriorMeasure(logfuncdensity(_ -> throw(failure)), Uniform(-1, 1))
        @test BAT.checked_logdensityof(blocked, 2.0) === -Inf
    end

    @testset "AD evaluation errors and successful gradients" begin
        v = [0.25, -0.5]
        for cause in (failure, NaN, Inf)
            f = _ -> cause isa Exception ? throw(cause) : cause
            for target in (TestDensity(f), PosteriorMeasure(logfuncdensity(f), MvNormal(zeros(2), ones(2))))
                # The existing posterior adjoint evaluates and reports the whole
                # posterior, whereas ordinary checked evaluation reports its likelihood.
                test_evalexception(() -> Zygote.pullback(BAT.checked_logdensityof(target), v), target, v, cause)
            end
        end
        for (target, factor) in ((TestDensity(x -> -sum(abs2, x)/2), 1),
            (PosteriorMeasure(logfuncdensity(x -> -sum(abs2, x)/2), MvNormal(zeros(2), ones(2))), 2))
            @test ForwardDiff.gradient(BAT.checked_logdensityof(target), v) ≈ -factor*v
            @test only(Zygote.gradient(BAT.checked_logdensityof(target), v)) ≈ -factor*v
        end
        known_shape = batmeasure(MvNormal(zeros(2), ones(2)))
        @test_throws ArgumentError Zygote.pullback(BAT.checked_logdensityof(known_shape), [0.2])
    end

    @testset "Prepared AD preserves failures and remains usable" begin
        adsel = BAT.get_adselector(BATContext(ad=Zygote))
        good, bad = [0.1, 0.2], [-0.1, 0.2]
        for cause in (failure, NaN, Inf)
            f = x -> x[1] < 0 ? (cause isa Exception ? throw(cause) : cause) : -sum(abs2, x)/2
            for (target, factor) in ((TestDensity(f), 1),
                (PosteriorMeasure(logfuncdensity(f), MvNormal(zeros(2), ones(2))), 2))
                fg = valgrad_func(BAT.checked_logdensityof(target), adsel, good)
                expected_value = -factor*sum(abs2, good)/2 - (factor-1)*log(2pi)
                value, gradient = fg(good)
                @test value ≈ expected_value
                @test gradient ≈ -factor*good
                test_evalexception(() -> fg(bad), target, bad, cause)
                value, gradient = fg(good)
                @test value ≈ expected_value
                @test gradient ≈ -factor*good
            end
        end
    end

    @testset "Zygote zero tangent" begin
        adsel = BAT.get_adselector(BATContext(ad = Zygote))
        uniform = unshaped(batmeasure(Uniform(-1, 1)))
        uniform_valgrad = valgrad_func(BAT.checked_logdensityof(uniform), adsel, [0.2])
        @test uniform_valgrad([0.2]) == (-log(2), [0.0])
    end

    @testset "Return types, inputs, evaluation count and RNG" begin
        for value in (0.0, -0.0, -Inf, Float32(0.5), Float32(-Inf), big(0.5), big(-Inf), 3, 1//3)
            target = logfuncdensity(_ -> value)
            @test (@inferred BAT.checked_logdensityof(target, 0.25)) === value
        end
        calls = Ref(0)
        v = [0.25, -0.5]
        before = copy(v)
        target = logfuncdensity(x -> (calls[] += 1; -sum(abs2, x)/2))
        default_next = rand(copy(Random.default_rng()), UInt64, 8)
        @test BAT.checked_logdensityof(target, v) == -5/32
        @test calls[] == 1
        @test v == before
        @test logfuncdensity(BAT.checked_logdensityof(target)) === target
        failing = logfuncdensity(_ -> throw(failure))
        test_evalexception(() -> BAT.checked_logdensityof(failing, v), failing, v, failure)
        @test v == before
        @test rand(copy(Random.default_rng()), UInt64, 8) == default_next
    end

    @testset "Logging and nested cause identity" begin
        saved_log = BAT.error_log()
        try
            BAT.enable_error_log(false)
            v = [0.1]
            target = logfuncdensity(_ -> throw(failure))
            test_evalexception(() -> BAT.checked_logdensityof(target, v), target, v, failure)
            @test BAT.error_log() === nothing
            BAT.enable_error_log(true)
            err = test_evalexception(() -> BAT.checked_logdensityof(target, v), target, v, failure)
            @test length(BAT.error_log()) == 1
            @test only(BAT.error_log()).error === err
            target = logfuncdensity(_ -> NaN)
            err = test_evalexception(() -> BAT.checked_logdensityof(target, v), target, v, NaN)
            @test length(BAT.error_log()) == 2
            @test last(BAT.error_log()).error === err
        finally
            # Restore the original object, including any caller-owned entries.
            BAT.g_error_log[] = saved_log
        end
        inner = BAT.EvalException(logdensityof, logfuncdensity(identity), 0.1, failure)
        target = logfuncdensity(_ -> throw(inner))
        err = test_evalexception(() -> BAT.checked_logdensityof(target, 0.1), target, 0.1, inner)
        @test err isa BAT.EvalException && err.ret.ret === failure
        # Preserve the existing wrapping policy, without claiming native signal safety.
        interruption = InterruptException()
        target = logfuncdensity(_ -> throw(interruption))
        test_evalexception(() -> BAT.checked_logdensityof(target, 0.1), target, 0.1, interruption)
    end

    @testset "Diagnostic display for generic variates" begin
        for v in ((0.2,), :opaque, (parameters=[0.2], tag=:group), Any[0.2, :group])
            target = logfuncdensity(_ -> throw(failure))
            test_evalexception(() -> BAT.checked_logdensityof(target, v), target, v, failure)
        end
        dual = ForwardDiff.Dual(0.25, 1.0)
        target = logfuncdensity(x -> x*NaN)
        err = caught(() -> BAT.checked_logdensityof(target, dual))
        @test err isa BAT.EvalException
        if err isa BAT.EvalException
            @test err.v === dual
            @test err.ret isa ForwardDiff.Dual
            @test isnan(ForwardDiff.value(err.ret))
            message = sprint(showerror, err)
            @test occursin("NaN", message)
            @test !occursin("Dual", message)
        end
        @test BAT.value_for_msg([dual]) == [0.25]
        @test BAT.value_for_msg((a=dual, tag=:group)) == (a=0.25, tag=:group)
    end
end

end # module
