# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random
using ValueShapes, ArraysOfArrays, Distributions, MeasureBase
using DensityInterface, InverseFunctions, ChangesOfVariables
using MeasureBase: StdUniform, StdNormal, transport_to, pushfwd, mbind
import ForwardDiff
using Random123

import Cuba
using Optim


@testset "pushfwd_measure" begin
    context = BATContext(rng = Philox4x((564, 18)), ad = ForwardDiff)

    @testset "transports of univariate measures" begin
        function test_uv_transported(ν::MeasureBase.StdMeasure, source_dist::Distribution)
            context = BATContext(rng = Philox4x((564, 19)), ad = ForwardDiff)
            μ = batmeasure(source_dist)
            f_transform = @inferred transport_to(ν, μ)
            @testset "$(typeof(source_dist)) to $(typeof(ν))" begin
                source_x = mean(source_dist) + std(source_dist) / 2
                @test @inferred(f_transform(source_x)) isa Real
                target_x = f_transform(source_x)

                source_X = rand(Xoshiro(564020), source_dist, 10^5)
                target_X = broadcast(f_transform, source_X)
                @test isapprox(broadcast(inverse(f_transform), target_X), source_X, atol = 10^-8)
                @test isapprox(mean(target_X), mean(rand(Xoshiro(564021), ν^10^5)), atol = 0.05)

                m = @inferred(pushfwd(f_transform, μ))
                @test m isa MeasureBase.PushforwardMeasure

                @test isapprox(@inferred(inverse(MeasureBase.gettransform(m))(target_x)), source_x, atol = 10^-5)

                @test isfinite(@inferred(bat_initval(m, InitFromTarget(), context)).result)
                @test all(isfinite, @inferred(bat_initval(m, 100, InitFromTarget(), context)).result)

                # The pushforward of a measure to a standard measure is that
                # standard measure:
                tX = ν isa StdUniform ? [0.25, 0.5, 0.75] : [-2.1, -1.2, 0.0, 1.2, 2.1]
                @test isapprox(logdensityof(m).(tX), logdensityof(ν).(tX), atol = 1e-10)
                @test @inferred(logdensityof(m)(target_x)) isa Real
                @test @inferred(logdensityof(unshaped(m))([target_x])) ≈ logdensityof(m)(target_x)

                @test isapprox(
                    broadcast(ForwardDiff.derivative, logdensityof(m), tX),
                    broadcast(ForwardDiff.derivative, logdensityof(ν), tX), atol = 10^-7
                )

                @test isfinite(bat_findmode(m, TransformedMaxDensity(optalg = OptimAlg(optalg = LBFGS()), pretransform = DoNotTransform()), context).result)

                if ν isa StdUniform
                    @test isapprox(bat_integrate(m, VEGASIntegration(pretransform = DoNotTransform()), context).result, 1, rtol = 10^-7)
                end
            end
        end

        for ν in (StdUniform(), StdNormal()), d in (Weibull(), Normal(2, 4), Uniform(-2, 3))
            test_uv_transported(ν, d)
        end
    end

    @testset "transports of structured measures" begin
        src_m = distprod(a = Exponential(), b = [4.2, 3.3], c = Weibull(), d = [Normal(1, 3), Normal(3, 2)], e = Uniform(-2, 3), f = MvNormal([0.3, -2.9], [1.7 0.5; 0.5 2.3]))
        n = BAT.some_dof(src_m)
        f_transform = transport_to(StdNormal()^n, src_m)
        m = pushfwd(f_transform, src_m)
        @test isfinite(logdensityof(m)(bat_initval(m, context).result))
        @test isapprox(cov(bat_initval(m, 10^4, context).result), I(n), rtol = 0.1)

        samples_is = bat_sample(m, TransformedMCMC(proposal = HamiltonianMC(), pretransform = DoNotTransform(), nsteps = 10^4), BATContext(rng = Philox4x((564, 21)), ad = ForwardDiff)).result
        @test isapprox(cov(samples_is), I(n), rtol = 0.1)
        samples_os = inverse(f_transform).(samples_is)
        @test all(isfinite, logdensityof(src_m).(samples_os.v))

        vs = varshape(src_m)
        ref_x = unshaped.(rand(Xoshiro(564022), src_m^10^5), Ref(vs))
        @test isapprox(mean(unshaped.(samples_os)), mean(ref_x), rtol = 0.1, atol = 0.05)
        @test isapprox(cov(unshaped.(samples_os)), cov(ref_x), rtol = 0.1, atol = 0.05)
    end

    @testset "hierarchical measures" begin
        primary = distprod(a = Normal(), b = Weibull(), c = 5)
        prior = mbind(primary, merge) do x
            distprod(y = Normal(x.a, x.b), z = MvNormal([1.3 0.5; 0.5 2.2]))
        end
        n = totalndof(varshape(prior))
        likelihood = logfuncdensity(logdensityof(varshape(prior)(batmeasure(MvNormal(Diagonal(fill(1.0, n)))))))
        m = PosteriorMeasure(likelihood, prior)
        hmc_samples = bat_sample(m, TransformedMCMC(proposal = HamiltonianMC(), pretransform = NormalBased(), nsteps = 10^4), BATContext(rng = Philox4x((564, 23)), ad = ForwardDiff)).result
        is_samples = bat_sample(m, PriorImportanceSampler(nsamples = 10^4), BATContext(rng = Philox4x((564, 24)), ad = ForwardDiff)).result
        # Compare the means on the scale of the distribution itself: most of
        # these variates have a mean close to zero, where a relative tolerance
        # demands far more precision than the Monte Carlo error allows. The
        # constant component of the prior has no spread at all, it must be
        # reproduced exactly:
        is_stddevs = sqrt.(diag(cov(unshaped.(is_samples))))
        mean_deviations = abs.(mean(unshaped.(hmc_samples)) .- mean(unshaped.(is_samples)))
        free = is_stddevs .> 1e-6
        @test all(mean_deviations[free] .<= 0.15 .* is_stddevs[free])
        @test all(mean_deviations[.!free] .< 1e-9)
        @test isapprox(cov(unshaped.(hmc_samples)), cov(unshaped.(is_samples)), rtol = 0.2)
    end
end
