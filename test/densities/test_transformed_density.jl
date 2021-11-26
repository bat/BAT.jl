# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra
using ValueShapes, ArraysOfArrays, Distributions, ForwardDiff
using DensityInterface, InverseFunctions, ChangesOfVariables

import Cuba


@testset "transformed_density" begin
    @testset "distribution transforms" begin
        function test_uv_transformed(target_type::Type{<:Distribution}, source_dist::Distribution)
            trafo = BAT.DistributionTransform(target_type, source_dist)
            @testset "$(typeof(trafo.source_dist))) to $(typeof(trafo.target_dist))" begin
                if target_type == Uniform
                    @test trafo.target_dist isa BAT.StandardUvUniform
                elseif target_type == Normal
                    @test trafo.target_dist isa BAT.StandardUvNormal
                end

                target_dist = trafo.target_dist
                source_dist = trafo.source_dist

                source_x = mean(target_dist) + std(target_dist) / 2
                @test @inferred(trafo(source_x)) isa Real
                target_x = trafo(source_x)

                source_X = rand(source_dist, 10^5)
                target_X = @inferred broadcast(trafo, source_X)
                @test isapprox(@inferred(broadcast(inverse(trafo), (target_X))), source_X, atol = 10^-8)
                @test isapprox(mean(target_X), mean(target_dist), atol = 0.05)
                @test isapprox(var(target_X), var(target_dist), atol = 0.1)

                @test @inferred(trafo(convert(AbstractDensity, source_dist))) isa BAT.TransformedDensity
                density = trafo(convert(AbstractDensity, source_dist))

                @test isapprox(@inferred(inverse(density.trafo)(target_x)), source_x, atol = 10^-5)

                @test minimum(target_dist) <= @inferred(bat_initval(density, InitFromTarget())).result <= maximum(target_dist)
                @test all(minimum(target_dist) .<= @inferred(bat_initval(density, 100, InitFromTarget())).result .<= maximum(target_dist))

                fix_nni(x::T) where {T<:Real} = x <= BAT.near_neg_inf(T) ? T(-Inf) : x

                tX = if target_type == Uniform
                    [0+eps(Float64), 0.25, 0.75, 1-eps(Float64)]
                elseif target_type == Normal
                    [-Inf, -2.1, -1.2, 0.0, 1.2, 2.1, Inf]
                else
                    @assert false
                end
    
                @test isapprox(fix_nni.(logdensityof(density).(tX)), logpdf.(target_dist, tX), atol = 1e-10)
                @test @inferred(logdensityof(density)(target_x)) isa Real
                @test @inferred(logdensityof(unshaped(density))(unshaped(target_x))) isa Real
                !any(isnan, @inferred(broadcast(ForwardDiff.derivative, logdensityof(density), tX)))

                tX_finite = tX[findall(isfinite, fix_nni.(logdensityof(density).(tX)))]
                @test isapprox(@inferred(broadcast(ForwardDiff.derivative, logdensityof(density), tX_finite)), broadcast(ForwardDiff.derivative, x -> logpdf(target_dist, x), tX_finite), atol = 10^-7)

                @test minimum(target_dist) <= bat_findmode(density, MaxDensityLBFGS(trafo = NoDensityTransform())).result <= maximum(target_dist)

                if trafo.target_dist isa Union{BAT.StandardUvUniform,BAT.StandardMvUniform}
                    @test isapprox(bat_integrate(density, VEGASIntegration(trafo = NoDensityTransform())).result, 1, rtol = 10^-7)
                end
            end
        end

        test_uv_transformed(Uniform, Weibull())
        test_uv_transformed(Uniform, Normal(2, 4))
        test_uv_transformed(Uniform, Uniform(-2, 3))

        test_uv_transformed(Normal, Weibull())
        test_uv_transformed(Normal, Normal(2, 4))
        test_uv_transformed(Normal, Uniform(-2, 3))


        src_d = NamedTupleDist(a = Exponential(), b = [4.2, 3.3], c = Weibull(), d = [Normal(1, 3), Normal(3, 2)], e = Uniform(-2, 3), f = MvNormal([0.3, -2.9], [1.7 0.5; 0.5 2.3]))
        trafo = @inferred(BAT.DistributionTransform(Normal, src_d))
        density = @inferred(trafo(convert(AbstractDensity, trafo.source_dist)))
        @test isfinite(@inferred logdensityof(density)(@inferred(bat_initval(density)).result))
        @test isapprox(cov(@inferred(bat_initval(density, 10^4)).result), I(totalndof(density)), rtol = 0.1)

        samples_is = bat_sample(density, MCMCSampling(mcalg = HamiltonianMC(), trafo = NoDensityTransform(), nsteps = 10^4)).result
        @test isapprox(cov(samples_is), I(totalndof(density)), rtol = 0.1)
        samples_os = inverse(trafo).(samples_is)
        @test all(isfinite, logpdf.(Ref(src_d), samples_os.v))
        @test isapprox(cov(unshaped.(samples_os)), cov(unshaped(src_d)), rtol = 0.1)
        @test isapprox(mean(unshaped.(samples_os)), mean(rand(unshaped(src_d), 10^5), dims = 2), rtol = 0.1)

        primary_dist = NamedTupleDist(a = Normal(), b = Weibull(), c = 5)
        f_secondary = x -> NamedTupleDist(y = Normal(x.a, x.b), z = MvNormal([1.3 0.5; 0.5 2.2]))
        prior = HierarchicalDistribution(f_secondary, primary_dist)
        likelihood = varshape(prior)(MvNormal(Diagonal(fill(1.0, totalndof(varshape(prior))))))
        density = PosteriorDensity(likelihood, prior)
        hmc_samples = bat_sample(density, MCMCSampling(mcalg = HamiltonianMC(), trafo = PriorToGaussian(), nsteps = 10^4)).result
        is_samples = bat_sample(density, PriorImportanceSampler(nsamples = 10^4)).result
        @test isapprox(mean(unshaped.(hmc_samples)), mean(unshaped.(is_samples)), rtol = 0.1)
        @test isapprox(cov(unshaped.(hmc_samples)), cov(unshaped.(is_samples)), rtol = 0.2)
    end
end
