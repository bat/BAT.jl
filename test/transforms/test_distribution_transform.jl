# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra
using ValueShapes, Distributions, ArraysOfArrays
using ForwardDiff, Zygote, DistributionsAD
using InverseFunctions, ChangesOfVariables, DensityInterface
using IntervalSets
using MeasureBase

@testset "test_distribution_transform" begin
    function test_back_and_forth(trg_d, src_d)
        @testset "transform $(typeof(trg_d).name) <-> $(typeof(src_d).name)" begin
            src_v = rand(src_d)
            trg_v = BAT.apply_dist_trafo(trg_d, src_d, src_v)
            src_v_reco = BAT.apply_dist_trafo(src_d, trg_d, trg_v)

            @test isapprox(src_v, src_v_reco, rtol = 1e-5)

            if (totalndof(varshape(trg_d)) == totalndof(varshape(src_d)))
                let vs_trg = varshape(trg_d), vs_src = varshape(src_d)
                    f = unshaped_x -> inverse(vs_trg)(BAT.apply_dist_trafo(trg_d, src_d, vs_src(unshaped_x)))
                    ref_ladj = logpdf(src_d, src_v) - logpdf(trg_d, trg_v)
                    @test isapprox(ref_ladj, logabsdet(ForwardDiff.jacobian(f, inverse(vs_src)(src_v)))[1], rtol = 1e-6, atol = 1e-6)
                end
            end
        end
    end

    function get_trgxs(trg_d, src_d, X)
        return (x -> BAT.apply_dist_trafo(trg_d, src_d, x)).(nestedview(X))
    end

    function get_trgxs(trg_d, src_d::Distribution{Univariate}, X)
        return (x -> BAT.apply_dist_trafo(trg_d, src_d, x)).(X)
    end

    function test_dist_trafo_moments(trg_d, src_d)
        @testset "check moments of transform $(typeof(trg_d).name) <- $(typeof(src_d).name)" begin
            X = flatview(rand(src_d, 10^5))
            trgxs = get_trgxs(trg_d, src_d, X)
            unshaped_trgxs = broadcast(unshaped, trgxs, Ref(varshape(trg_d)))
            @test isapprox(mean(unshaped_trgxs), mean(unshaped(trg_d)), atol = 0.1)
            @test isapprox(cov(unshaped_trgxs), cov(unshaped(trg_d)), rtol = 0.1)
        end
    end

    stduvuni = BAT.StandardUvUniform()
    stduvnorm = BAT.StandardUvNormal()

    uniform1 = Uniform(-5.0, -0.01)
    uniform2 = Uniform(0.01, 5.0)

    normal1 = Normal(-10, 1)
    normal2 = Normal(10, 5)

    stdmvnorm1 = BAT.StandardMvNormal(1)
    stdmvnorm2 = BAT.StandardMvNormal(2)

    stdmvuni2 = BAT.StandardMvUniform(2)

    standnorm2_reshaped = ReshapedDist(stdmvnorm2, varshape(stdmvnorm2))

    mvnorm = MvNormal([0.3, -2.9], [1.7 0.5; 0.5 2.3])
    beta = Beta(3,1)
    gamma = Gamma(0.1,0.7)
    dirich = Dirichlet([0.5, 4.0, 2.2, 0.7])

    locscaledist = -2 * LogUniform(0.1, 10.0) + 3
    @test locscaledist isa Distributions.AffineDistribution
    locscaledist2 = 3 * LogUniform(0.5, 20.0) -2

    ntdist = NamedTupleDist(
        a = uniform1,
        b = mvnorm,
        c = [4.2, 3.7],
        x = beta,
        y = gamma,
        z = locscaledist
    )


    test_back_and_forth(stduvuni, stduvuni)
    test_back_and_forth(stduvnorm, stduvnorm)
    test_back_and_forth(stduvuni, stduvnorm)
    test_back_and_forth(stduvnorm, stduvuni)

    test_back_and_forth(stdmvuni2, stdmvuni2)
    test_back_and_forth(stdmvnorm2, stdmvnorm2)
    test_back_and_forth(stdmvuni2, stdmvnorm2)
    test_back_and_forth(stdmvnorm2, stdmvuni2)

    test_back_and_forth(beta, stduvnorm)
    test_back_and_forth(gamma, stduvnorm)

    test_back_and_forth(locscaledist, stduvuni)
    test_back_and_forth(locscaledist, stduvnorm)
    test_back_and_forth(stduvuni, locscaledist)
    test_back_and_forth(stduvnorm, locscaledist)
    test_back_and_forth(locscaledist2, locscaledist2)

    test_dist_trafo_moments(locscaledist, stduvnorm)
    test_dist_trafo_moments(locscaledist, locscaledist2)

    test_dist_trafo_moments(normal2, normal1)
    test_dist_trafo_moments(uniform2, uniform1)

    test_dist_trafo_moments(beta, gamma)

    test_dist_trafo_moments(beta, stduvnorm)
    test_dist_trafo_moments(gamma, stduvnorm)

    test_dist_trafo_moments(mvnorm, stdmvnorm2)
    test_dist_trafo_moments(dirich, BAT.StandardMvNormal(3))

    test_dist_trafo_moments(mvnorm, stdmvuni2)
    test_dist_trafo_moments(stdmvuni2, mvnorm)

    test_dist_trafo_moments(stdmvnorm2, stdmvuni2)

    test_dist_trafo_moments(mvnorm, standnorm2_reshaped)
    test_dist_trafo_moments(standnorm2_reshaped, mvnorm)
    test_dist_trafo_moments(stdmvnorm2, standnorm2_reshaped)
    test_dist_trafo_moments(standnorm2_reshaped, standnorm2_reshaped)

    test_back_and_forth(dirich, BAT.StandardMvNormal(3))

    test_back_and_forth(ntdist, BAT.StandardMvNormal(6))
    test_back_and_forth(ntdist, BAT.StandardMvUniform(6))

    let
        mvuni = product_distribution([Uniform(), Uniform()])

        x = rand()
        @test_throws ArgumentError BAT.apply_dist_trafo(stduvnorm, mvnorm, x)
        @test_throws ArgumentError BAT.apply_dist_trafo(stduvnorm, stdmvnorm1, x)
        @test_throws ArgumentError BAT.apply_dist_trafo(stduvnorm, stdmvnorm2, x)

        x = rand(2)
        @test_throws ArgumentError BAT.apply_dist_trafo(mvuni, mvnorm, x)
        @test_throws ArgumentError BAT.apply_dist_trafo(mvnorm, mvuni, x)
        @test_throws ArgumentError BAT.apply_dist_trafo(stduvnorm, mvnorm, x)
        @test_throws ArgumentError BAT.apply_dist_trafo(stduvnorm, stdmvnorm1, x)
        @test_throws ArgumentError BAT.apply_dist_trafo(stduvnorm, stdmvnorm2, x)
    end

    let
        primary_dist = NamedTupleDist(x = Normal(2), c = 5)
        f = x -> NamedTupleDist(y = Normal(x.x, 3), z = MvNormal([1.3 0.5; 0.5 2.2]))
        trg_d = @inferred(HierarchicalDistribution(f, primary_dist))
        src_d = BAT.StandardMvNormal(totalndof(varshape(trg_d)))
        test_back_and_forth(trg_d, src_d)
        test_dist_trafo_moments(trg_d, src_d)
    end

    @testset "Custom cdf and quantile for dual numbers" begin
        Dual = ForwardDiff.Dual

        @test BAT._trafo_cdf(Normal(Dual(0, 1, 0, 0), Dual(1, 0, 1, 0)), Dual(0.5, 0, 0, 1)) ≈ cdf(Normal(Dual(0, 1, 0, 0), Dual(1, 0, 1, 0)), Dual(0.5, 0, 0, 1))
        @test BAT._trafo_cdf(Normal(0, 1), Dual(0.5, 1)) ≈ cdf(Normal(0, 1), Dual(0.5, 1))

        @test BAT._trafo_quantile(Normal(0, 1), Dual(0.5, 1)) ≈ quantile(Normal(0, 1), Dual(0.5, 1))
        @test BAT._trafo_quantile(Normal(Dual(0, 1, 0, 0), Dual(1, 0, 1, 0)), Dual(0.5, 0, 0, 1)) ≈ quantile(Normal(Dual(0, 1, 0, 0), Dual(1, 0, 1, 0)), Dual(0.5, 0, 0, 1))
    end

    for VT in (NamedTuple, ShapedAsNT)
        src_dist = unshaped(NamedTupleDist(VT, a = Weibull(), b = MvNormal([1.3 0.6; 0.6 2.4])))
        f = BAT.DistributionTransform(Normal, src_dist)
        x = rand(src_dist)
        InverseFunctions.test_inverse(f, x)
        ChangesOfVariables.test_with_logabsdet_jacobian(f, x, ForwardDiff.jacobian)
    end

    @testset "transfom broadcasting" begin
        dist = NamedTupleDist(a = Weibull(), b = Exponential())
        smpls = bat_sample(dist, IIDSampling(nsamples = 100)).result
        f_transform = BAT.DistributionTransform(Normal, dist)
        @inferred(broadcast(f_transform, smpls)) isa DensitySampleVector
        smpls_tr = f_transform.(smpls)
        smpls_tr_cmp = [f_transform(s) for s in smpls]
        @test smpls_tr == smpls_tr_cmp
        @test @inferred(resultshape(f_transform, elshape(smpls.v))) == varshape(f_transform.target_dist)
    end

    # @testset "transform composition" begin
    #     dist1 = @inferred(NamedTupleDist(a = Normal(), b = Uniform(), c = Cauchy()))
    #     dist2 = @inferred(NamedTupleDist(a = Exponential(), b = Weibull(), c = Beta()))
    #     normal1 = Normal()
    #     normal2 = Normal(2)
    # 
    #     f_transform = @inferred(BAT.DistributionTransform(dist1, dist2))
    #     inv_trafo = @inferred(inverse(f_transform))
    # 
    #     composed_trafo = @inferred(∘(f_transform, inv_trafo))
    #     @test composed_trafo.source_dist == composed_trafo.target_dist == dist1
    #     @test composed_trafo ∘ f_transform == f_transform
    #     @test_throws ArgumentError  f_transform ∘ composed_trafo
    # 
    #     f_transform = @inferred(BAT.DistributionTransform(normal1, normal2))
    #     @test_throws ArgumentError f_transform ∘ f_transform
    # end

    @testset "full density transform" begin
        context = BATContext()

        likelihood = logfuncdensity(logdensityof(NamedTupleDist(a = Normal(), b = Exponential())))
        prior = NamedTupleDist(a = Normal(), b = Gamma())
        posterior_density = PosteriorMeasure(likelihood, prior)

        posterior_density_trafod = @inferred(bat_transform(PriorToUniform(), posterior_density, BAT.FullMeasureTransform(), context))

        @test posterior_density_trafod.result.origin.likelihood._log_f == likelihood._log_f
        @test posterior_density_trafod.result.origin.prior.dist == prior

        @test posterior_density_trafod.result.f.target_dist isa BAT.StandardMvUniform
    end

    @testset "transform autodiff pullbacks" begin
        # ToDo: Test for type stability and fix where necessary.

        xs = rand(5)
        @test Zygote.jacobian(BAT._pushfront, xs, 42)[1] ≈ ForwardDiff.jacobian(xs -> BAT._pushfront(xs, 1), xs)
        @test Zygote.jacobian(BAT._pushfront, xs, 42)[2] ≈ vec(ForwardDiff.jacobian(x -> BAT._pushfront(xs, x[1]), [42]))
        @test Zygote.jacobian(BAT._pushback, xs, 42)[1] ≈ ForwardDiff.jacobian(xs -> BAT._pushback(xs, 1), xs)
        @test Zygote.jacobian(BAT._pushback, xs, 42)[2] ≈ vec(ForwardDiff.jacobian(x -> BAT._pushback(xs, x[1]), [42]))
        @test Zygote.jacobian(BAT._rev_cumsum, xs)[1] ≈ ForwardDiff.jacobian(BAT._rev_cumsum, xs)
        @test Zygote.jacobian(BAT._exp_cumsum_log, xs)[1] ≈ ForwardDiff.jacobian(BAT._exp_cumsum_log, xs) ≈ ForwardDiff.jacobian(cumprod, xs)

        src_v = [0.6, 0.7, 0.8, 0.9]
        f = inverse(BAT.DistributionTransform(Uniform, DistributionsAD.TuringDirichlet([3.0, 4.0, 5.0, 6.0, 7.0])))
        @test isapprox(ForwardDiff.jacobian(f, src_v), Zygote.jacobian(f, src_v)[1], rtol = 10^-4)
        f = inverse(BAT.DistributionTransform(Uniform, Dirichlet([3.0, 4.0, 5.0, 6.0, 7.0])))
        @test isapprox(ForwardDiff.jacobian(f, src_v), Zygote.jacobian(f, src_v)[1], rtol = 10^-4)
        f = inverse(BAT.DistributionTransform(Normal, Dirichlet([3.0, 4.0, 5.0, 6.0, 7.0])))
        @test isapprox(ForwardDiff.jacobian(f, src_v), Zygote.jacobian(f, src_v)[1], rtol = 10^-4)
    end
end


@testset "bat_transform_defaults" begin
    context = BATContext()

    mvn = @inferred(product_distribution([Normal(-1), Normal(), Normal(1)]))
    uniform_prior = @inferred(product_distribution([Uniform(-3, 1), Uniform(-2, 2), Uniform(-1, 3)]))

    posterior_uniform_prior = @inferred(PosteriorMeasure(logfuncdensity(logdensityof(mvn)), uniform_prior))
    posterior_gaussian_prior = @inferred(PosteriorMeasure(logfuncdensity(logdensityof(mvn)), mvn))

    @test @inferred(bat_transform(PriorToNormal(), posterior_uniform_prior, context)).result.prior.dist == @inferred(BAT.StandardMvNormal(3))
    @test @inferred(bat_transform(PriorToUniform(), posterior_gaussian_prior, context)).result.prior.dist == @inferred(BAT.StandardMvUniform(3))
    @test @inferred(bat_transform(DoNotTransform(), posterior_uniform_prior, context)).result.prior.dist == uniform_prior
    pd = @inferred(product_distribution([Uniform() for i in 1:3]))
    density = @inferred(BAT.BATDistMeasure(pd))
    @test @inferred(bat_transform(DoNotTransform(), density, context)).result.dist == density.dist

    # ToDo: Improve comparison for bounds so `.dist` is not required here:
    @inferred(bat_transform(PriorToUniform(), batmeasure(BAT.StandardUvUniform()), context)).result.dist == batmeasure(BAT.StandardUvUniform()).dist
    @inferred(bat_transform(PriorToUniform(), batmeasure(BAT.StandardMvUniform(4)), context)).result.dist == batmeasure(BAT.StandardMvUniform(4)).dist
    @inferred(bat_transform(PriorToNormal(), batmeasure(BAT.StandardUvNormal()), context)).result.dist == batmeasure(BAT.StandardUvNormal()).dist
    @inferred(bat_transform(PriorToNormal(), batmeasure(BAT.StandardMvNormal(4)), context)).result.dist == batmeasure(BAT.StandardMvNormal(4)).dist
end
