# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra
using ValueShapes, Distributions, ArraysOfArrays, ForwardDiff

@testset "test_distribution_transform" begin
    function test_back_and_forth(trg_d, src_d)
        @testset "transform $(typeof(trg_d).name) <-> $(typeof(src_d).name)" begin
            src_v = rand(src_d)
            prev_ladj = 7.9

            @test @inferred(BAT.apply_dist_trafo(trg_d, src_d, src_v, prev_ladj)) isa NamedTuple{(:v,:ladj)}
            trg_v, trg_ladj = BAT.apply_dist_trafo(trg_d, src_d, src_v, prev_ladj)

            @test BAT.apply_dist_trafo(src_d, trg_d, trg_v, trg_ladj) isa NamedTuple{(:v,:ladj)}
            src_v_reco, prev_ladj_reco = BAT.apply_dist_trafo(src_d, trg_d, trg_v, trg_ladj)

            @test src_v ≈ src_v_reco
            @test prev_ladj ≈ prev_ladj_reco
            @test trg_ladj ≈ logabsdet(ForwardDiff.jacobian(x -> unshaped(BAT.apply_dist_trafo(trg_d, src_d, x, prev_ladj).v), src_v))[1] + prev_ladj
        end
    end

    function get_trgxs(trg_d, src_d, X)
        return (x -> BAT.apply_dist_trafo(trg_d, src_d, x, 0.0).v).(nestedview(X))
    end

    function get_trgxs(trg_d, src_d::Distribution{Univariate}, X)
        return (x -> BAT.apply_dist_trafo(trg_d, src_d, x, 0.0).v).(X)
    end

    function test_dist_trafo_moments(trg_d, src_d)
        @testset "check moments of trafo $(typeof(trg_d).name) <- $(typeof(src_d).name)" begin
            let trg_d = trg_d, src_d = src_d
                X = flatview(rand(src_d, 10^5))
                trgxs = get_trgxs(trg_d, src_d, X)
                unshaped_trgxs = map(unshaped, trgxs)
                @test isapprox(mean(unshaped_trgxs), mean(unshaped(trg_d)), atol = 0.1)
                @test isapprox(cov(unshaped_trgxs), cov(unshaped(trg_d)), rtol = 0.1)
            end
        end
    end

    uniform1 = Uniform(-5.0, -0.01)
    uniform2 = Uniform(0.01, 5.0)

    normal1 = Normal(-10, 1)
    normal2 = Normal(10, 5)

    uvnorm = BAT.StandardUvNormal()

    standnorm1 = BAT.StandardMvNormal(1)
    standnorm2 = BAT.StandardMvNormal(2)

    standuni2 = BAT.StandardMvUniform(2)

    standnorm2_reshaped = ReshapedDist(standnorm2, varshape(standnorm2))

    mvnorm = MvNormal([0.3, -2.9], [1.7 0.5; 0.5 2.3])
    beta = Beta(3,1)
    gamma = Gamma(0.1,0.7)
    dirich = Dirichlet([0.1,4])

    ntdist = NamedTupleDist(
        a = uniform1,
        b = mvnorm,
        c = [4.2, 3.7],
        x = beta,
        y = gamma
    )

    test_back_and_forth(beta, standnorm1)
    test_back_and_forth(gamma, standnorm1)

    test_back_and_forth(mvnorm, mvnorm)
    test_back_and_forth(mvnorm, standnorm2)
    test_back_and_forth(mvnorm, standuni2)

    test_dist_trafo_moments(normal2, normal1)
    test_dist_trafo_moments(uniform2, uniform1)

    test_dist_trafo_moments(beta, gamma)
    test_dist_trafo_moments(uvnorm, standnorm1)

    test_dist_trafo_moments(beta, standnorm1)
    test_dist_trafo_moments(gamma, standnorm1)

    test_dist_trafo_moments(mvnorm, standnorm2)
    test_dist_trafo_moments(dirich, standnorm1)

    test_dist_trafo_moments(mvnorm, standuni2)
    test_dist_trafo_moments(standuni2, mvnorm)

    test_dist_trafo_moments(standnorm2, standuni2)

    test_dist_trafo_moments(mvnorm, standnorm2_reshaped)
    test_dist_trafo_moments(standnorm2_reshaped, mvnorm)
    test_dist_trafo_moments(standnorm2, standnorm2_reshaped)
    test_dist_trafo_moments(standnorm2_reshaped, standnorm2_reshaped)
    
    test_back_and_forth(ntdist, BAT.StandardMvNormal(5))
    test_back_and_forth(ntdist, BAT.StandardMvUniform(5))

    let
        mvuni = product_distribution([Uniform(), Uniform()])

        x = rand()
        @test_throws ArgumentError BAT.apply_dist_trafo(uvnorm, mvnorm, x, 0.0)
        @test_throws ArgumentError BAT.apply_dist_trafo(uvnorm, standnorm1, x, 0.0)
        @test_throws ArgumentError BAT.apply_dist_trafo(uvnorm, standnorm2, x, 0.0)

        x = rand(2)
        @test_throws ArgumentError BAT.apply_dist_trafo(mvuni, mvnorm, x, 0.0)
        @test_throws ArgumentError BAT.apply_dist_trafo(mvnorm, mvuni, x, 0.0)
        @test_throws ArgumentError BAT.apply_dist_trafo(uvnorm, mvnorm, x, 0.0)
        @test_throws ArgumentError BAT.apply_dist_trafo(uvnorm, standnorm1, x, 0.0)
        @test_throws ArgumentError BAT.apply_dist_trafo(uvnorm, standnorm2, x, 0.0)
    end

    let
        primary_dist = NamedTupleDist(x = Normal(2), c = 5)
        f = x -> NamedTupleDist(y = Normal(x.x, 3), z = MvNormal([1.3 0.5; 0.5 2.2]))
        trg_d = @inferred(HierarchicalDistribution(f, primary_dist))
        src_d = BAT.StandardMvNormal(totalndof(varshape(trg_d)))
        test_back_and_forth(trg_d, src_d)
        test_dist_trafo_moments(trg_d, src_d)
    end


    #=
    using Cuba
    function integrate_over_unit(density::AbstractDensity)
        vs = varshape(density)
        f_cuba(source_x, y) = y[1] = exp(logvalof(density)(vs(source_x)))
        Cuba.vegas(f_cuba, 1, 1).integral[1]
    end
    =#

    @testset "Custom cdf and quantile for dual numbers" begin
        Dual = ForwardDiff.Dual

        @test BAT._trafo_cdf(Normal(Dual(0, 1, 0, 0), Dual(1, 0, 1, 0)), Dual(0.5, 0, 0, 1)) == cdf(Normal(Dual(0, 1, 0, 0), Dual(1, 0, 1, 0)), Dual(0.5, 0, 0, 1))
        @test BAT._trafo_cdf(Normal(0, 1), Dual(0.5, 1)) == cdf(Normal(0, 1), Dual(0.5, 1))

        @test BAT._trafo_quantile(Normal(0, 1), Dual(0.5, 1)) == quantile(Normal(0, 1), Dual(0.5, 1))
        @test BAT._trafo_quantile(Normal(Dual(0, 1, 0, 0), Dual(1, 0, 1, 0)), Dual(0.5, 0, 0, 1)) == quantile(Normal(Dual(0, 1, 0, 0), Dual(1, 0, 1, 0)), Dual(0.5, 0, 0, 1))
    end


    @testset "trafo broadcasting" begin
        dist = NamedTupleDist(a = Weibull(), b = Exponential())
        smpls = bat_sample(dist, IIDSampling(nsamples = 100)).result
        trafo = BAT.DistributionTransform(Normal, dist)
        @inferred(broadcast(trafo, smpls)) isa DensitySampleVector
        smpls_tr = trafo.(smpls)
        smpls_tr_cmp = [trafo(s) for s in smpls]
        @test smpls_tr == smpls_tr_cmp
	    @test @inferred(varshape(trafo)) == @inferred(varshape(dist)) == trafo.source_varshape
    end

    @testset "trafo composition" begin
        dist1 = @inferred(NamedTupleDist(a = Normal(), b = Uniform(), c = Cauchy()))
        dist2 = @inferred(NamedTupleDist(a = Exponential(), b = Weibull(), c = Beta()))
        normal1 = Normal()
        normal2 = Normal(2)

        trafo = @inferred(BAT.DistributionTransform(dist1, dist2))
        inv_trafo = @inferred(inv(trafo))

        composed_trafo = @inferred(∘(trafo, inv_trafo))
        @test composed_trafo.source_dist == composed_trafo.target_dist == dist1
        @test composed_trafo ∘ trafo == trafo
        @test_throws ArgumentError  trafo ∘ composed_trafo

        trafo = @inferred(BAT.DistributionTransform(normal1, normal2))
        @test_throws ArgumentError trafo ∘ trafo
    end

    @testset "full density transform" begin
        likelihood = @inferred(NamedTupleDist(a = Normal(), b = Exponential()))
        prior = product_distribution([Normal(), Gamma()])
        posterior_density = PosteriorDensity(likelihood, prior)

        posterior_density_trafod = @inferred(bat_transform(PriorToUniform(), posterior_density, FullDensityTransform()))

        @test posterior_density_trafod.result.orig.likelihood.dist == likelihood
        @test posterior_density_trafod.result.orig.prior.dist == prior

        @test posterior_density_trafod.result.trafo.target_dist isa BAT.StandardMvUniform

        lower_bounds = Float32.([-10, -10, -10])
        upper_bounds = Float32.([10, 10, 10])
        rect_bounds = @inferred(BAT.HyperRectBounds(lower_bounds, upper_bounds))
            mvn = @inferred(product_distribution(Normal.(randn(3))))
        dist_density = @inferred(BAT.DistributionDensity(mvn, rect_bounds))

        dist_density_trafod = @inferred(bat_transform(PriorToUniform(), dist_density, FullDensityTransform()))

        @test dist_density_trafod.trafo.target_dist isa BAT.StandardMvUniform
        @test dist_density_trafod.result.orig == dist_density
        @test dist_density_trafod.trafo.source_dist == dist_density_trafod.result.trafo.source_dist == mvn

        @test dist_density_trafod.trafo.target_varshape == @inferred(varshape(dist_density))

        dist_density_trafod = @inferred(bat_transform(PriorToGaussian(), dist_density, FullDensityTransform()))

        @test dist_density_trafod.trafo.target_dist isa BAT.StandardMvNormal
        @test dist_density_trafod.result.orig == dist_density
        @test dist_density_trafod.trafo.source_dist == dist_density_trafod.result.trafo.source_dist == mvn

        @test dist_density_trafod.trafo.target_varshape == @inferred(varshape(dist_density))
    end
end


@testset "bat_transform_defaults" begin
    mvn = @inferred(product_distribution([Normal(-1), Normal(), Normal(1)]))
    uniform_prior = @inferred(product_distribution([Uniform(-3, 1), Uniform(-2, 2), Uniform(-1, 3)]))

    posterior_uniform_prior = @inferred(PosteriorDensity(mvn, uniform_prior))
    posterior_gaussian_prior = @inferred(PosteriorDensity(mvn, mvn))

    @test @inferred(bat_transform(PriorToGaussian(), posterior_uniform_prior)).result.prior.dist == @inferred(BAT.StandardMvNormal(3))
    @test @inferred(bat_transform(PriorToUniform(), posterior_gaussian_prior)).result.prior.dist == @inferred(BAT.StandardMvUniform(3))
    @test @inferred(bat_transform(NoDensityTransform(), posterior_uniform_prior)).result.prior.dist == uniform_prior
    pd = @inferred(product_distribution([Uniform() for i in 1:3]))
    density = @inferred(BAT.DistributionDensity(pd))
    @test @inferred(bat_transform(NoDensityTransform(), density)).result.dist == density.dist

    # ToDo: Improve comparison for bounds so `.dist` is not required here:
    @inferred(bat_transform(PriorToUniform(), convert(AbstractDensity, BAT.StandardUvUniform()))).result.dist == convert(AbstractDensity, BAT.StandardUvUniform()).dist
    @inferred(bat_transform(PriorToUniform(), convert(AbstractDensity, BAT.StandardMvUniform(4)))).result.dist == convert(AbstractDensity, BAT.StandardMvUniform(4)).dist
    @inferred(bat_transform(PriorToGaussian(), convert(AbstractDensity, BAT.StandardUvNormal()))).result.dist == convert(AbstractDensity, BAT.StandardUvNormal()).dist
    @inferred(bat_transform(PriorToGaussian(), convert(AbstractDensity, BAT.StandardMvNormal(4)))).result.dist == convert(AbstractDensity, BAT.StandardMvNormal(4)).dist
end
