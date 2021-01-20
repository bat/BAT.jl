# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra
using ValueShapes, Distributions, ArraysOfArrays, ForwardDiff

@testset "test_distribution_transform" begin
    function test_back_and_forth(trg_d::Distribution, src_d::BAT.StdMvDist)
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

    function test_dist_trafo_moments(trg_d::Distribution, src_d::BAT.StdMvDist)
        @testset "check moments of trafo $(typeof(trg_d).name) <- $(typeof(src_d).name)" begin
            let trg_d = trg_d, src_d = src_d
                X = rand(src_d, 10^5)
                trgxs = (x -> BAT.apply_dist_trafo(trg_d, src_d, x, 0.0).v).(nestedview(X))
                unshaped_trgxs = map(unshaped, trgxs)
                @test isapprox(mean(unshaped_trgxs), mean(unshaped(trg_d)), rtol = 0.1)
                @test isapprox(cov(unshaped_trgxs), cov(unshaped(trg_d)), rtol = 0.1)
            end
        end
    end

    standnorm1 = BAT.StandardMvNormal(1)
    standnorm2 = BAT.StandardMvNormal(2)

    mvnorm = MvNormal([0.3, -2.9], [1.7 0.5; 0.5 2.3])
    beta = Beta(3,1)
    gamma = Gamma(0.1,0.7)
    dirich = Dirichlet([0.1,4])

    test_back_and_forth(beta, standnorm1)
    test_back_and_forth(gamma, standnorm1)
    test_back_and_forth(mvnorm, standnorm2)

    test_dist_trafo_moments(beta, standnorm1)
    test_dist_trafo_moments(gamma, standnorm1)

    test_dist_trafo_moments(mvnorm, standnorm2)
    test_dist_trafo_moments(dirich, standnorm1)

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
