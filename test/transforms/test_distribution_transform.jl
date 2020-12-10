# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra
using ValueShapes, Distributions, ArraysOfArrays, ForwardDiff

@testset "test_distribution_transform" begin
    @testset "transform MvNormal" begin
        for src_d in (BAT.StandardMvNormal(2), BAT.StandardMvUniform(2))
            src_d = BAT.StandardMvUniform(2)
            trg_d = MvNormal([0.3, -2.9], [1.7 0.5; 0.5 2.3])
            src_v = rand(src_d)
            prev_ladj = 7.9

            @test @inferred(BAT.apply_dist_trafo(trg_d, src_d, src_v, prev_ladj)) isa NamedTuple{(:v,:ladj)}
            trg_v, trg_ladj = BAT.apply_dist_trafo(trg_d, src_d, src_v, prev_ladj)

            @test BAT.apply_dist_trafo(src_d, trg_d, trg_v, trg_ladj) isa NamedTuple{(:v,:ladj)}
            src_v_reco, prev_ladj_reco = BAT.apply_dist_trafo(src_d, trg_d, trg_v, trg_ladj)

            @test src_v ≈ src_v_reco
            @test prev_ladj ≈ prev_ladj_reco
            @test trg_ladj ≈ logabsdet(ForwardDiff.jacobian(x -> BAT.apply_dist_trafo(trg_d, src_d, x, prev_ladj).v, src_v))[1] + prev_ladj

            let trg_d = trg_d, src_d = src_d
                trgxs = (x -> BAT.apply_dist_trafo(trg_d, src_d, x, 0.0).v).(nestedview(rand(src_d, 10^5)))
                @test isapprox(mean(trgxs), mean(trg_d), rtol = 0.1)
                @test isapprox(cov(trgxs), cov(trg_d), rtol = 0.1)
            end
        end
    end
    

    #=
    using Cuba
    function integrate_over_unit(density::AbstractDensity)
        vs = varshape(density)
        f_cuba(source_x, y) = y[1] = exp(logvalof(density)(vs(source_x)))
        Cuba.vegas(f_cuba, 1, 1).integral[1]
    end
    =#
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
