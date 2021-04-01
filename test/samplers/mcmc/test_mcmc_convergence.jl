using BAT
using Test

using Distributed, Random
using ArraysOfArrays, Distributions, PDMats, StatsBase, ValueShapes

@testset "mcmc_convergence" begin
    dist1 = MultivariateNormal([1.0 0.; 0. 1.0])
    dist2 = product_distribution([Normal(0.25, 1.), Normal(0.25, 1.)])

    xs1 = [rand(dist1) for i in 1:10^5]
    ys1 = [logpdf(dist1, xi) for xi in xs1]
    chain_samples1 = @inferred(VectorOfVectors(xs1))
    dens_sampl_vec1 = @inferred(DensitySampleVector(chain_samples1, ys1))

    xs2 = [rand(dist2) for i in 1:10^5]
    ys2 = [logpdf(dist2, xi) for xi in xs2]
    chain_samples2 = @inferred(VectorOfVectors(xs2))
    dens_sampl_vec2 = @inferred(DensitySampleVector(chain_samples2, ys2))

    @testset "Brooks Gelman" begin
        @test isapprox(@inferred(BAT.bg_R_2sqr([dens_sampl_vec1, dens_sampl_vec1])), [1.0, 1.0], rtol=0.0001)
        @test isnan.(@inferred(BAT.bg_R_2sqr([dens_sampl_vec1, dens_sampl_vec1], corrected=true))) == [1, 1]

        convergence_criteria = @inferred(BrooksGelmanConvergence(threshold=Inf))
        convergence_result = @inferred(BAT.check_convergence(convergence_criteria, [dens_sampl_vec1, dens_sampl_vec1]))
        @test convergence_result.converged == true
        @test isapprox(convergence_result.max_Rsqr, 1.0, rtol=0.0001)

        convergence_criteria = @inferred(BrooksGelmanConvergence(threshold=0.9999))
        convergence_result = @inferred(BAT.check_convergence(convergence_criteria, [dens_sampl_vec1, dens_sampl_vec1]))
        @test convergence_result.converged == false
        @test isapprox(convergence_result.max_Rsqr, 1.0, rtol=0.0001)

        convergence_criteria = @inferred(BrooksGelmanConvergence(corrected=true))
        convergence_result = @inferred(BAT.check_convergence(convergence_criteria, [dens_sampl_vec1, dens_sampl_vec2]))
        @test convergence_result.converged == true
        @test 1.0 <= convergence_result.max_Rsqr <= convergence_criteria.threshold
    end

    @testset "Gelman Rubin" begin
        @test @inferred(BAT.gr_Rsqr([dens_sampl_vec1, dens_sampl_vec1])) == [1.0, 1.0]
        @test isapprox(@inferred(BAT.gr_Rsqr([dens_sampl_vec1, dens_sampl_vec1])), [1.0, 1.0], rtol=0.0001)

        convergence_criteria = @inferred(GelmanRubinConvergence(threshold=Inf))
        convergence_result = @inferred(BAT.check_convergence(convergence_criteria, [dens_sampl_vec1, dens_sampl_vec1]))
        @test convergence_result.converged == true
        @test isapprox(convergence_result.max_Rsqr, 1.0, rtol=0.0001)

        convergence_criteria = @inferred(GelmanRubinConvergence(threshold=0.9999))
        convergence_result = @inferred(BAT.check_convergence(convergence_criteria, [dens_sampl_vec1, dens_sampl_vec1]))
        @test convergence_result.converged == false
        @test isapprox(convergence_result.max_Rsqr, 1.0, rtol=0.0001)
    end
end