using ArraysOfArrays
using BAT
using Distributions
using HypothesisTests
using StatsBase

@testset "ahmc_sample" begin
    function test_mvn(; 
        mcalg::HamiltonianMC=HamiltonianMC(), 
        nsteps::Integer=10^4, 
        trafo::BAT.AbstractDensityTransformTarget=PriorToGaussian(),
        burnin::MCMCBurninAlgorithm=MCMCMultiCycleBurnin(),
        strict::Bool=true
        )
        
        sampling_algorithm = MCMCSampling(mcalg=mcalg, nsteps=nsteps, trafo=trafo, burnin=burnin, strict=strict)
        
        μ1 = rand(-5:0.1:-1); σ1 = randn(Float32) |> abs
        μ2 = rand(1:0.5:5); σ2 = randn(Float32) |> abs

        normal1 = Normal(μ1, σ1)
        normal2 = Normal(μ2, σ2)

        mvn = product_distribution([normal1, normal2])
        prior = product_distribution([Normal(), Normal()])
        posterior = PosteriorDensity(mvn, prior)


        #samples_hmc = bat_sample(mvn, sampling_algorithm).result.v |> flatview
        samples_hmc = bat_sample(posterior, sampling_algorithm).result.v |> flatview
        
        x1_hmc = samples_hmc[1, :]
        x2_hmc = samples_hmc[2, :]

        @test pvalue(ExactOneSampleKSTest(x1_hmc, normal1)) > 0.05
        @test pvalue(ExactOneSampleKSTest(x2_hmc, normal2)) > 0.05

    end
    @testset "Multivariate Gaussian" begin
        test_mvn()
    end
end