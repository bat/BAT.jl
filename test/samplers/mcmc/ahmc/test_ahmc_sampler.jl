using BAT
using Distributions
using HypothesisTests
using Test

@testset "ahmc_sample" begin
    function test_mvn(; 
        num_dims::Integer = 2,
        mcalg::HamiltonianMC=HamiltonianMC(), 
        nsteps::Integer=10^4, 
        trafo::BAT.AbstractDensityTransformTarget=PriorToGaussian(),
        burnin::MCMCBurninAlgorithm=MCMCMultiCycleBurnin(),
        strict::Bool=true,
        test_name::String
        )
        @testset "$test_name" begin
        
            sampling_algorithm = MCMCSampling(mcalg=mcalg, nsteps=nsteps, trafo=trafo, burnin=burnin, strict=strict)
            
            μ = rand(num_dims)
            σ = rand(num_dims) |> x -> abs.(x)
    
            normal_dists = Vector{Normal{eltype(σ)}}(undef, num_dims)
            for i in eachindex(normal_dists)
                normal_dists[i] = Normal(μ[i], σ[i])
            end
    
            mvnormal = product_distribution(normal_dists)
    
            samples = bat_sample(mvnormal, sampling_algorithm)
            X = samples.result.v.data
    
            @testset "KS Test" begin
                for i in 1:num_dims
                    @test pvalue(ExactOneSampleKSTest(unique(X[i, :]), normal_dists[i]), tail=:both) > 0.05
                end
            end

            @testset "chain" begin
                chain = samples.generator.chains[1]
                chain_info = chain.info
                @test @inferred(BAT.mcmc_info(chain)) == chain_info
                @test chain_info.converged == true
                @test chain_info.tuned == true
            end
        end
        nothing
    end
    @testset "Multivariate Gaussian" begin
        tuning = BAT.StanHMCTuning(target_acceptance=0.90)
        mcalg = HamiltonianMC(tuning=tuning)
        test_mvn(num_dims=4, mcalg=mcalg, test_name="NUTS")
    end
end
