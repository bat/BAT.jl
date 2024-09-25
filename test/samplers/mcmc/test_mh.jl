# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using LinearAlgebra
using StatsBase, Distributions, StatsBase, ValueShapes, ArraysOfArrays, DensityInterface

@testset "MetropolisHastings" begin
    context = BATContext()
    objective = NamedTupleDist(a = Normal(1, 1.5), b = MvNormal([-1.0, 2.0], [2.0 1.5; 1.5 3.0]))

    shaped_target = @inferred(batmeasure(objective))
    @test shaped_target isa BAT.BATDistMeasure
    target = unshaped(shaped_target)
    @test target isa BAT.BATDistMeasure

    # These are default values in MCMCSampling()
    proposal = MetropolisHastings()
    nchains = 4

    sampling = MCMCSampling()
 
    @testset "MCMC iteration" begin
        v_init = bat_initval(target, InitFromTarget(), context).result
        # TODO: MD, Reactivate type inference tests
        # @test @inferred(BAT.MCMCState(sampling, target, 1, unshaped(v_init, varshape(target)), deepcopy(context))) isa BAT.MHState
        # chain = @inferred(BAT.MCMCState(sampling, target, 1, unshaped(v_init, varshape(target)), deepcopy(context))) 
        chain = BAT.MCMCState(sampling, target, 1, unshaped(v_init, varshape(target)), deepcopy(context)) 
        samples = DensitySampleVector(chain)
        BAT.mcmc_iterate!(samples, chain, max_nsteps = 10^5, nonzero_weights = false)
        @test chain.stepno == 10^5
        @test minimum(samples.weight) == 0
        @test isapprox(length(samples), 10^5, atol = 20)
        @test length(samples) == sum(samples.weight)
        @test isapprox(mean(samples), [1, -1, 2], atol = 0.2)
        @test isapprox(cov(samples), cov(unshaped(objective)), atol = 0.3)

        samples = DensitySampleVector(chain)
        BAT.mcmc_iterate!(samples, chain, max_nsteps = 10^3, nonzero_weights = true)
        @test minimum(samples.weight) == 1
    end
 
    @testset "MCMC tuning and burn-in" begin
        init_alg = MCMCChainPoolInit()
        tuning_alg = AdaptiveMHTuning()
        burnin_alg = MCMCMultiCycleBurnin()
        convergence_test = BrooksGelmanConvergence()
        strict = true
        nonzero_weights = false
        callback = (x...) -> nothing
        max_nsteps = 10^5

        sampling = MCMCSampling(
            proposal = proposal,
            tuning = tuning_alg,
            burnin = burnin_alg,
            nchains = nchains,
            convergence = convergence_test,
            strict = true,
            nonzero_weights = nonzero_weights
        )

        init_result = @inferred(BAT.mcmc_init!(
            sampling,
            target,
            init_alg,
            callback,
            context
        ))

        (chains, tuners, outputs) = init_result

        # TODO: MD, Reactivate, for some reason fail
        # @test chains isa AbstractVector{<:BAT.MHState}
        # @test tuners isa AbstractVector{<:BAT.ProposalCovTunerState}
        @test outputs isa AbstractVector{<:DensitySampleVector}

        BAT.mcmc_burnin!(
            outputs,
            tuners,
            chains,
            sampling,
            callback
        )

        BAT.mcmc_iterate!(
            outputs,
            chains;
            max_nsteps = div(max_nsteps, length(chains)),
            nonzero_weights = nonzero_weights
        )

        samples = DensitySampleVector(first(chains))
        append!.(Ref(samples), outputs)
        
        @test length(samples) == sum(samples.weight)
        @test BAT.test_dist_samples(unshaped(objective), samples)
    end

    @testset "bat_sample" begin
        samples = bat_sample(
            shaped_target,
            MCMCSampling(
                proposal = proposal,
                pre_transform = DoNotTransform(),
                store_burnin = true
            ),
            context
        ).result

        @test first(samples).info.chaincycle == 1

        smplres = BAT.sample_and_verify(
            shaped_target,
            MCMCSampling(
                proposal = proposal,
                pre_transform = DoNotTransform()
            ),
            objective
        )
        samples = smplres.result
        @test first(samples).info.chaincycle >= 2
        @test samples.v isa ShapedAsNTArray
        @test smplres.verified
    end

    @testset "MCMC sampling in transformed space" begin
        prior = BAT.example_posterior().prior
        likelihood = logfuncdensity(v -> 0)
        inner_posterior = PosteriorMeasure(likelihood, prior)
        # Test with nested posteriors:
        posterior = PosteriorMeasure(likelihood, inner_posterior)
        @test BAT.sample_and_verify(posterior, MCMCSampling(proposal = MetropolisHastings(), pre_transform = PriorToGaussian()), prior.dist).verified
    end
end

# Test with a custom proposal distribution                                                                                                                                                                                                                                                                                                                                  _A<:Function, _B<:BAT.MCMCProposalState, _C<:DensitySample                                                                
#MCMCState{BAT.BATDistMeasure{ValueShapes.UnshapedNTD{NamedTupleDist{(:a, :b), Tuple{Normal{Float64}, FullNormal}, Tuple{ValueAccessor{ScalarShape{Real}}, ValueAccessor{ArrayShape{Real, 1}}}, NamedTuple}}}, BAT.RNGPartition{Philox4x{UInt64, 10}, Tuple{UInt64, UInt64}, NTuple{6, UInt32}, UnitRange{Int64}}, AffineMaps.Mul{LowerTriangular{Float64, Matrix{Float64}}}, MHProposalState{Normal{Float64}, RepetitionWeighting{Int64}}, DensitySample{Vector{Float64},Float64, Int64, BAT.MCMCSampleID, Nothing}, StructArrays.StructVector{DensitySample{Vector{Float64}, Float64, Int64, BAT.MCMCSampleID, Nothing}, NamedTuple{(:v, :logd, :weight, :info, :aux), Tuple{ArrayOfSimilarArrays{Float64, 1, 1, 2, ElasticMatrix{Float64, Vector{Float64}}}, Vector{Float64}, Vector{Int64}, StructArrays.StructVector{BAT.MCMCSampleID, NamedTuple{(:chainid, :chaincycle, :stepno, :sampletype), Tuple{Vector{Int32}, Vector{Int32}, Vector{Int64}, Vector{Int64}}}, Int64}, Vector{Nothing}}}, Int64}, BATContext{Float64, Philox4x{UInt64, 10}, HeterogeneousComputing.CPUnit, BAT._NoADSelected}} 
#MCMCState{BAT.BATDistMeasure{ValueShapes.UnshapedNTD{NamedTupleDist{(:a, :b), Tuple{Normal{Float64}, FullNormal}, Tuple{ValueAccessor{ScalarShape{Real}}, ValueAccessor{ArrayShape{Real, 1}}}, NamedTuple}}}, BAT.RNGPartition{Philox4x{UInt64, 10}, Tuple{UInt64, UInt64}, NTuple{6, UInt32}, UnitRange{Int64}}, _A,                                                       _B,                                                            _C,                                                                       StructArrays.StructVector{DensitySample{Vector{Float64}, Float64, Int64, BAT.MCMCSampleID, Nothing}, NamedTuple{(:v, :logd, :weight, :info, :aux), Tuple{ArrayOfSimilarArrays{Float64, 1, 1, 2, ElasticMatrix{Float64, Vector{Float64}}}, Vector{Float64}, Vector{Int64}, StructArrays.StructVector{BAT.MCMCSampleID, NamedTuple{(:chainid, :chaincycle, :stepno, :sampletype), Tuple{Vector{Int32}, Vector{Int32}, Vector{Int64}, Vector{Int64}}}, Int64}, Vector{Nothing}}}, Int64}, BATContext{Float64, Philox4x{UInt64, 10}, HeterogeneousComputing.CPUnit, BAT._NoADSelected}}