# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test
import AdvancedHMC
import LinearAlgebra


@testset "Metrics" begin
    diag_metric = DiagEuclideanMetric()
    @test isa(BAT.AHMCMetric(diag_metric, 3), AdvancedHMC.Adaptation.DiagEuclideanMetric{Float64,Array{Float64,1}})

    unit_metric = UnitEuclideanMetric()
    @test isa(BAT.AHMCMetric(unit_metric, 3), AdvancedHMC.Adaptation.UnitEuclideanMetric{Float64,Tuple{Int64}})

    dense_metric = DenseEuclideanMetric()
    @test isa(BAT.AHMCMetric(dense_metric, 3), AdvancedHMC.Adaptation.DenseEuclideanMetric{Float64,Array{Float64,1},Array{Float64,2},LinearAlgebra.UpperTriangular{Float64,Array{Float64,2}}})
end


@testset "Integrator" begin
    leapfrog = LeapfrogIntegrator(step_size=0.321)
    @test BAT.AHMCIntegrator(leapfrog) == AdvancedHMC.Leapfrog(0.321)

    jittered_leapfrog = JitteredLeapfrogIntegrator(step_size=0.321, jitter_rate=0.567)
    @test BAT.AHMCIntegrator(jittered_leapfrog) == AdvancedHMC.JitteredLeapfrog(0.321, 0.567)

    tempered_leapfrog = TemperedLeapfrogIntegrator(step_size=0.321, tempering_rate=0.567)
    @test BAT.AHMCIntegrator(tempered_leapfrog) == AdvancedHMC.TemperedLeapfrog(0.321, 0.567)
end


@testset "Proposal" begin
    integrator = AdvancedHMC.Leapfrog(0.321)

    fixed_steps = FixedStepNumber(n_steps=100)
    @test BAT.AHMCProposal(fixed_steps, integrator) == AdvancedHMC.StaticTrajectory(integrator, 100)

    fixed_length = FixedTrajectoryLength(trajectory_length=10.5)
    @test BAT.AHMCProposal(fixed_length, integrator) == AdvancedHMC.HMCDA(integrator, 10.5)


    @test BAT.AHMCProposal(NUTS(), integrator) == AdvancedHMC.NUTS{AdvancedHMC.MultinomialTS, AdvancedHMC.ClassicNoUTurn}(integrator)
    @test BAT.AHMCProposal(NUTS(:SliceTS, :GeneralisedNoUTurn), integrator) == AdvancedHMC.NUTS{AdvancedHMC.SliceTS, AdvancedHMC.GeneralisedNoUTurn}(integrator)
end


@testset "Adaptor" begin
    integrator = AdvancedHMC.Leapfrog(0.321)
    metric = AdvancedHMC.DiagEuclideanMetric(3)

    @test_broken BAT.AHMCAdaptor(MassMatrixAdaptor(), metric, integrator) ==  AdvancedHMC.MassMatrixAdaptor(metric)

    @test_broken BAT.AHMCAdaptor(StepSizeAdaptor(target_acceptance=0.5), metric, integrator) ==  AdvancedHMC.StepSizeAdaptor(0.5, integrator)


    mma = AdvancedHMC.MassMatrixAdaptor(metric)
    ssa = AdvancedHMC.StepSizeAdaptor(0.5, integrator)

    @test_broken BAT.AHMCAdaptor(NaiveHMCAdaptor(target_acceptance=0.5), metric, integrator) ==  AdvancedHMC.NaiveHMCAdaptor(mma, ssa)
    @test_broken BAT.AHMCAdaptor(StanHMCAdaptor(target_acceptance=0.5), metric, integrator) ==  AdvancedHMC.StanHMCAdaptor(mma, ssa)
end
