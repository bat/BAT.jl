abstract type HMCProposal end



@with_kw struct FixedStepNumber <: HMCProposal
    nsteps::Int64 = 10
end

function ahmc_proposal(
    proposal::FixedStepNumber,
    integrator::AdvancedHMC.AbstractIntegrator
    )
    @unpack nsteps = proposal
    return AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(integrator, AdvancedHMC.FixedNSteps(nsteps))

end



@with_kw struct FixedTrajectoryLength <: HMCProposal
    trajectory_length::Float64 = 2.0
end

function ahmc_proposal(
    proposal::FixedTrajectoryLength,
    integrator::AdvancedHMC.AbstractIntegrator
)
    @unpack trajectory_length = proposal
    return AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(integrator, AdvancedHMC.FixedIntegrationTime(trajectory_length))
end

struct NUTSProposal <: HMCProposal end

function ahmc_proposal(
    proposal::NUTSProposal,
    integrator::AdvancedHMC.AbstractIntegrator
)
    return AdvancedHMC.NUTS{AdvancedHMC.MultinomialTS, AdvancedHMC.ClassicNoUTurn}(integrator)
end
