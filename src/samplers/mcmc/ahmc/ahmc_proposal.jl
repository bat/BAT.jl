abstract type HMCProposal end



@with_kw struct FixedStepNumber <: HMCProposal
    nsteps::Int64 = 10
end

function ahmc_proposal(
    proposal::FixedStepNumber,
    integrator::AdvancedHMC.AbstractIntegrator
)
    return AdvancedHMC.StaticTrajectory(integrator, proposal.nsteps)
end



@with_kw struct FixedTrajectoryLength <: HMCProposal
    trajectory_length::Float64 = 2.0
end

function ahmc_proposal(
    proposal::FixedTrajectoryLength,
    integrator::AdvancedHMC.AbstractIntegrator
)
    return AdvancedHMC.HMCDA(integrator, proposal.trajectory_length)
end



@with_kw struct NUTS <: HMCProposal
    sampling::Symbol = :MultinomialTS
    nuts::Symbol = :ClassicNoUTurn
end

function ahmc_proposal(
    proposal::NUTS,
    integrator::AdvancedHMC.AbstractIntegrator
)
    sampling_type = getfield(AdvancedHMC, proposal.sampling)
    nuts_type = getfield(AdvancedHMC, proposal.nuts)

    return AdvancedHMC.NUTS{sampling_type, nuts_type}(integrator)
end
