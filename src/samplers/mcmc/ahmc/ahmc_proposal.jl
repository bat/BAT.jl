abstract type HMCProposal end

@with_kw struct FixedStepNumber <: HMCProposal
    n_steps::Int64 = 10
end

@with_kw struct FixedTrajectoryLength <: HMCProposal
    trajectory_length::Float64 = 2.0
end

@with_kw struct NUTS <: HMCProposal
    sampling::Symbol = :MultinomialTS
    nuts::Symbol = :ClassicNoUTurn
end



function AHMCProposal(
    proposal::FixedStepNumber,
    integrator::AdvancedHMC.AbstractIntegrator
)
    return AdvancedHMC.StaticTrajectory(integrator, proposal.n_steps)
end


function AHMCProposal(
    proposal::FixedTrajectoryLength,
    integrator::AdvancedHMC.AbstractIntegrator
)
    return AdvancedHMC.HMCDA(integrator, proposal.trajectory_length)
end


function AHMCProposal(
    proposal::NUTS,
    integrator::AdvancedHMC.AbstractIntegrator
)
    sampling_type = getfield(AdvancedHMC, proposal.sampling)
    nuts_type = getfield(AdvancedHMC, proposal.nuts)

    return AdvancedHMC.NUTS{sampling_type, nuts_type}(integrator)
end
