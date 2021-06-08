abstract type HMCProposal end

@with_kw struct FixedStepNumber{T<:Int64} <: HMCProposal
    nsteps::T = 10
end

@with_kw struct FixedTrajectoryLength{T<:Float64} <: HMCProposal
    trajectory_length::T = 2.0
end

@with_kw struct NUTS{T<:Symbol} <: HMCProposal
    sampling::T = :MultinomialTS
    nuts::T = :ClassicNoUTurn
end



function AHMCProposal(
    proposal::FixedStepNumber,
    integrator::AdvancedHMC.AbstractIntegrator
)
    return AdvancedHMC.StaticTrajectory(integrator, proposal.nsteps)
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
