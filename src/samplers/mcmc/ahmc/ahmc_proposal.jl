export StaticTrajectory
export HMCDA
export NUTS


abstract type AHMCProposal end

struct StaticTrajectory <: AHMCProposal
    n::Real
    StaticTrajectory(; n=10) = new(n)
end

struct HMCDA <: AHMCProposal
    len_traj::Real
    HMCDA(; len_traj=2) = new(len_traj)
end

struct NUTS<: AHMCProposal
    sampling::Symbol
    nuts::Symbol
    NUTS(; sampling = :MultinomialTS, nuts = :ClassicNoUTurn) = new(sampling, nuts)
end



function get_AHMCproposal(
    proposal::StaticTrajectory,
    integrator::AdvancedHMC.AbstractIntegrator
)
    return AdvancedHMC.StaticTrajectory(integrator, proposal.n)
end

function get_AHMCproposal(
    proposal::HMCDA,
    integrator::AdvancedHMC.AbstractIntegrator
)
    return AdvancedHMC.HMCDA(integrator, proposal.len_traj)
end

function get_AHMCproposal(
    proposal::NUTS,
    integrator::AdvancedHMC.AbstractIntegrator
)

    sampling_type = getfield(AdvancedHMC, proposal.sampling)
    nuts_type = getfield(AdvancedHMC, proposal.nuts)

    return AdvancedHMC.NUTS{sampling_type, nuts_type}(integrator)
end
