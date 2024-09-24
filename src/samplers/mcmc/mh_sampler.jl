# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    abstract type MHProposalDistTuning

Abstract type for Metropolis-Hastings tuning strategies for
proposal distributions.
"""
abstract type MHProposalDistTuning <: MCMCTuning end
export MHProposalDistTuning


"""
    struct MetropolisHastings <: MCMCAlgorithm

Metropolis-Hastings MCMC sampling algorithm.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MetropolisHastings{
    Q<:ContinuousDistribution,
    WS<:AbstractMCMCWeightingScheme,
} <: MCMCProposal
    proposaldist::Q = TDist(1.0)
    weighting::WS = RepetitionWeighting()
end

export MetropolisHastings

mutable struct MHProposalState{
    Q<:ContinuousDistribution,
    WS<:AbstractMCMCWeightingScheme,
} <: MCMCProposalState
    proposaldist::Q
    weighting::WS
end
export MHProposalState

function _create_proposal_state(
    proposal::MetropolisHastings, 
    target::BATMeasure, 
    context::BATContext, 
    v_init::AbstractVector{<:Real}, 
    rng::AbstractRNG
)
    return MHProposalState(proposal.proposaldist, proposal.weighting)
end

function _get_sampleid(proposal::MHProposalState, id::Int32, cycle::Int32, stepno::Int64, sampletype::Integer)
    return MCMCSampleID(id, cycle, stepno, sampletype), MCMCSampleID
end


const MHState = MCMCState{<:BATMeasure,
                          <:RNGPartition,
                          <:Function,
                          <:MHProposalState,
                          <:DensitySampleVector,
                          <:DensitySampleVector,
                          <:BATContext
} 

function _weights(
    proposal::MHProposalState{Q,<:RepetitionWeighting},
    p_accept::Real,
    accepted::Bool
) where Q
    if accepted
        (0, 1)
    else
        (1, 0)
    end
end

function _weights(
    proposal::MHProposalState{Q,<:ARPWeighting},
    p_accept::Real,
    accepted::Bool
) where Q
    T = typeof(p_accept)
    if p_accept ≈ 1
        (zero(T), one(T))
    elseif p_accept ≈ 0
        (one(T), zero(T))
    else
        (T(1 - p_accept), p_accept)
    end
end

eff_acceptance_ratio(mc_state::MHState) = nsamples(mc_state) / nsteps(mc_state)
