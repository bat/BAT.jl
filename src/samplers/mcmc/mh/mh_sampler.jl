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

function samples_available(mc_state::MHState)
    i = _current_sample_idx(mc_state)
    mc_state.samples.info.sampletype[i] == ACCEPTED_SAMPLE
end

function get_samples!(appendable, mc_state::MHState, nonzero_weights::Bool)::typeof(appendable)
    if samples_available(mc_state)
        samples = mc_state.samples

        for i in eachindex(samples)
            st = samples.info.sampletype[i]
            if (
                (st == ACCEPTED_SAMPLE || st == REJECTED_SAMPLE) &&
                (samples.weight[i] > 0 || !nonzero_weights)
            )
                push!(appendable, samples[i])
            end
        end
    end
    appendable
end

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
