# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct TransformedMCMC <: AbstractSamplingAlgorithm

Samples a probability density using Markov chain Monte Carlo.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct TransformedMCMC{
    PR<:MCMCProposal,
    PRT<:MCMCProposalTuning,
    TR<:AbstractTransformTarget,
    AT<:AbstractAdaptiveTransform,
    ATT<:MCMCTransformTuning,
    TE<:MCMCTempering,
    IN<:MCMCInitAlgorithm,
    BI<:MCMCBurninAlgorithm,
    CT<:ConvergenceTest,
    WS<:AbstractMCMCWeightingScheme,
    CB<:Function
} <: AbstractSamplingAlgorithm
    proposal::PR = RandomWalk(proposaldist = TDist(1.0))
    proposal_tuning::PRT = bat_default(TransformedMCMC, Val(:proposal_tuning), proposal)
    pretransform::TR = bat_default(TransformedMCMC, Val(:pretransform), proposal)
    adaptive_transform::AT = bat_default(TransformedMCMC, Val(:adaptive_transform), proposal)
    transform_tuning::ATT = bat_default(TransformedMCMC, Val(:transform_tuning), adaptive_transform)
    tempering::TE = bat_default(TransformedMCMC, Val(:tempering), proposal)
    nchains::Int = 4
    nwalkers::Int = bat_default(TransformedMCMC, Val(:nwalkers), proposal, pretransform, transform_tuning, nchains)
    nsteps::Int = bat_default(TransformedMCMC, Val(:nsteps), proposal, pretransform, transform_tuning, nchains, nwalkers)
    #TODO: max_time ?
    init::IN = bat_default(TransformedMCMC, Val(:init), proposal, pretransform, transform_tuning, nchains, nwalkers, nsteps)
    burnin::BI = bat_default(TransformedMCMC, Val(:burnin), proposal, pretransform, transform_tuning, nchains, nwalkers, nsteps)
    convergence::CT = BrooksGelmanConvergence()
    strict::Bool = true
    store_burnin::Bool = false
    nonzero_weights::Bool = true
    sample_weighting::WS = RepetitionWeighting()
    callback::CB = nop_func
end
export TransformedMCMC


bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, ::CustomTransform) = NoMCMCTransformTuning()
bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, ::NoAdaptiveTransform) = NoMCMCTransformTuning()
bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, ::TriangularAffineTransform) = RAMTuning()

#TODO: MD, Decide what default to use trafo chain tuning. Use the defaults for the chain components?
bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, ::AdaptiveTransformChain) = NoMCMCTransformTuning()

function MCMCState(samplingalg::TransformedMCMC, target::BATMeasure, id::Integer, v_init::AbstractVector, context::BATContext)
    target_unevaluated = unevaluated(target)
    chain_state = MCMCChainState(samplingalg, target_unevaluated, Int32(id), v_init, context)
    trafo_tuner_state = create_trafo_tuner_state(samplingalg.transform_tuning, chain_state, 0)
    proposal_tuner_state = create_proposal_tuner_state(samplingalg.proposal_tuning, chain_state, chain_state.proposal, 0)
    temperer_state = create_temperering_state(samplingalg.tempering, target)
    
    MCMCState(chain_state, proposal_tuner_state, trafo_tuner_state, temperer_state)
end


bat_default(
    ::Type{TransformedMCMC}, 
    ::Val{:pretransform},
    ::MCMCProposal
) = PriorToNormal()

bat_default(
    ::Type{TransformedMCMC}, 
    ::Val{:nwalkers}, 
    ::MCMCProposal, 
    ::AbstractTransformTarget, 
    ::MCMCTransformTuning, 
    nchains::Integer
) = 1

bat_default(
    ::Type{TransformedMCMC}, 
    ::Val{:nsteps}, 
    ::MCMCProposal, 
    ::AbstractTransformTarget, 
    ::MCMCTransformTuning, 
    nchains::Integer, 
    nwalkers::Integer
) = 10^5

bat_default(
    ::Type{TransformedMCMC}, 
    ::Val{:init}, 
    ::MCMCProposal, 
    ::AbstractTransformTarget, 
    ::MCMCTransformTuning, 
    nchains::Integer, 
    nwalkers::Integer, 
    nsteps::Integer
) = MCMCChainPoolInit(nsteps_init = max(div(nsteps, 100), 250))

bat_default(
    ::Type{TransformedMCMC}, 
    ::Val{:burnin}, 
    ::MCMCProposal, 
    ::AbstractTransformTarget, 
    ::MCMCTransformTuning, 
    nchains::Integer, 
    nwalkers::Integer, 
    nsteps::Integer
) = MCMCMultiCycleBurnin(nsteps_per_cycle = max(div(nsteps, 10), 2500))

function bat_sample_impl(m::BATMeasure, samplingalg::TransformedMCMC, context::BATContext)
    transformed_m, f_pretransform = transform_and_unshape(samplingalg.pretransform, m, context)

    mcmc_states, chain_outputs = mcmc_init!(
        samplingalg,
        transformed_m,
        apply_trafo_to_init(f_pretransform, samplingalg.init), # TODO: MD: at which point should the init_alg be transformed? Might be better to read, if it's transformed later during init of states
        samplingalg.store_burnin ? samplingalg.callback : nop_func,
        context
    )

    if !samplingalg.store_burnin
        chain_outputs = _empty_chain_outputs.(mcmc_states)
    end
    
    mcmc_states = mcmc_burnin!(
        samplingalg.store_burnin ? chain_outputs : nothing,
        mcmc_states,
        samplingalg,
        samplingalg.store_burnin ? samplingalg.callback : nop_func
    )

    next_cycle!.(mcmc_states)

    mcmc_states = mcmc_iterate!!(
        chain_outputs,
        mcmc_states;
        max_nsteps = samplingalg.nsteps,
        nonzero_weights = samplingalg.nonzero_weights
    )

    samples_transformed = _merge_chain_outputs(first(mcmc_states), chain_outputs)

    smpls = inverse(f_pretransform).(samples_transformed)

    (result = smpls, result_trafo = samples_transformed, f_pretransform = f_pretransform, generator = MCMCSampleGenerator(mcmc_states))
end

function _merge_chain_outputs(mcmc_state::MCMCState, chain_outputs::AbstractVector{<:AbstractVector{<:DensitySampleVector}})
    merged_output = _empty_DensitySampleVector(mcmc_state)

    for walker_outputs in chain_outputs
        for walker_output in walker_outputs
            if !isempty(walker_output)
                append!(merged_output, walker_output)
            end
        end
    end

    return merged_output
end
