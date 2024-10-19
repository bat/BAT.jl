# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct MCMCSampling <: AbstractSamplingAlgorithm

Samples a probability density using Markov chain Monte Carlo.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MCMCSampling{
    PR<:MCMCProposal,
    PRT<:MCMCProposalTuning,
    TR<:AbstractTransformTarget,
    AT<:AbstractAdaptiveTransform,
    ATT<:MCMCTransformTuning,
    TE<:MCMCTempering,
    IN<:MCMCInitAlgorithm,
    BI<:MCMCBurninAlgorithm,
    CT<:ConvergenceTest,
    CB<:Function
} <: AbstractSamplingAlgorithm
    proposal::PR = RandomWalk(proposaldist = TDist(1.0))
    proposal_tuning::PRT = bat_default(MCMCSampling, Val(:proposal_tuning), proposal)
    pre_transform::TR = bat_default(MCMCSampling, Val(:pre_transform), proposal)
    adaptive_transform::AT = bat_default(MCMCSampling, Val(:adaptive_transform), proposal)
    transform_tuning::ATT = bat_default(MCMCSampling, Val(:transform_tuning), adaptive_transform)
    tempering::TE = bat_default(MCMCSampling, Val(:tempering), proposal)
    nchains::Int = 4
    nsteps::Int = bat_default(MCMCSampling, Val(:nsteps), proposal, pre_transform, nchains)
    #TODO: max_time ?
    init::IN = bat_default(MCMCSampling, Val(:init), proposal, pre_transform, nchains, nsteps)
    burnin::BI = bat_default(MCMCSampling, Val(:burnin), proposal, pre_transform, nchains, nsteps)
    convergence::CT = BrooksGelmanConvergence()
    strict::Bool = true
    store_burnin::Bool = false
    nonzero_weights::Bool = true
    callback::CB = nop_func
end
export MCMCSampling


bat_default(::Type{MCMCSampling}, ::Val{:transform_tuning}, ::CustomTransform) = NoMCMCTransformTuning()
bat_default(::Type{MCMCSampling}, ::Val{:transform_tuning}, ::NoAdaptiveTransform) = NoMCMCTransformTuning()
bat_default(::Type{MCMCSampling}, ::Val{:transform_tuning}, ::TriangularAffineTransform) = RAMTuning()


function MCMCState(samplingalg::MCMCSampling, target::BATMeasure, id::Integer, v_init::AbstractVector, context::BATContext)
    chain_state = MCMCChainState(samplingalg, target, Int32(id), v_init, context)
    trafo_tuner_state = create_trafo_tuner_state(samplingalg.transform_tuning, chain_state, 0)
    proposal_tuner_state = create_proposal_tuner_state(samplingalg.proposal_tuning, chain_state, 0)
    temperer_state = create_temperering_state(samplingalg.tempering, target)
    
    MCMCState(chain_state, proposal_tuner_state, trafo_tuner_state, temperer_state)
end


bat_default(::MCMCSampling, ::Val{:pre_transform}) = PriorToGaussian()

bat_default(::MCMCSampling, ::Val{:nsteps}, trafo::AbstractTransformTarget, nchains::Integer) = 10^5

bat_default(::MCMCSampling, ::Val{:init}, trafo::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCChainPoolInit(nsteps_init = max(div(nsteps, 100), 250))

bat_default(::MCMCSampling, ::Val{:burnin}, trafo::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCMultiCycleBurnin(nsteps_per_cycle = max(div(nsteps, 10), 2500))

function bat_sample_impl(target::BATMeasure, samplingalg::MCMCSampling, context::BATContext)
        
    target_transformed, pre_transform = transform_and_unshape(samplingalg.pre_transform, target, context)

    mcmc_states, chain_outputs = mcmc_init!(
        samplingalg,
        target_transformed,
        apply_trafo_to_init(pre_transform, samplingalg.init), # TODO: MD: at which point should the init_alg be transformed? Might be better to read, if it's transformed later during init of states
        samplingalg.store_burnin ? samplingalg.callback : nop_func,
        context
    )

    if !samplingalg.store_burnin
        chain_outputs .= DensitySampleVector.(mcmc_states)
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

    samples_transformed = DensitySampleVector(first(mcmc_states))
    isempty(chain_outputs) || append!.(Ref(samples_transformed), chain_outputs)

    smpls = inverse(pre_transform).(samples_transformed)

    (result = smpls, result_trafo = samples_transformed, trafo = pre_transform, generator = MCMCSampleGenerator(mcmc_states))
end
