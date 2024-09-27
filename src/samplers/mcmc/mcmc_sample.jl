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
    TR<:AbstractTransformTarget,
    IN<:MCMCInitAlgorithm,
    BI<:MCMCBurninAlgorithm,
    CT<:ConvergenceTest,
    CB<:Function
} <: AbstractSamplingAlgorithm
    # TODO: MD, use bat_default to set default values
    pre_transform::TR = PriorToGaussian()
    trafo_tuning::MCMCTuning = AdaptiveMHTuning()
    proposal_tuning::MCMCTuning = trafo_tuning
    adaptive_transform::AdaptiveTransformSpec = default_adaptive_transform(trafo_tuning)
    proposal::MCMCProposal = MetropolisHastings(proposaldist = Normal())
    tempering = NoMCMCTempering() 
    nchains::Int = 4
    nsteps::Int = 10^5
    #TODO: max_time ?
    init::IN = MCMCChainPoolInit(nsteps_init = max(div(nsteps, 100), 250))
    burnin::BI = MCMCMultiCycleBurnin(nsteps_per_cycle = max(div(nsteps, 10), 2500))
    convergence::CT = BrooksGelmanConvergence()
    strict::Bool = true
    store_burnin::Bool = false
    nonzero_weights::Bool = true
    callback::CB = nop_func
end
export MCMCSampling


function MCMCState(sampling::MCMCSampling, target::BATMeasure, id::Integer, v_init::AbstractVector, context::BATContext)
    chain_state = MCMCChainState(sampling, target, Int32(id), v_init, context)
    trafo_tuner_state = create_trafo_tuner_state(sampling.trafo_tuning, chain_state, 0)
    proposal_tuner_state = create_proposal_tuner_state(sampling.proposal_tuning, chain_state, 0)
    temperer_state = create_temperering_state(sampling.tempering, target)
    
    MCMCState(chain_state, trafo_tuner_state, proposal_tuner_state, temperer_state)
end


bat_default(::MCMCSampling, ::Val{:pre_transform}) = PriorToGaussian()

bat_default(::MCMCSampling, ::Val{:nsteps}, trafo::AbstractTransformTarget, nchains::Integer) = 10^5

bat_default(::MCMCSampling, ::Val{:init}, trafo::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCChainPoolInit(nsteps_init = max(div(nsteps, 100), 250))

bat_default(::MCMCSampling, ::Val{:burnin}, trafo::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    MCMCMultiCycleBurnin(nsteps_per_cycle = max(div(nsteps, 10), 2500))

function bat_sample_impl(target::BATMeasure, sampling::MCMCSampling, context::BATContext)
    
    target_transformed, pre_transform = transform_and_unshape(sampling.pre_transform, target, context)

    mcmc_states, chain_outputs = mcmc_init!(
        sampling,
        target_transformed,
        apply_trafo_to_init(pre_transform, sampling.init), # TODO: MD: at which point should the init_alg be transformed? Might be better to read, if it's transformed later during init of states
        sampling.store_burnin ? sampling.callback : nop_func,
        context
    )

    if !sampling.store_burnin
        chain_outputs .= DensitySampleVector.(mcmc_states)
    end

    mcmc_burnin!(
        sampling.store_burnin ? chain_outputs : nothing,
        mcmc_states,
        sampling,
        sampling.store_burnin ? sampling.callback : nop_func
    )

    next_cycle!.(mcmc_states)

    mcmc_states = mcmc_iterate!!(
        chain_outputs,
        mcmc_states;
        max_nsteps = sampling.nsteps,
        nonzero_weights = sampling.nonzero_weights
    )

    samples_transformed = DensitySampleVector(first(mcmc_states))
    isempty(chain_outputs) || append!.(Ref(samples_transformed), chain_outputs)

    smpls = inverse(pre_transform).(samples_transformed)

    (result = smpls, result_trafo = samples_transformed, trafo = pre_transform, generator = MCMCSampleGenerator(mcmc_states))
end
