# This file is a part of BAT.jl, licensed under the MIT License (MIT).

abstract type MCMCProposal end

struct MHProposal{
    D<:Union{Distribution, AbstractMeasure},
    WS<:AbstractMCMCWeightingScheme
}<: TransformedMCMCProposal
    proposal_dist::D
end

# TODO AC: find a better solution for this. Problem is that in the with_kw constructor below, we need to dispatch on this type.
struct TransformedMCMCDispatch end


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
    BI<:TransformedMCMCBurninAlgorithm,
    CT<:ConvergenceTest,
    CB<:Function
} <: AbstractSamplingAlgorithm
    pre_transform::TR = bat_default(TransformedMCMCDispatch, Val(:pre_transform))
    tuning_alg::MCMCTuningAlgorithm = TransformedRAMTuner() # TODO: use bat_defaults
    adaptive_transform::AdaptiveTransformSpec = default_adaptive_transform(tuning_alg)
    proposal::MCMCProposal = TransformedMHProposal(Normal(), RepetitionWeighting()) #TODO: use bat_defaults
    tempering = NoMCMCTempering() # TODO: use bat_defaults (was TransformedNoTransformedMCMCTempering in RenameTransformed)
    nchains::Int = 4
    nsteps::Int = 10^5
    #TODO: max_time ?
    init::IN =  bat_default(TransformedMCMCDispatch, Val(:init), pre_transform, nchains, nsteps) #MCMCChainPoolInit()#TODO AC: use bat_defaults bat_default(MCMCSampling, Val(:init), MetropolisHastings(), pre_transform, nchains, nsteps) #TODO
    burnin::BI = bat_default(TransformedMCMCDispatch, Val(:burnin), pre_transform, nchains, nsteps)
    convergence::CT = BrooksGelmanConvergence()
    strict::Bool = true
    store_burnin::Bool = false
    nonzero_weights::Bool = true
    callback::CB = nop_func
end
export MCMCSampling


function bat_sample_impl(target::BATMeasure, sampling::MCMCSampling, context::BATContext)
    
    target_transformed, pre_transform = transform_and_unshape(sampling.pre_transform, target, context)

    (chains, tuners, chain_outputs, temperers) = mcmc_init!(
        sampling,
        target_transformed,
        apply_trafo_to_init(pre_transform, sampling.init), # TODO: MD: at which point should the init_alg be transformed? Might be better to read, if it's transformed later during init of states
        sampling.store_burnin ? sampling.callback : nop_func,
        context
    )

    if !sampling.store_burnin
        chain_outputs .= DensitySampleVector.(chains)
    end

    mcmc_burnin!(
        sampling.store_burnin ? chain_outputs : nothing,
        tuners,
        chains,
        sampling.burnin,
        sampling.convergence,
        sampling.strict,
        sampling.nonzero_weights,
        sampling.store_burnin ? sampling.callback : nop_func
    )

    next_cycle!.(chains)

    mcmc_iterate!(
        chain_outputs,
        chains;
        max_nsteps = sampling.nsteps,
        nonzero_weights = sampling.nonzero_weights,
        callback = sampling.callback
    )

    samples_transformed = DensitySampleVector(first(chains))
    isempty(chain_outputs) || append!.(Ref(samples_transformed), chain_outputs)

    smpls = inverse(trafo).(samples_transformed)

    (result = smpls, result_trafo = samples_transformed, trafo = trafo, generator = MCMCSampleGenerator(chains))
end
