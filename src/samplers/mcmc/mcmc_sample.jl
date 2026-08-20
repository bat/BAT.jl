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
    TR<:TransformIntent,
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
    transform_tuning::ATT = bat_default(TransformedMCMC, Val(:transform_tuning), proposal, adaptive_transform)
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


# The transform-tuning default depends on the proposal as well: the tuning
# rule must match the statistics the proposal generates (see e.g.
# FisherTransformTuning for gradient-based proposals vs. RAMTuning for
# random-walk proposals):
bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, ::MCMCProposal, ::CustomTransform) = NoMCMCTransformTuning()
bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, ::MCMCProposal, ::NoAdaptiveTransform) = NoMCMCTransformTuning()
bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, ::MCMCProposal, ::TriangularAffineTransform) = RAMTuning()

function bat_default(TM::Type{TransformedMCMC}, tt::Val{:transform_tuning}, proposal::MCMCProposal, f_transform::AdaptiveTransformChain)
    tunings = bat_default.(TM, tt, Ref(proposal), f_transform.f)
    # Fail at configuration time, not later during state creation: the
    # per-component defaults for gradient-based proposals are score-based
    # (Fisher), which transform chains don't support yet:
    if any(t -> t isa FisherTransformTuning, tunings)
        throw(ArgumentError(
            "The default transform tuning for $(nameof(typeof(proposal))) components is score-based (FisherTransformTuning), which is not supported inside an AdaptiveTransformChain yet - please specify a supported transform_tuning (e.g. MultiTrafoTuning of RAMTuning components) explicitly"
        ))
    end
    return MultiTrafoTuning(Tuple(tunings))
end


function MCMCState(samplingalg::TransformedMCMC, target::BATMeasure, id::Integer, v_init::AbstractVector, context::BATContext)
    target_unevaluated = unevaluated(target)
    chain_state = MCMCChainState(samplingalg, target_unevaluated, Int32(id), v_init, context)
    trafo_tuner_state = create_trafo_tuner_state(samplingalg.transform_tuning, chain_state, 0, samplingalg.adaptive_transform)
    proposal_tuner_state = create_proposal_tuner_state(samplingalg.proposal_tuning, chain_state, chain_state.proposal, 0)
    temperer_state = create_temperering_state(samplingalg.tempering, target)
    
    MCMCState(chain_state, proposal_tuner_state, trafo_tuner_state, temperer_state)
end


bat_default(
    ::Type{TransformedMCMC}, 
    ::Val{:pretransform},
    ::MCMCProposal
) = NormalBased()

bat_default(
    ::Type{TransformedMCMC}, 
    ::Val{:nwalkers}, 
    ::MCMCProposal, 
    ::TransformIntent, 
    ::MCMCTransformTuning, 
    nchains::Integer
) = 1

bat_default(
    ::Type{TransformedMCMC}, 
    ::Val{:nsteps}, 
    ::MCMCProposal, 
    ::TransformIntent, 
    ::MCMCTransformTuning, 
    nchains::Integer, 
    nwalkers::Integer
) = 10^5

bat_default(
    ::Type{TransformedMCMC}, 
    ::Val{:init}, 
    ::MCMCProposal, 
    ::TransformIntent, 
    ::MCMCTransformTuning, 
    nchains::Integer, 
    nwalkers::Integer, 
    nsteps::Integer
) = MCMCChainPoolInit(nsteps_init = max(div(nsteps, 100), 250))

bat_default(
    ::Type{TransformedMCMC}, 
    ::Val{:burnin}, 
    ::MCMCProposal, 
    ::TransformIntent, 
    ::MCMCTransformTuning, 
    nchains::Integer, 
    nwalkers::Integer, 
    nsteps::Integer
) = MCMCMultiCycleBurnin(nsteps_per_cycle = max(div(nsteps, 10), 2500))

function evalmeasure_impl(em::EvaluatedMeasure, samplingalg::TransformedMCMC, context::BATContext)
    # ToDo: Warm-restart from em.samplegen if available and compatible.

    transformed_m, f_pretransform = transform_and_unshape(samplingalg.pretransform, em, context)
    n_dof = some_dof(transformed_m)

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

    @info "Generate main samples using $(length(mcmc_states)) MCMC chain(s)."
    mcmc_states = mcmc_iterate!!(
        chain_outputs,
        mcmc_states;
        max_nsteps = samplingalg.nsteps,
        nonzero_weights = samplingalg.nonzero_weights
    )

    @debug "Merge samples of chains and transform to original space."

    samples_transformed = _merge_chain_outputs(first(mcmc_states), chain_outputs)

    smpls = transform_samples(inverse(f_pretransform), samples_transformed)

    samplegen = MCMCSampleGenerator(mcmc_states)

    ess = _pooled_walker_ess(chain_outputs, samplingalg.sample_weighting, context)
    dsm = DensitySampleMeasure(smpls, dof = n_dof, ess = ess)

    # The samples and the bare target measure in the transformed space are
    # preserved so that follow-up evaluations with the same transform intent
    # need neither sample transport nor measure reconstruction:
    return EvaluatedMeasure(em;
        transform_intent = samplingalg.pretransform,
        f_transform = _viewrep_f(f_pretransform, samplingalg.pretransform),
        empirical = _viewrep_empirical(dsm, samples_transformed, f_pretransform, samplingalg.pretransform, n_dof, ess),
        # ToDo:
        # approx = ...,
        dof = n_dof,
        samplegen = samplegen,
        transformed = _viewrep_measure(transformed_m, samplingalg.pretransform),
        evalinfo = MeasureEvalInfo(samplingalg, _mcmc_diagnostics_summary(mcmc_states))
    )
end

# Per-chain trajectory diagnostics (whole run, including warmup), for
# proposals that record them:
_proposal_diagnostics(::MCMCProposalState) = nothing

function _mcmc_diagnostics_summary(mcmc_states::AbstractVector{<:MCMCState})
    diags = [_proposal_diagnostics(get_active_proposal(s.chain_state.proposal)) for s in mcmc_states]
    return all(isnothing, diags) ? (;) : (chain_diagnostics = diags,)
end

# Autocorrelation ESS is a property of the ordered stochastic process, not
# of its empirical measure: it must be computed on each walker's ordered
# output sequence separately, before chains and walkers are merged. The
# independent per-walker contributions are then pooled with the walkers'
# empirical mass fractions (see `_pooled_ess`) - a plain sum would
# overstate the merged estimator's effective size whenever mixing
# efficiency differs between walkers. Weight provenance is still known
# here (unlike at the generic sample-vector level, which deliberately
# erases it), so repetition weights are reconstructed into the exact
# ordered chain:
function _pooled_walker_ess(
    chain_outputs::AbstractVector{<:AbstractVector{<:DensitySampleVector}},
    weighting::AbstractMCMCWeightingScheme,
    context::BATContext
)
    ess_parts = Vector{Any}()
    masses = Float64[]
    for walker_outputs in chain_outputs, walker_output in walker_outputs
        isempty(walker_output) && continue
        wsum = sum(float, walker_output.weight)
        wsum > 0 || continue
        ess_w = if weighting isa RepetitionWeighting
            _repetition_exact_ess(walker_output, EffSampleSizeFromAC(), context)
        else
            bat_eff_sample_size_impl(walker_output, EffSampleSizeFromAC(), context).result
        end
        push!(ess_parts, ess_w)
        push!(masses, wsum)
    end
    ess_pooled = _pooled_ess(ess_parts, masses)
    return isnothing(ess_pooled) ? nothing : minimum(ess_pooled)
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
