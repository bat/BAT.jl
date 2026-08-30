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

    if samplingalg.nchains == 1 && samplingalg.convergence isa Union{GelmanRubinConvergence,BrooksGelmanConvergence}
        throw(ArgumentError("$(nameof(typeof(samplingalg.convergence))) requires at least two chains. Use convergence = AssumeConvergence() to sample with one chain."))
    end

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

    ess = _mcmc_ess(
        chain_outputs,
        samples_transformed,
        samplingalg.proposal,
        samplingalg.sample_weighting,
        samplingalg.store_burnin,
        context,
    )
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
_proposal_diagnostics(proposal::MCMCProposalState, chain_state) =
    _proposal_diagnostics(proposal)

function _mcmc_diagnostics_summary(mcmc_states::AbstractVector{<:MCMCState})
    diags = [_proposal_diagnostics(s.chain_state.proposal, s.chain_state) for s in mcmc_states]
    return all(isnothing, diags) ? (;) : (chain_diagnostics = diags,)
end

# The default MCMC process ESS is computed separately for proposals whose
# walker processes are independent. Coupled proposals specialize `_mcmc_ess`.
# Weight provenance is still known here (unlike at the generic sample-vector
# level), so repetition weights can reconstruct each exact ordered process:
const _MCMC_PROCESS_ESS_MEMORY_BUDGET = 64 * 1024^2

function _mcmc_process_ess_memory_estimate(
    walker_output::DensitySampleVector,
    n_process_samples::Real,
)
    storage_type = eltype(flatview(walker_output.v))
    value_type = float(storage_type)
    concrete_bits_type = isconcretetype(storage_type) && isbitstype(storage_type) &&
        isconcretetype(value_type) && isbitstype(value_type)
    # Boxed or abstract values can allocate outside their array storage.
    concrete_bits_type || return Inf
    # `_resample_ac_ess` caps non-repetition resampling at ten times the stored sample count.
    # FFT autocorrelation uses several full process-value work buffers; for
    # repetition weighting this also covers run-length decoding buffers.
    return 16.0 * sizeof(value_type) * n_process_samples * totalndof(varshape(walker_output))
end

function _pooled_walker_ess(
    chain_outputs::AbstractVector{<:AbstractVector{<:DensitySampleVector}},
    merged_output::DensitySampleVector,
    weighting::AbstractMCMCWeightingScheme,
    context::BATContext
)
    isempty(merged_output) && return nothing
    rel_weights = _canonical_rel_weights(merged_output.weight)
    T = _weight_accum_type(rel_weights)
    is_repetition = weighting isa RepetitionWeighting
    estimated_bytes = 0.0
    masses = T[]
    process_lengths = Float64[]
    offset = 0
    for walker_outputs in chain_outputs, walker_output in walker_outputs
        isempty(walker_output) && continue
        next_offset = offset + length(walker_output)
        wsum = sum(T, view(rel_weights, (offset + 1):next_offset))
        offset = next_offset
        n_process_samples = is_repetition ?
            _validated_repetition_length(walker_output.weight) :
            10.0 * length(walker_output)
        push!(masses, wsum)
        push!(process_lengths, n_process_samples)
        wsum > 0 || continue
        estimated_bytes += _mcmc_process_ess_memory_estimate(walker_output, n_process_samples)
    end
    estimated_bytes <= _MCMC_PROCESS_ESS_MEMORY_BUDGET || return nothing

    ess_parts = Vector{Any}()
    walker_idx = 0
    # `_merge_chain_outputs` preserves this traversal order:
    for walker_outputs in chain_outputs, walker_output in walker_outputs
        isempty(walker_output) && continue
        walker_idx += 1
        iszero(masses[walker_idx]) && continue
        ess_w = if is_repetition
            _repetition_exact_ess(
                walker_output,
                EffSampleSizeFromAC(),
                context,
                process_lengths[walker_idx],
            )
        else
            bat_eff_sample_size_impl(walker_output, EffSampleSizeFromAC(), context).result
        end
        push!(ess_parts, ess_w)
    end
    filter!(x -> !iszero(x), masses)
    ess_pooled = _pooled_ess(ess_parts, masses)
    return isnothing(ess_pooled) ? nothing : minimum(ess_pooled)
end

function _mcmc_ess(
    chain_outputs::AbstractVector{<:AbstractVector{<:DensitySampleVector}},
    merged_output::DensitySampleVector,
    ::MCMCProposal,
    weighting::AbstractMCMCWeightingScheme,
    ::Bool,
    context::BATContext,
)
    return _pooled_walker_ess(chain_outputs, merged_output, weighting, context)
end

function _ensemble_mean_process(
    walker_outputs::AbstractVector{<:DensitySampleVector},
    n_process_samples::Float64,
)
    first_values = _repetition_exact_values(first(walker_outputs), n_process_samples)
    first_matrix = flatview(first_values)
    T = float(eltype(first_matrix))
    ensemble_mean = Matrix{T}(undef, size(first_matrix))
    copyto!(ensemble_mean, first_matrix)

    for i in 2:length(walker_outputs)
        values = _repetition_exact_values(walker_outputs[i], n_process_samples)
        values_matrix = flatview(values)
        count = T(i)
        inv_count = inv(count)
        previous_fraction = one(T) - inv_count
        for j in eachindex(ensemble_mean, values_matrix)
            previous_mean = ensemble_mean[j]
            value = T(values_matrix[j])
            # Same-sign subtraction is bounded; opposite signs require
            # bounded weighted terms instead of a potentially infinite delta.
            ensemble_mean[j] = if signbit(previous_mean) == signbit(value)
                previous_mean + (value - previous_mean) / count
            else
                previous_mean * previous_fraction + value * inv_count
            end
        end
    end
    return VectorOfSimilarVectors(ensemble_mean)
end

function _repetition_sweep_provenance(walker_output::DensitySampleVector)
    chainid = zero(Int32)
    walkerid = zero(Int32)
    chaincycle = zero(Int32)
    first_step = zero(Int64)
    next_step = zero(Int64)
    found_run = false

    for i in eachindex(walker_output.weight)
        weight = walker_output.weight[i]
        iszero(weight) && continue
        id = walker_output.info[i]
        id isa MCMCSampleID || return nothing
        id.chainid > 0 && id.walkerid > 0 && id.chaincycle > 0 || return nothing
        id.stepno >= 0 || return nothing
        weight < typemax(Int64) || return nothing
        count = Int64(weight)

        if !found_run
            chainid = id.chainid
            walkerid = id.walkerid
            chaincycle = id.chaincycle
            # The cycle's initial state is tagged as step zero; a positive
            # repetition count represents its post-transition sweeps from one.
            first_step = iszero(id.stepno) ? one(Int64) : id.stepno
            next_step = first_step
            found_run = true
        else
            id.chainid == chainid && id.walkerid == walkerid &&
                id.chaincycle == chaincycle && id.stepno == next_step || return nothing
        end
        count <= typemax(Int64) - next_step || return nothing
        next_step += count
    end

    found_run || return nothing
    return (;chainid, walkerid, chaincycle, first_step, next_step)
end

function _pooled_ensemble_ess(
    chain_outputs::AbstractVector{<:AbstractVector{<:DensitySampleVector}},
    merged_output::DensitySampleVector,
    context::BATContext,
)
    isempty(merged_output) && return nothing
    isempty(chain_outputs) && return nothing

    process_lengths = Float64[]
    masses = Float64[]
    chainids = Int32[]
    estimated_bytes = 0.0
    for walker_outputs in chain_outputs
        isempty(walker_outputs) && return nothing
        walkerids = Vector{Int32}(undef, length(walker_outputs))
        ensemble_provenance = nothing
        n_process_samples = 0.0
        for (i, output) in pairs(walker_outputs)
            process_length = _validated_repetition_length(output.weight)
            process_length > 0 || return nothing
            provenance = _repetition_sweep_provenance(output)
            isnothing(provenance) && return nothing

            if isnothing(ensemble_provenance)
                ensemble_provenance = provenance
                n_process_samples = process_length
            else
                process_length == n_process_samples || return nothing
                provenance.chainid == ensemble_provenance.chainid &&
                    provenance.chaincycle == ensemble_provenance.chaincycle &&
                    provenance.first_step == ensemble_provenance.first_step &&
                    provenance.next_step == ensemble_provenance.next_step || return nothing
            end
            provenance.walkerid in @view(walkerids[1:(i - 1)]) && return nothing
            walkerids[i] = provenance.walkerid
        end
        ensemble_provenance.chainid in chainids && return nothing
        push!(chainids, ensemble_provenance.chainid)

        push!(process_lengths, n_process_samples)
        push!(masses, length(walker_outputs) * n_process_samples)
        estimated_bytes += maximum(
            _mcmc_process_ess_memory_estimate(output, n_process_samples)
            for output in walker_outputs
        )
    end
    estimated_bytes <= _MCMC_PROCESS_ESS_MEMORY_BUDGET || return nothing

    algorithm = EffSampleSizeFromAC()
    ess_parts = Vector{Any}(undef, length(chain_outputs))
    for i in eachindex(chain_outputs)
        walker_outputs = chain_outputs[i]
        ensemble_mean = _ensemble_mean_process(walker_outputs, process_lengths[i])
        mean_ess = bat_eff_sample_size_impl(ensemble_mean, algorithm, context).result
        ess_parts[i] = length(walker_outputs) .* mean_ess
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
