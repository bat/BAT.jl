# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# For validated positive thresholds and finite factors >= 1. Saturate before
# converting to Int, and prevent Float64 rounding from lowering the threshold.
function _scaled_buffer_threshold(n::Int, factor::Float64)
    value = n * factor
    value >= typemax(Int) && return typemax(Int)
    return max(n, ceil(Int, value))
end

# Counts completed walker-steps, independently of row merging or sample weights.
_add_buffer_steps(n::Int, steps::Int) = n > typemax(Int) - steps ? typemax(Int) : n + steps

# Called on the producer task while its current state is stable. Every published
# snapshot owns its sample data and is replaced, never mutated, by later updates.
function _pending_walker_outputs(chain_state::MCMCChainState, nonzero_weights::Bool)
    pending = _empty_chain_outputs(chain_state)
    for i in eachindex(pending)
        sample = chain_state.current.x[i]
        if !nonzero_weights || sample.weight > 0
            push!(pending[i], deepcopy(sample))
        end
    end
    return pending
end

function register_state_for_vis!(
    vis::BATVisualizer{BATMakieVisualization},
    mcmc_state::MCMCState,
    samples,
    timing = _empty_walker_spans(samples)
)
    (; graph, chain_ids, output_buffer, pending_samples) = vis.content

    empty_walker_outputs = _empty_chain_outputs(mcmc_state)
    samples_new, timing_new = _append_timed_walkers(mcmc_state, empty_walker_outputs, samples, _empty_walker_spans(samples), timing)

    samples_graph = graph[:samples][]
    push!(samples_graph, samples_new)

    current_idxs = length.(samples_new)
    current_idxs_graph = graph[:current_idxs][]
    push!(current_idxs_graph, current_idxs)

    # One batched update! -- see flush_buffer!'s matching comment (each update! call is a full invalidate/notify cycle of its own).
    graph_timing = vis.content.step_mode[] ?
        vcat(isnothing(graph[:sample_timing][]) ? ChainStepSpans[] : graph[:sample_timing][], [timing_new]) : nothing
    update!(graph; samples=samples_graph, current_idxs=current_idxs_graph, sample_timing=graph_timing)

    push!(chain_ids, mcmc_state.chain_state.info.id)
    empty_chain_output = _empty_chain_outputs(mcmc_state)
    push!(output_buffer, empty_chain_output)
    push!(pending_samples, _empty_chain_outputs(mcmc_state))
    push!(vis.content.output_timing, _empty_walker_spans(samples))
    push!(vis.content.pending_timing, _empty_walker_spans(samples))
end

# Core of the vsel choke point, usable without a live BATVisualizer (static bat_makie_plot path); clamps against n_dof/N_max and updates only on change.
function _apply_vsel_to_graph!(graph::ComputeGraph, n_dof::Integer, N_max::Integer, requested_vsel::AbstractVector{<:Integer})
    clamped = _clamp_vsel(requested_vsel, n_dof, N_max)
    if clamped != graph[:idxs][]
        update!(graph, idxs=clamped)
    end
    return clamped
end

# Live-path vsel entry point; buffer_lock guards against racing the listener task.
function _apply_vsel!(vis::BATVisualizer{BATMakieVisualization}, requested_vsel::AbstractVector{<:Integer})
    (; graph, buffer_lock, n_dof) = vis.content
    lock(buffer_lock) do
        _apply_vsel_to_graph!(graph, n_dof[], vis.backend.N_max, requested_vsel)
    end
    return nothing
end

function BAT.init_visualizer!(
    vis::BATVisualizer{BATMakieVisualization};
    rng::AbstractRNG,
    mcmc_states::Vector{<:MCMCState},
    outputs,#::Vector{Vector{DensitySampleVector}},
    f_pretransform::Function,
    # The ORIGINAL untransformed measure (bat_sample's `m`); nothing falls back to the transformed target and disables support-derived KDE boundary correction.
    target = nothing
)
    # A BATVisualizer is single-use: its content IS one run's live state; reuse
    # would append a second run's chains and deadlock its samplers on the latched is_live/backpressure.
    if isnothing(vis.content) || !vis.content.is_live[] || !isempty(vis.content.chain_ids)
        throw(ArgumentError(
            "This BATVisualizer has already been used by a previous bat_sample run " *
            "(or is a stripped copy from a deepcopied BATContext, e.g. out of a " *
            "result's provenance info) -- create a fresh " *
            "BATVisualizer(BATMakieVisualization(...)) and a fresh BATContext " *
            "carrying it for each run."))
    end

    # Domain estimates must not advance the sampler's RNG or the default RNG.
    # Copy the complete state, including any partially consumed random block.
    domain_rng = copy(rng)

    warmup_makie_shaders()

    vis.content.n_dof[] = totalndof(varshape(mcmc_target(mcmc_states[1])))

    # Fixed axis/bin domain estimated from the prior (the original measure when available); hard support bounds are only derivable from `target`.
    n_dof = vis.content.n_dof[]
    domain_lo, domain_hi = _estimate_prior_domain(
        domain_rng, isnothing(target) ? mcmc_target(mcmc_states[1]) : target, n_dof)
    support_lo, support_hi = isnothing(target) ?
        (Float64[], Float64[]) : _support_bounds(target, n_dof)
    update!(vis.content.graph,
        domain_lo=domain_lo, domain_hi=domain_hi,
        support_lo=support_lo, support_hi=support_hi)

    initial_timing = [_initial_step_spans(vis, state, outputs[i]) for (i, state) in enumerate(mcmc_states)]
    vis.content.step_mode[] = all(!isnothing, initial_timing)
    for (i, state) in enumerate(mcmc_states)
        register_state_for_vis!(vis, state, _transform_walker_outputs(f_pretransform, outputs[i]),
            vis.content.step_mode[] ? initial_timing[i] : _empty_walker_spans(outputs[i]))
    end

    (; graph, buffer_lock, buffer_cond, output_buffer, pending_samples, n_buffer_steps, buffer_dirty, effective_batch_size, is_live, listener_task) = vis.content
    (; poll_interval, adaptive_batching, batch_growth_rate) = vis.backend

    picker_info = PickerInfo(
        vis.content.n_dof[],
        vis.backend.N_max,
        vis.backend.vsel,
        new_vsel -> _apply_vsel!(vis, new_vsel),
    )

    # Checked via the real sample data (same predicate as the static path); .info is unaffected by _transform_walker_outputs.
    has_chain_info = _samples_have_chain_ids(outputs[1][1])
    has_trace_info = vis.content.step_mode[]

    add_index_slider! = Ref{Any}(nothing)

    with_theme(vis.backend.dark ? bat_theme_dark() : bat_theme()) do
        gridlayout = _init_gridlayout(graph, vis.backend.N_max)
        built = _build_fig(graph, gridlayout, picker_info; has_chain_info=has_chain_info, has_trace_info=has_trace_info)
        _maybe_display(built.fig)
        add_index_slider![] = built.add_index_slider!
    end

    # The displayed history includes provisional tails; only emitted samples may
    # become permanent. Pending tails are replaced on every refresh, not appended.
    committed_samples = graph[:samples][]
    committed_timing = vis.content.step_mode[] ? graph[:sample_timing][] : [_empty_walker_spans(c) for c in committed_samples]

    # A terminal flush changes data but contributes no steps. The dirty flag
    # ensures that the forced final drain also handles those zero-step packets.
    function flush_buffer!(; force::Bool = false)
        # lock(...) do for exception safety: a bare lock/unlock left buffer_lock held forever on throw. Returns the drained contents, or nothing.
        extracted_buffers = lock(buffer_lock) do
            update_graph = force ? buffer_dirty[] : n_buffer_steps[] >= effective_batch_size[]
            update_graph || return nothing
            # Compute growth before detaching data; a threshold calculation must
            # not interrupt a drain between resetting the buffer and notification.
            next_batch_size = !force && adaptive_batching ?
                _scaled_buffer_threshold(effective_batch_size[], batch_growth_rate) : effective_batch_size[]
            # Shallow copy suffices: the slots below are replaced, not mutated.
            extracted = (output = copy(output_buffer), pending = copy(pending_samples),
                output_time = copy(vis.content.output_timing), pending_time = copy(vis.content.pending_timing))
            output_buffer .= _empty_chain_outputs.(mcmc_states)
            vis.content.output_timing .= _empty_walker_spans.(output_buffer)
            n_buffer_steps[] = 0
            buffer_dirty[] = false
            # Geometric growth: O(log N) full-dataset recomputes over a run instead of O(N). Skipped on the forced final drain.
            effective_batch_size[] = next_batch_size
            # notify while holding buffer_lock (Threads.Condition contract).
            notify(buffer_cond, all=true)
            return extracted
        end

        isnothing(extracted_buffers) && return nothing

        fresh_batch_trafo = _transform_chain_outputs(f_pretransform, extracted_buffers.output)
        pending_trafo = _transform_chain_outputs(f_pretransform, extracted_buffers.pending)

        if any(chain -> any(walker -> !isempty(walker), chain), fresh_batch_trafo)
            committed_samples, committed_timing = _append_timed_chains(mcmc_states[1], committed_samples, fresh_batch_trafo,
                committed_timing, extracted_buffers.output_time)
        end
        samples_new, timing_new = any(chain -> any(walker -> !isempty(walker), chain), pending_trafo) ?
            _append_timed_chains(mcmc_states[1], committed_samples, pending_trafo, committed_timing, extracted_buffers.pending_time) :
            (committed_samples, committed_timing)

        # Derived from the merged length: checked_push! can collapse a batch-boundary sample into a weight increment instead of a new row.
        current_idxs_new = [length.(chain_samples) for chain_samples in samples_new]

        # Full-domain recompute per flush, not incremental widening: incremental
        # state is corruptible by one non-finite value (eps(Inf) is NaN).
        all_flat_views = Any[]
        for chain_all in samples_new
            for walker_all in chain_all
                push!(all_flat_views, walker_all)
            end
        end
        all_flat_samples = vcat(all_flat_views...)
        new_lo, new_hi = isempty(all_flat_samples) ?
            (copy(graph[:domain_lo][]), copy(graph[:domain_hi][])) :
            _domain_including(all_flat_samples, domain_lo, domain_hi)

        # ONE batched update! per flush: every update! eagerly resolves the observable-attached :gridlayout before returning.
        update!(graph;
            samples=samples_new,
            sample_timing=vis.content.step_mode[] ? timing_new : nothing,
            current_idxs=current_idxs_new,
            domain_lo=new_lo,
            domain_hi=new_hi,
        )
        return nothing
    end

    # One-time initial vsel activation; latches only on success so a throwing apply is retried next tick.
    vsel_initialized = Ref(false)

    # Started only after the figure has resolved its initial state -- the first tick's _apply_vsel! would otherwise race the in-progress construction.
    listener_task[] = errormonitor(
        @async begin
            while is_live[]
                sleep(poll_interval)

                # Catch per tick: an escaping exception would freeze the display and deadlock samplers.
                try
                    # Applied ONCE, not per tick: the picker writes graph[:idxs] directly, so a per-tick re-apply would revert live changes.
                    if !vsel_initialized[]
                        _apply_vsel!(vis, vis.backend.vsel)
                        vsel_initialized[] = true
                    end

                    flush_buffer!()
                catch e
                    @error "Error updating live Makie visualization; skipping this tick" exception = (e, catch_backtrace())
                end
            end

            # is_live[] can flip false between ticks; one forced flush drains whatever is still buffered.
            try
                flush_buffer!(force=true)

                # Post-run step-slider retrofit (needs max_step > 0, never true at build time); with_theme so widgets match this run's dark setting.
                if graph[:max_step][] > 0
                    with_theme(vis.backend.dark ? bat_theme_dark() : bat_theme()) do
                        add_index_slider![]()
                    end
                end
            catch e
                @error "Error on final live Makie visualization flush" exception = (e, catch_backtrace())
            end
        end
    )
end

function BAT.update_visualizer_impl!(
    vis::BATVisualizer{BATMakieVisualization};
    chain_state::MCMCChainState,
    nonzero_weights::Bool,
    is_sampling_step::Bool
)
    (; buffer_lock, buffer_cond, output_buffer, pending_samples, chain_ids, n_buffer_steps, buffer_dirty, effective_batch_size, buffer_ratio) = vis.content
    output_id = findfirst(==(chain_state.info.id), chain_ids)
    # A clear error instead of the opaque "invalid index: nothing" below.
    isnothing(output_id) && throw(ArgumentError(
        "MCMC chain id $(chain_state.info.id) was never registered with the visualizer " *
        "(register_state_for_vis!) -- known chain ids: $(chain_ids)"))

    # Copy before modifying the buffer. Terminal output already contains the old
    # current sample; the sampler has cleared its weight, so remove the old tail.
    pending = is_sampling_step ? _pending_walker_outputs(chain_state, nonzero_weights) : _empty_chain_outputs(chain_state)

    # get_samples! mutates output_buffer[output_id] while flush_buffer! copies it on another task, so it must run under buffer_lock (try/finally so a throw can't leave the lock held forever).
    lock(buffer_lock)
    try
        if vis.content.step_mode[]
            pending_time = _empty_walker_spans(pending)
            t = chain_state.stepno
            cycle = chain_state.info.cycle
            for j in eachindex(pending)
                sample = chain_state.output[j]
                if !nonzero_weights || sample.weight > 0
                    BAT.checked_push!(output_buffer[output_id][j], sample)
                    if sample.weight > 0
                        endpoint = is_sampling_step ? t - 1 : t
                        _push_step_span!(vis.content.output_timing[output_id][j],
                            SampleStepSpan(length(output_buffer[output_id][j]), cycle, endpoint - sample.weight + 1, endpoint))
                    end
                end
                if !isempty(pending[j]) && pending[j].weight[1] > 0
                    push!(pending_time[j], SampleStepSpan(1, cycle, t - pending[j].weight[1] + 1, t))
                end
            end
            vis.content.pending_timing[output_id] = pending_time
        else
            get_samples!(output_buffer[output_id], chain_state, nonzero_weights)
        end
        pending_samples[output_id] = pending
        n_buffer_steps[] = _add_buffer_steps(n_buffer_steps[], is_sampling_step ? length(pending) : 0)
        buffer_dirty[] = true
        # Backpressure: block sampling once the buffer runs too far ahead; the threshold derives from effective_batch_size[] so it stays >= the flush trigger.
        while n_buffer_steps[] >= _scaled_buffer_threshold(effective_batch_size[], buffer_ratio)
            wait(buffer_cond)
        end
    finally
        unlock(buffer_lock)
    end
end
