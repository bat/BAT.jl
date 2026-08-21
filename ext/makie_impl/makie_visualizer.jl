# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function register_state_for_vis!(
    vis::BATVisualizer{BATMakieVisualization},
    mcmc_state::MCMCState,
    samples#::Vector{DensitySampleVector}
)
    (; graph, chain_ids, output_buffer) = vis.content

    empty_walker_outputs = _empty_chain_outputs(mcmc_state)
    samples_new = _append_walker_outputs(mcmc_state, empty_walker_outputs, samples)

    samples_graph = graph[:samples][]
    push!(samples_graph, samples_new)

    current_idxs = length.(samples)
    current_idxs_graph = graph[:current_idxs][]
    push!(current_idxs_graph, current_idxs)

    # One batched update! -- see flush_buffer!'s matching comment (each update! call is a full invalidate/notify cycle of its own).
    update!(graph; samples=samples_graph, current_idxs=current_idxs_graph)

    push!(chain_ids, mcmc_state.chain_state.info.id)
    empty_chain_output = _empty_chain_outputs(mcmc_state)
    push!(output_buffer, empty_chain_output)
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

    warmup_makie_shaders()

    vis.content.n_dof[] = totalndof(varshape(mcmc_target(mcmc_states[1])))

    # Fixed axis/bin domain estimated from the prior (the original measure when available); hard support bounds are only derivable from `target`.
    n_dof = vis.content.n_dof[]
    domain_lo, domain_hi = _estimate_prior_domain(
        isnothing(target) ? mcmc_target(mcmc_states[1]) : target, n_dof)
    support_lo, support_hi = isnothing(target) ?
        (Float64[], Float64[]) : _support_bounds(target, n_dof)
    update!(vis.content.graph,
        domain_lo=domain_lo, domain_hi=domain_hi,
        support_lo=support_lo, support_hi=support_hi)

    for (i, state) in enumerate(mcmc_states)
        register_state_for_vis!(vis, state, _transform_walker_outputs(f_pretransform, outputs[i]))
    end

    (; graph, buffer_lock, buffer_cond, output_buffer, n_buffer_samples, effective_batch_size, is_live, listener_task) = vis.content
    (; poll_interval, adaptive_batching, batch_growth_rate) = vis.backend

    picker_info = PickerInfo(
        vis.content.n_dof[],
        vis.backend.N_max,
        vis.backend.vsel,
        new_vsel -> _apply_vsel!(vis, new_vsel),
    )

    # Checked via the real sample data (same predicate as the static path); .info is unaffected by _transform_walker_outputs.
    has_chain_info = _samples_have_chain_ids(outputs[1][1])
    has_trace_info = _samples_have_trace_info(outputs[1][1])

    add_index_slider! = Ref{Any}(nothing)

    with_theme(vis.backend.dark ? bat_theme_dark() : bat_theme()) do
        gridlayout = _init_gridlayout(graph, vis.backend.N_max)
        built = _build_fig(graph, gridlayout, picker_info; has_chain_info=has_chain_info, has_trace_info=has_trace_info)
        _maybe_display(built.fig)
        add_index_slider![] = built.add_index_slider!
    end

    # force=false flushes only once effective_batch_size[] samples are buffered; force=true flushes whatever is buffered (the final drain).
    function flush_buffer!(; force::Bool = false)
        # lock(...) do for exception safety: a bare lock/unlock left buffer_lock held forever on throw. Returns the drained contents, or nothing.
        extracted_output_buffer = lock(buffer_lock) do
            update_graph = force ? n_buffer_samples[] > 0 : n_buffer_samples[] >= effective_batch_size[]
            update_graph || return nothing
            # Shallow copy suffices: the slots below are replaced, not mutated.
            extracted = copy(output_buffer)
            output_buffer .= _empty_chain_outputs.(mcmc_states)
            n_buffer_samples[] = 0
            # Geometric growth: O(log N) full-dataset recomputes over a run instead of O(N). Skipped on the forced final drain.
            if !force && adaptive_batching
                effective_batch_size[] = ceil(Int, effective_batch_size[] * batch_growth_rate)
            end
            # notify while holding buffer_lock (Threads.Condition contract).
            notify(buffer_cond, all=true)
            return extracted
        end

        isnothing(extracted_output_buffer) && return nothing

        fresh_batch_trafo = _transform_chain_outputs(f_pretransform, extracted_output_buffer)

        samples = graph[:samples][]
        samples_new = _append_chain_outputs(mcmc_states[1], samples, fresh_batch_trafo)

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
    nonzero_weights::Bool
)
    (; buffer_lock, buffer_cond, output_buffer, chain_ids, n_buffer_samples, effective_batch_size, buffer_ratio) = vis.content
    output_id = findfirst(==(chain_state.info.id), chain_ids)
    # A clear error instead of the opaque "invalid index: nothing" below.
    isnothing(output_id) && throw(ArgumentError(
        "MCMC chain id $(chain_state.info.id) was never registered with the visualizer " *
        "(register_state_for_vis!) -- known chain ids: $(chain_ids)"))

    # get_samples! mutates output_buffer[output_id] while flush_buffer! copies it on another task, so it must run under buffer_lock (try/finally so a throw can't leave the lock held forever).
    lock(buffer_lock)
    try
        # sum(length, ...): no intermediate array on this per-step hot path.
        n_smpls_start = sum(length, output_buffer[output_id])
        get_samples!(output_buffer[output_id], chain_state, nonzero_weights)
        n_smpls_end = sum(length, output_buffer[output_id])
        n_new = n_smpls_end - n_smpls_start
        n_buffer_samples[] += n_new
        # Backpressure: block sampling once the buffer runs too far ahead; the threshold derives from effective_batch_size[] so it stays >= the flush trigger.
        while n_buffer_samples[] >= ceil(Int, effective_batch_size[] * buffer_ratio)
            wait(buffer_cond)
        end
    finally
        unlock(buffer_lock)
    end
end
