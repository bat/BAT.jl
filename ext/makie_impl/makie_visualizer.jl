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
    update!(graph, samples=samples_graph)

    current_idxs = length.(samples)
    current_idxs_graph = graph[:current_idxs][]
    push!(current_idxs_graph, current_idxs)
    update!(graph, current_idxs=current_idxs_graph)

    push!(chain_ids, mcmc_state.chain_state.info.id)
    empty_chain_output = _empty_chain_outputs(mcmc_state)
    push!(output_buffer, empty_chain_output)
end

# Core of the vsel choke point, with no dependency on a live BATVisualizer --
# usable directly by the static bat_makie_plot path, which has no vis/content
# (and no concurrent listener task to race, so no locking needed there).
# Clamps the requested selection against both the model's dimensionality and
# the grid size (N_max), and only touches the graph if the (clamped) selection
# actually differs from what's already there.
function _apply_vsel_to_graph!(graph::ComputeGraph, n_dof::Integer, N_max::Integer, requested_vsel::AbstractVector{<:Integer})
    clamped = _clamp_vsel(requested_vsel, n_dof, N_max)
    if clamped != graph[:idxs][]
        update!(graph, idxs=clamped)
    end
    return clamped
end

# Live-path wrapper: this is what a variable-selection UI widget should call to
# change vsel while the plot is live; it's also used below for the initial
# activation once real samples exist. Guarded by buffer_lock so a future
# widget running on a different task can't race the listener's own tick.
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
    f_pretransform::Function
)
    warmup_makie_shaders()

    vis.content.n_dof[] = totalndof(varshape(mcmc_target(mcmc_states[1])))

    # Fixed axis-limit/histogram-bin-edge domain, estimated from the prior
    # before any real samples exist (see _estimate_prior_domain) -- set here,
    # before the figure is first built, so the very first render already uses
    # it rather than some other placeholder.
    domain_lo, domain_hi = _estimate_prior_domain(mcmc_states, vis.content.n_dof[])
    update!(vis.content.graph, domain_lo=domain_lo, domain_hi=domain_hi)

    for (i, state) in enumerate(mcmc_states)
        register_state_for_vis!(vis, state, _transform_walker_outputs(f_pretransform, outputs[i]))
    end

    (; graph, buffer_lock, buffer_cond, output_buffer, n_buffer_samples, effective_batch_size, is_live, listener_task) = vis.content
    (; poll_interval, adaptive_batching, batch_growth_rate) = vis.backend

    picker_info = (
        N=vis.content.n_dof[],
        N_max=vis.backend.N_max,
        initial_vsel=vis.backend.vsel,
        (apply_vsel!)=new_vsel -> _apply_vsel!(vis, new_vsel),
    )

    # This entry point is only ever reached from MCMC sampling (mcmc_states::
    # Vector{<:MCMCState} above), so it's tempting to assume chain info is
    # always present -- checked via the real sample data anyway (same
    # predicate the static path uses), so this stays correct even if some
    # future MCMC-adjacent sampler reuses this visualizer without actually
    # producing chain-tagged samples. .info is unaffected by
    # _transform_walker_outputs (that only transforms .v), so checking the
    # untransformed first chain/walker's output is equivalent to checking
    # what's actually registered.
    has_chain_info = _samples_have_chain_ids(outputs[1][1])

    with_theme(vis.backend.dark ? bat_theme_dark() : bat_theme()) do
        gridlayout = _init_gridlayout(graph, vis.backend.N_max)
        fig = _build_fig(graph, gridlayout, picker_info; has_chain_info=has_chain_info)
        display(fig)
    end

    # force=false: only flush once effective_batch_size[] samples are buffered
    # (the normal per-tick check). force=true: flush whatever is currently
    # buffered regardless of threshold, used once for the final drain below so
    # the last partial batch isn't silently dropped when is_live[] flips false.
    function flush_buffer!(; force::Bool=false)
        # lock(...) do -- exception-safe by construction (try/finally inside),
        # replacing a bare lock/unlock pair: anything throwing in this
        # critical section previously left buffer_lock held FOREVER. The
        # listener's own per-tick catch swallows the exception but cannot
        # release a lock acquired deeper in the call stack, so every sampling
        # thread (and the listener's own next tick) would then deadlock on
        # lock(buffer_lock) -- the same silent-freeze-then-OOM failure mode
        # already documented and try/finally-hardened in
        # update_visualizer_impl! below, just via this other code path.
        # Returns the drained buffer contents, or nothing if no flush was due.
        extracted_output_buffer = lock(buffer_lock) do
            update_graph = force ? n_buffer_samples[] > 0 : n_buffer_samples[] >= effective_batch_size[]
            update_graph || return nothing
            # Shallow copy: output_buffer's slots get replaced (not mutated)
            # below, so the extracted inner vectors are never touched again --
            # no need to deep-copy the actual sample data out of them.
            extracted = copy(output_buffer)
            output_buffer .= _empty_chain_outputs.(mcmc_states)
            n_buffer_samples[] = 0
            # Geometric growth (not on the forced final drain -- there's no
            # "next tick" for it to matter to): the same amortized-doubling
            # trick as dynamic array growth, so the number of full-dataset
            # recomputes over a run is O(log N) instead of O(N), keeping total
            # redraw work near-linear instead of quadratic in sample count.
            if !force && adaptive_batching
                effective_batch_size[] = ceil(Int, effective_batch_size[] * batch_growth_rate)
            end
            # Wakes any sampling threads blocked on the high-watermark in
            # update_visualizer_impl! -- notify while still holding buffer_lock,
            # matching Threads.Condition's contract.
            notify(buffer_cond, all=true)
            return extracted
        end

        isnothing(extracted_output_buffer) && return nothing

        fresh_batch_trafo = _transform_chain_outputs(f_pretransform, extracted_output_buffer)

        samples = graph[:samples][]
        samples_new = _append_chain_outputs(mcmc_states[1], samples, fresh_batch_trafo)

        update!(graph, samples=samples_new)

        # Derived from the actual merged length (not the raw batch length):
        # checked_push! inside _append_chain_outputs can collapse a sample at
        # the batch boundary into a weight increment instead of a new row.
        current_idxs_new = [length.(chain_samples) for chain_samples in samples_new]
        update!(graph, current_idxs=current_idxs_new)

        # Deliberately flattens samples_new (the full accumulated dataset)
        # rather than just fresh_batch_trafo (this batch alone) -- see
        # _recompute_domain!'s docs for why a full recompute every flush
        # was chosen over a cheaper incremental version.
        all_flat_views = Any[]
        for chain_all in samples_new
            for walker_all in chain_all
                push!(all_flat_views, walker_all)
            end
        end
        _recompute_domain!(graph, vcat(all_flat_views...), domain_lo, domain_hi)
        return nothing
    end

    # One-time initial vsel activation flag (see the listener body below):
    # graph[:idxs] starts empty, so the grid only becomes active once the
    # configured vsel is applied for the first time. Latches only on success
    # -- if the apply throws (caught by the tick's try/catch), the next tick
    # retries, preserving the pre-existing retry-on-error behavior exactly.
    vsel_initialized = Ref(false)

    # Started only after the figure above has fully resolved its initial state --
    # the listener's first tick can mutate :idxs/:samples via _apply_vsel!, which
    # would otherwise race the still-in-progress initial construction (observed
    # in practice specifically on a cold/slow first JIT compile of the compute
    # graph's closures, where construction can take longer than poll_interval).
    listener_task[] = errormonitor(
        @async begin
            while is_live[]
                sleep(poll_interval)

                # Any exception escaping this block previously killed the
                # whole listener task silently -- errormonitor only logs it
                # to stderr, it doesn't restart the loop or surface anything
                # to the user. Since nothing else ever calls flush_buffer! or
                # notifies buffer_cond, that failure mode wasn't "one bad
                # tick, skip it" -- it was "the display silently freezes,
                # sampling tasks blocked on backpressure hang forever, and
                # output_buffer grows without bound," often manifesting much
                # later as an apparently unrelated OutOfMemoryError
                # elsewhere. Catching here bounds any future bug in this loop
                # to a single skipped tick instead.
                try
                    # Applied ONCE (not per tick): this is purely the initial
                    # activation of the configured vsel -- the vsel picker's
                    # own checkbox callbacks (picker_info.apply_vsel! above)
                    # are the sole runtime writer of graph[:idxs] after that.
                    # This used to run unconditionally every tick, on the
                    # assumption that a UI widget would communicate changes by
                    # mutating vis.backend.vsel -- but the picker widget that
                    # actually got built writes the graph directly instead, so
                    # the per-tick re-apply of the (immutable, never-updated)
                    # configured vsel silently REVERTED every live picker
                    # change within one poll interval (confirmed via direct
                    # repro), and re-fired _clamp_vsel's @warn ~10x/s for
                    # models with fewer free parameters than the default vsel.
                    if !vsel_initialized[]
                        _apply_vsel!(vis, vis.backend.vsel)
                        vsel_initialized[] = true
                    end

                    flush_buffer!()
                catch e
                    @error "Error updating live Makie visualization; skipping this tick" exception = (e, catch_backtrace())
                end
            end

            # is_live[] can flip false between two ticks (bat_sample_impl sets it
            # right before waiting on this task), which would otherwise drop
            # whatever's currently buffered -- below n_batch, or even a full
            # batch this tick hadn't gotten to yet -- leaving the display behind
            # the true final sample count. One last unconditional flush closes
            # that gap before the task returns.
            try
                flush_buffer!(force=true)
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
    output_id = findfirst(x -> x == chain_state.info.id, chain_ids)

    # get_samples! mutates output_buffer[output_id] (a DensitySampleVector/
    # StructArray) field-by-field, non-atomically (StructArrays.push! pushes
    # onto .v, then .logd, then .weight, then .info, then .aux in sequence --
    # see BAT.checked_push!). Every MCMC chain runs this on its own thread
    # concurrently (Threads.@spawn in mcmc_iterate!!), while flush_buffer!
    # (on a separate listener task) reads/copies output_buffer under
    # buffer_lock. This used to acquire buffer_lock only *after* get_samples!
    # had already run -- so a chain's mutation here could race a concurrent
    # flush_buffer! copy and be caught mid-push, with some fields already
    # updated and others not yet. Confirmed to reproduce a real
    # "ArgumentError: all component arrays must have the same shape" deep in
    # flush_buffer!'s transform step -- thrown inside an errormonitor-wrapped
    # background task, so it silently kills the listener loop rather than
    # surfacing to the user. From that point on nothing ever drains
    # output_buffer or re-notifies buffer_cond again, so sampling threads
    # blocked on the backpressure wait below hang forever and the buffer
    # grows without bound -- a very plausible root cause for a much later,
    # seemingly unrelated OutOfMemoryError. Far more likely to manifest with
    # a vector-valued parameter: .v's push there resizes an
    # ElasticArrays-backed ArrayOfSimilarArrays (a measurably slower
    # operation than a plain Vector's amortized push!), widening the race
    # window between .v's own push and the other four fields'.
    #
    # Wrapping get_samples! itself in the same lock flush_buffer! already
    # uses closes this race at its source, matching the lock discipline the
    # rest of this function already follows below. try/finally (not a bare
    # lock/unlock pair) specifically because get_samples! now runs inside
    # this critical section -- unlike the backpressure wait below (which
    # isn't expected to throw), get_samples! is real, more complex sampler
    # code; if it ever did throw here without a finally, buffer_lock would
    # stay held forever, turning any future bug in it into a permanent
    # deadlock instead of a recoverable error.
    lock(buffer_lock)
    try
        n_smpls_start = sum(length.(output_buffer[output_id]))
        get_samples!(output_buffer[output_id], chain_state, nonzero_weights)
        n_smpls_end = sum(length.(output_buffer[output_id]))
        n_new = n_smpls_end - n_smpls_start
        n_buffer_samples[] += n_new
        # Backpressure: block this chain's sampling task once the buffer has
        # grown far enough ahead of the listener that the display would
        # otherwise lag noticeably behind the true latest samples. Bounded,
        # not per-sample -- sampling still runs in free bursts up to
        # max_buffered before it has to wait -- so the listener remains the
        # sole bottleneck only when it's genuinely falling behind, not on
        # every single step.
        #
        # Derived from effective_batch_size[] (scaled by the fixed configured
        # ratio) rather than a separately-tracked/grown value: this threshold must
        # always stay >= the current flush trigger, or sampling would permanently
        # block below a threshold the listener is never allowed to reach --
        # deriving it from the same value the flush trigger uses makes that
        # invariant automatic instead of something two independently-growing
        # counters could drift out of.
        # Recomputed every iteration (not hoisted out of the loop) so a waiter
        # woken by one flush re-checks against whatever effective_batch_size[]
        # grew to by the time it wakes, not a value that was already stale then.
        while n_buffer_samples[] >= ceil(Int, effective_batch_size[] * buffer_ratio)
            wait(buffer_cond)
        end
    finally
        unlock(buffer_lock)
    end
end
