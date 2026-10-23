# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# The original measure owns the display shape; an initialization source may
# instead be an empirical approximation with different shape metadata.
_display_target_shape(::Any) = missing
_display_target_shape(target::Union{BAT.BATMeasure,Distribution}) = varshape(target)
function _live_display_shape(target, f_pretransform, state)
    shape = _display_target_shape(target)
    if ismissing(shape)
        backward = inverse(f_pretransform)
        shape = backward isa AbstractValueShape ? backward :
            ValueShapes.resultshape(backward, varshape(mcmc_target(state)))
    end
    return shape
end

# Shape-only transforms have zero Jacobian and already support BAT's sample
# broadcast. Generic transform inference can erase their constant fields.
function _inverse_display_samples(f_pretransform, samples, executor)
    isempty(samples) && throw(ArgumentError("A display prototype requires a nonempty sample container"))
    backward = inverse(f_pretransform)
    return backward isa AbstractValueShape ? backward.(samples) :
        transform_samples(backward, samples, executor)
end

_display_has_shape(values) = values isa Union{ArrayOfSimilarArrays,ShapedAsNTArray,AbstractVector{<:Real}}
# Value-derived shapes describe coordinates, not storage precision. In
# particular, Float32 observations can share a domain with Float64 prior draws.
_display_inferred_shape(shape::AbstractValueShape) = shape
_display_inferred_shape(shape::ArrayShape{<:Real}) = ArrayShape{Real}(size(shape))
_display_inferred_shape(::ScalarShape{<:Real}) = ScalarShape{Real}()
_display_inferred_shape(shape::NamedTupleShape{names,A,V}) where {names,A,V} =
    NamedTupleShape(V, map(_display_inferred_shape, (; shape...)))
_display_value_shape(samples) = _display_inferred_shape(
    _display_has_shape(samples.v) ? varshape(samples) : valshape(first(samples.v)))

function _canonical_display_samples(samples, shape::AbstractValueShape)
    # Avoid querying elshape on raw nested NamedTuple vectors. Bulk unshaping
    # also rejects an erased constant shape; row-wise unshaping repairs it.
    result = if _display_has_shape(samples.v) && varshape(samples) <= shape
        unshaped.(samples)
    else
        DensitySampleVector(v=map(v -> unshaped(v, shape), samples.v),
            logd=samples.logd, weight=samples.weight, info=samples.info, aux=samples.aux)
    end
    totalndof(varshape(result)) == totalndof(shape) ||
        throw(ArgumentError("Transformed samples do not match the original display shape"))
    return result
end

# Unlike a slice/similar of transformed storage, this constructor guarantees
# appendable coordinates and mutable columns (including missing-Jacobian logd).
_empty_display_samples(template) = DensitySampleVector(eltype(template), totalndof(varshape(template)))

function _initial_display_outputs(states, outputs, f_pretransform, shape)
    axes(states) == axes(outputs) || throw(ArgumentError("Mismatched display chain topology"))
    result = [Vector{DensitySampleVector}(undef, length(chain)) for chain in outputs]
    # Finish all retained rows in their original order before any additional
    # prototype evaluation, preserving caller effects and startup failures.
    for i in eachindex(states)
        length(outputs[i]) == BAT.nwalkers(states[i]) ||
            throw(ArgumentError("Mismatched display walker topology"))
        for j in eachindex(outputs[i])
            isempty(outputs[i][j]) && continue
            converted = _inverse_display_samples(f_pretransform, outputs[i][j], BAT.SequentialExec())
            ismissing(shape) && (shape = _display_value_shape(converted))
            result[i][j] = _canonical_display_samples(converted, shape)
        end
    end
    for i in eachindex(states)
        populated = findfirst(!isempty, outputs[i])
        template = if isnothing(populated)
            # Numerical initialization has joined. Current x is populated even
            # when retained output/current weights are zero. No prototype row
            # enters the history, and nonfinite logd does not preclude a shape.
            owned = deepcopy(BAT.current_sample(states[i].chain_state))
            converted = _inverse_display_samples(f_pretransform, owned, BAT.SequentialExec())
            ismissing(shape) && (shape = _display_value_shape(converted))
            _canonical_display_samples(converted, shape)
        else
            result[i][populated]
        end
        for j in eachindex(outputs[i])
            isempty(outputs[i][j]) && (result[i][j] = _empty_display_samples(template))
        end
    end
    return result, shape
end

function _transform_display_outputs(f_pretransform, outputs, templates, shape;
    executor=BAT.MultiThreadedExec())
    axes(outputs) == axes(templates) || throw(ArgumentError("Mismatched display chain topology"))
    return map(outputs, templates) do chain, old
        axes(chain) == axes(old) || throw(ArgumentError("Mismatched display walker topology"))
        map(chain, old) do samples, template
            if isempty(samples)
                _empty_display_samples(template)
            else
                converted = _inverse_display_samples(f_pretransform, samples, executor)
                _canonical_display_samples(converted,
                    ismissing(shape) ? _display_value_shape(converted) : shape)
            end
        end
    end
end

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
    (; chain_ids, output_buffer, pending_samples) = vis.content
    graph = _live_graph(vis)

    empty_walker_outputs = _empty_display_samples.(samples)
    samples_new, timing_new = _append_timed_walkers(empty_walker_outputs, samples, _empty_walker_spans(samples), timing)

    samples_graph = graph[:samples][]
    push!(samples_graph, samples_new)

    current_idxs = length.(samples_new)
    current_idxs_graph = graph[:current_idxs][]
    push!(current_idxs_graph, current_idxs)

    # Keep one batched update: each call starts a full invalidation/notification cycle.
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

# Live changes use the view owner; the transport lock never covers rendering.
function _apply_vsel!(vis::BATVisualizer{BATMakieVisualization}, requested_vsel::AbstractVector{<:Integer})
    clamped = _clamp_vsel(requested_vsel, vis.content.n_dof[], vis.backend.N_max)
    # Do not discard an intent equal to the current graph value: another intent
    # may already be queued by a reentrant callback.
    _live_graph_update!(vis; idxs=clamped)
    return nothing
end

function BAT.init_visualizer!(
    vis::BATVisualizer{BATMakieVisualization}; rng::AbstractRNG,
    mcmc_states::Vector{<:MCMCState}, outputs, f_pretransform::Function, target=nothing
)
    isnothing(vis.content) && throw(ArgumentError("Create a fresh BATVisualizer for sampling"))
    ctl = vis.content.lifecycle
    claimed = Ref(false)
    admitted = false
    try
        start_view = lock(vis.content.buffer_lock) do
            # Direct callers can acquire here; sampling normally acquires before
            # numerical initialization. Both paths use the same one-shot permit.
            ctl.phase === :fresh && _claim_live_visualizer!(vis, claimed)
            permit = ctl.startup_owner
            (!isnothing(permit) && permit.value === current_task() &&
                ctl.owner_thread == Threads.threadid() && !ctl.initializing) ||
                throw(ArgumentError("Create a fresh BATVisualizer for sampling"))
            disabled = ctl.outcome === :disabled &&
                ctl.reason in (:closed, :disposed, :disabled) && isnothing(ctl.error)
            (ctl.phase === :starting || disabled) ||
                throw(ArgumentError("Create a fresh BATVisualizer for sampling"))
            # Cleanup is already installed. Never leave a consumed permit and
            # an unprotected initializer if SIGINT arrives at this boundary.
            Base.disable_sigint() do
                admitted = true
                ctl.startup_owner = nothing
                ctl.initializing = !disabled
            end
            return !disabled
        end
        # Disposal can already have released the graph and strong owner. This
        # acquisition continues sampling without recreating an observer.
        start_view || return nothing
        backend = Makie.current_backend()
        if backend isa Module && nameof(backend) === :GLMakie && Threads.threadid() != 1
            throw(ArgumentError("Initialize a live GLMakie visualization on Julia's main thread"))
        end
        _init_live_visualizer!(vis, rng, mcmc_states, outputs, f_pretransform, target)
    catch err
        if claimed[] || admitted
            # Hand off before rollback joins a coordinator waiting for startup.
            _finish_live_initialization!(vis)
            _fail_visualizer!(vis, err, catch_backtrace(), :startup)
            BAT._safe_clear_visualizer_init!(vis; suppress_interrupt=true)
            BAT._safe_finish_visualizer!(vis; success=false)
        end
        rethrow()
    finally
        admitted && _finish_live_initialization!(vis)
    end
    return nothing
end

function _init_live_visualizer!(vis, rng, mcmc_states, outputs, f_pretransform, target)
    ctl = vis.content.lifecycle
    graph = _live_graph(vis)
    domain_rng = copy(rng)
    domain_lo, domain_hi = Float64[], Float64[]
    display_shape = missing
    built = nothing
    started = _with_live_view(vis) do
        warmup_makie_shaders()
        isempty(mcmc_states) && throw(ArgumentError("Live visualization requires at least one chain"))
        display_shape = _live_display_shape(target, f_pretransform, first(mcmc_states))
        domain_target = isnothing(target) ? mcmc_target(first(mcmc_states)) : target
        domain_transform = isnothing(target) ? inverse(f_pretransform) : identity
        domain_ready = !ismissing(display_shape)
        if domain_ready
            domain_lo, domain_hi = _estimate_prior_domain(domain_rng, domain_target,
                totalndof(display_shape); display_shape, f_display=domain_transform)
        end
        display_outputs, display_shape = _initial_display_outputs(
            mcmc_states, outputs, f_pretransform, display_shape)
        n_dof = totalndof(display_shape)
        vis.content.n_dof[] = n_dof
        if !domain_ready
            domain_lo, domain_hi = _estimate_prior_domain(domain_rng, domain_target,
                n_dof; display_shape, f_display=domain_transform)
        end
        support_lo, support_hi = isnothing(target) ?
            (Float64[], Float64[]) : _support_bounds(target, n_dof)
        update!(graph; domain_lo=domain_lo, domain_hi=domain_hi,
            support_lo=support_lo, support_hi=support_hi)
        initial_timing = [_initial_step_spans(vis, state, outputs[i]) for (i, state) in enumerate(mcmc_states)]
        vis.content.step_mode[] = all(!isnothing, initial_timing)
        for (i, state) in enumerate(mcmc_states)
            register_state_for_vis!(vis, state, display_outputs[i],
                vis.content.step_mode[] ? initial_timing[i] : _empty_walker_spans(outputs[i]))
        end
        picker_info = PickerInfo(n_dof, vis.backend.N_max, vis.backend.vsel,
            new_vsel -> _apply_vsel!(vis, new_vsel))
        with_theme(vis.backend.dark ? bat_theme_dark() : bat_theme()) do
            gridlayout = _init_gridlayout(graph, vis.backend.N_max)
            built = _build_fig(graph, gridlayout, picker_info;
                has_chain_info=_samples_have_chain_ids(outputs[1][1]),
                has_trace_info=vis.content.step_mode[], live_vis=vis)
        end
    end
    started || return nothing
    # Native renderer startup can yield to callbacks which acquire the view
    # guard. Do not join it while holding that guard. The initialization barrier
    # still owns the figure/graph and postpones disposal until this call settles.
    # The construction theme has also been restored: renderer callbacks must
    # not be joined while this task holds Makie's global theme lock.
    _maybe_display(built.fig;
        on_failure=(err, bt) -> _fail_visualizer!(vis, err, bt, :startup))
    history = Ref((samples=graph[:samples][], timing=vis.content.step_mode[] ?
        graph[:sample_timing][] : [_empty_walker_spans(c) for c in graph[:samples][]]))
    tick! = () -> _flush_live_buffer!(vis, mcmc_states, f_pretransform, history, domain_lo, domain_hi; display_shape)
    finish! = function()
        _flush_live_buffer!(vis, mcmc_states, f_pretransform, history, domain_lo, domain_hi; force=true, display_shape)
        lock(vis.content.buffer_lock) do
            _live_can_publish_locked(vis) && (ctl.data_complete = true)
        end
        _with_live_view(vis; publication=true) do
            if graph[:max_step][] > 0
                with_theme(vis.backend.dark ? bat_theme_dark() : bat_theme()) do
                    built.add_index_slider!()
                end
            end
        end
    end
    _start_live_listener!(vis, tick!, finish!)
    return nothing
end

function _detach_live_batch!(vis, mcmc_states; force::Bool=false)
    ready = lock(vis.content.buffer_lock) do
        _live_can_publish_locked(vis) && (force ? vis.content.buffer_dirty[] :
            vis.content.n_buffer_steps[] >= vis.content.effective_batch_size[])
    end
    ready || return nothing
    # Allocate every replacement before changing any shared slot. A failed
    # preparation leaves the original batch owned by the transport.
    replacements = _empty_chain_outputs.(mcmc_states)
    replacement_timing = _empty_walker_spans.(replacements)
    lock(vis.content.buffer_lock) do
        _live_can_publish_locked(vis) || return nothing
        c = vis.content
        should_drain = force ? c.buffer_dirty[] : c.n_buffer_steps[] >= c.effective_batch_size[]
        should_drain || return nothing
        try
            next_batch = !force && vis.backend.adaptive_batching ?
                _scaled_buffer_threshold(c.effective_batch_size[], vis.backend.batch_growth_rate) : c.effective_batch_size[]
            batch = (output=copy(c.output_buffer), pending=copy(c.pending_samples),
                output_time=copy(c.output_timing), pending_time=copy(c.pending_timing))
            c.output_buffer .= replacements
            c.output_timing .= replacement_timing
            c.n_buffer_steps[] = 0
            c.buffer_dirty[] = false
            c.effective_batch_size[] = next_batch
            notify(c.buffer_cond; all=true)
            return batch
        catch err
            _stop_live_locked!(vis, :failed, CapturedException(err, catch_backtrace()); phase=:detach)
            rethrow()
        end
    end
end

function _publish_live_frame!(vis, payload::NamedTuple)
    published = _live_graph_update!(vis; publication=true, payload...)
    if published
        lock(vis.content.buffer_lock) do
            _live_can_publish_locked(vis) && (vis.content.lifecycle.last_publication += 1)
        end
    end
    return published
end

function _flush_live_buffer!(vis, mcmc_states, f_pretransform, history, domain_lo, domain_hi;
    force::Bool=false, display_shape=missing)
    batch = _detach_live_batch!(vis, mcmc_states; force=force)
    isnothing(batch) && return nothing
    # The listener retains batch until it either commits or records failure.
    # There is deliberately no replay after a partial/uncertain publication.
    committed = history[]
    fresh = _transform_display_outputs(f_pretransform, batch.output, committed.samples, display_shape)
    pending = _transform_display_outputs(f_pretransform, batch.pending, committed.samples, display_shape)
    if any(chain -> any(!isempty, chain), fresh)
        samples, timing = _append_timed_chains(committed.samples, fresh,
            committed.timing, batch.output_time)
        candidate = (samples=samples, timing=timing)
        accepted = lock(vis.content.buffer_lock) do
            _live_can_publish_locked(vis) || return false
            history[] = candidate
            true
        end
        accepted || return nothing
        committed = candidate
    end
    samples, timing = any(chain -> any(!isempty, chain), pending) ?
        _append_timed_chains(committed.samples, pending, committed.timing, batch.pending_time) :
        (committed.samples, committed.timing)
    current_idxs = [length.(chain) for chain in samples]
    lo, hi = _domain_including(Iterators.flatten(samples), domain_lo, domain_hi)
    _publish_live_frame!(vis, (samples=samples,
        sample_timing=vis.content.step_mode[] ? timing : nothing,
        current_idxs=current_idxs, domain_lo=lo, domain_hi=hi))
    return nothing
end

function BAT.update_visualizer_impl!(vis::BATVisualizer{BATMakieVisualization};
    chain_state::MCMCChainState, nonzero_weights::Bool, is_sampling_step::Bool)
    isnothing(vis.content) && return nothing
    # Content stays type-erased in BAT's context; recover this private control
    # type so the disabled-path atomic read can be compiled directly.
    ctl = vis.content.lifecycle::_MakieLiveControl
    ctl.accepting[] || return nothing
    try
        _ingest_live_observation!(vis, chain_state, nonzero_weights, is_sampling_step)
    catch err
        if err isa InterruptException
            BAT.request_visualizer_stop!(vis; reason=:aborted, exception=CapturedException(err, catch_backtrace()))
            rethrow()
        end
        _fail_visualizer!(vis, err, catch_backtrace(), :ingest)
    end
    return nothing
end

function _ingest_live_observation!(vis, chain_state, nonzero_weights, is_sampling_step)
    (; buffer_lock, buffer_cond, output_buffer, pending_samples, chain_ids, n_buffer_steps, buffer_dirty, effective_batch_size, buffer_ratio) = vis.content
    output_id = findfirst(==(chain_state.info.id), chain_ids)
    # A clear error instead of the opaque "invalid index: nothing" below.
    isnothing(output_id) && throw(ArgumentError(
        "MCMC chain id $(chain_state.info.id) was never registered with the visualizer " *
        "(register_state_for_vis!) -- known chain ids: $(chain_ids)"))

    # Copy before modifying the buffer. Terminal output already contains the old
    # current sample; the sampler has cleared its weight, so remove the old tail.
    pending = is_sampling_step ? _pending_walker_outputs(chain_state, nonzero_weights) : _empty_chain_outputs(chain_state)

    # Serialize producer mutation with listener detachment. Always release the
    # transport lock, including when extraction or timing validation throws.
    timeout_error = nothing
    lock(buffer_lock)
    try
        vis.content.lifecycle.accepting[] || return nothing
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
        wait_started = time_ns()
        while vis.content.lifecycle.accepting[] &&
                n_buffer_steps[] >= _scaled_buffer_threshold(effective_batch_size[], buffer_ratio)
            if _live_elapsed(wait_started) >= vis.backend.observer_timeout
                timeout_error = ErrorException("Visualization backpressure exceeded observer_timeout")
                _stop_live_locked!(vis, :backpressure_timeout,
                    CapturedException(timeout_error, Any[]);
                    phase=:backpressure_timeout)
                break
            end
            wait(buffer_cond)
        end
    catch err
        reason = err isa InterruptException ? :aborted : :failed
        _stop_live_locked!(vis, reason, CapturedException(err, catch_backtrace()); phase=:ingest)
        rethrow()
    finally
        unlock(buffer_lock)
    end
    isnothing(timeout_error) || _fail_visualizer!(vis, timeout_error, Any[], :backpressure_timeout)
    return nothing
end
