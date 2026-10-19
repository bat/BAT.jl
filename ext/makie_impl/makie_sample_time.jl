# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Owned display metadata. A compressed sample row can cover several cycles.
struct SampleStepSpan
    row::Int
    cycle::Int32
    first::Int64
    last::Int64
end
const WalkerStepSpans = Vector{SampleStepSpan}
const ChainStepSpans = Vector{WalkerStepSpans}
const SampleStepSpans = Vector{ChainStepSpans}
_empty_walker_spans(outputs) = [SampleStepSpan[] for _ in outputs]

function _push_step_span!(spans, span::SampleStepSpan)
    if !isempty(spans)
        prev = last(spans)
        if prev.row == span.row && prev.cycle == span.cycle && prev.last < typemax(Int64) && prev.last + 1 == span.first
            spans[end] = SampleStepSpan(prev.row, prev.cycle, prev.first, span.last)
            return spans
        end
    end
    push!(spans, span)
    return spans
end

# Use BAT's merge rule, recording where each input row went. Never change the
# sampler's merging or infer time identity from coordinate equality.
function _append_timed_walkers(state, old, fresh, old_time, fresh_time)
    if all(isempty, old_time) && all(isempty, fresh_time)
        return _append_walker_outputs(state, old, fresh), _empty_walker_spans(old)
    end
    merged = _empty_chain_outputs(state)
    timing = _empty_walker_spans(merged)
    for (source, times) in ((old, old_time), (fresh, fresh_time))
        for j in eachindex(source)
            rowmap = Vector{Int}(undef, length(source[j]))
            for k in eachindex(source[j])
                BAT.checked_push!(merged[j], source[j][k])
                rowmap[k] = length(merged[j])
            end
            for s in times[j]
                _push_step_span!(timing[j], SampleStepSpan(rowmap[s.row], s.cycle, s.first, s.last))
            end
        end
    end
    return merged, timing
end
function _append_timed_chains(state, old, fresh, old_time, fresh_time)
    pairs = [_append_timed_walkers(state, old[i], fresh[i], old_time[i], fresh_time[i]) for i in eachindex(old)]
    return first.(pairs), last.(pairs)
end

# Static callers explicitly declare repetition intervals, rather than asking us
# to guess a weighting scheme or recover cycle boundaries from sample IDs.
function _static_step_spans(samples, intervals)
    isnothing(intervals) && return nothing
    length(intervals) == length(samples) || throw(ArgumentError("step_intervals must have one entry per sample row"))
    result = SampleStepSpan[]
    groups = Dict{Tuple{Int32,Int32,Int32},Vector{SampleStepSpan}}()
    for row in eachindex(samples)
        mass = big(0)
        for item in intervals[row]
            all(k -> hasproperty(item, k) && getproperty(item, k) isa Integer, (:cycle, :first, :last)) ||
                throw(ArgumentError("step intervals require integer cycle, first and last fields"))
            typemin(Int32) <= item.cycle <= typemax(Int32) &&
                1 <= item.first <= item.last <= typemax(Int64) ||
                throw(ArgumentError("step intervals require an Int32 cycle and 1 <= first <= last <= typemax(Int64)"))
            cycle, lo, hi = Int32(item.cycle), Int64(item.first), Int64(item.last)
            mass += big(hi) - lo + 1
            span = SampleStepSpan(row, cycle, lo, hi)
            push!(result, span)
            info = samples.info[row]
            cid = hasproperty(info, :chainid) ? Int32(info.chainid) : Int32(1)
            wid = hasproperty(info, :walkerid) ? Int32(info.walkerid) : hasproperty(info, :walker) ? Int32(info.walker) : Int32(1)
            push!(get!(() -> SampleStepSpan[], groups, (cid, wid, cycle)), span)
        end
        mass == samples.weight[row] || throw(ArgumentError("step interval duration must equal the repetition weight at row $row"))
    end
    for (key, spans) in groups
        sort!(spans; by=s -> s.first)
        all(k -> spans[k-1].last < spans[k].first, 2:length(spans)) ||
            throw(ArgumentError("Overlapping step intervals for chain, walker, cycle $key"))
    end
    return result
end

struct StepIndexGroup
    chain::Int32
    walker::Int32
    cycle::Int32
    rows::Vector{Int}
    starts::Vector{Int64}
    ends::Vector{Int64}
end
struct SampleWindowData{S}
    samples::S
    walker_ranges::Vector{UnitRange{Int}}
    groups::Vector{StepIndexGroup}
    limit::Int64
    timed::Bool
    chainids::Vector{Int32}
end

function _sample_window_data(samples, current_idxs, timing)
    walkers = Any[]
    ranges = UnitRange{Int}[]
    groups = StepIndexGroup[]
    offset = 0
    limit = Int64(0)
    chainids = Int32[]
    for i in eachindex(samples), j in eachindex(samples[i])
        walker = samples[i][j]
        nrows = min(length(walker), current_idxs[i][j])
        push!(walkers, view(walker, 1:nrows))
        push!(ranges, offset + 1:offset + nrows)
        if nrows > 0
            inf = walker.info[1]
            cid = hasproperty(inf, :chainid) ? Int32(inf.chainid) : Int32(i)
            wid = hasproperty(inf, :walkerid) ? Int32(inf.walkerid) : hasproperty(inf, :walker) ? Int32(inf.walker) : Int32(j)
            push!(chainids, cid)
            if !isnothing(timing)
                cycles = Dict{Int32,Vector{SampleStepSpan}}()
                for s in timing[i][j]
                    s.row <= nrows || continue
                    1 <= s.row && 1 <= s.first <= s.last || throw(ArgumentError("Invalid sample timing"))
                    push!(get!(() -> SampleStepSpan[], cycles, s.cycle), s)
                end
                for cycle in sort!(collect(keys(cycles)))
                    spans = cycles[cycle]
                    issorted(spans; by=s -> s.first) || sort!(spans; by=s -> s.first)
                    for k in 2:length(spans)
                        spans[k-1].last < spans[k].first || throw(ArgumentError("Overlapping step intervals in chain $cid, walker $wid, cycle $cycle"))
                    end
                    push!(groups, StepIndexGroup(cid, wid, cycle,
                        [offset + s.row for s in spans], [s.first for s in spans], [s.last for s in spans]))
                    limit = max(limit, last(spans).last)
                end
            end
        end
        isnothing(timing) && (limit = max(limit, nrows))
        offset += nrows
    end
    flat = vcat(walkers...)
    return SampleWindowData(flat, ranges, groups, limit, !isnothing(timing), sort!(unique(chainids)))
end

struct WindowStepGroup
    chain::Int32
    walker::Int32
    cycle::Int32
    rows::Vector{Int}
    starts::Vector{Int64}
    ends::Vector{Int64}
end

function _window_selection(data::SampleWindowData, window)
    selected = WindowStepGroup[]
    if data.timed
        lo, hi = isnothing(window) ? (Int64(1), typemax(Int64)) : window
        mass = Dict{Int,Int64}()
        for g in data.groups
            a, b = searchsortedfirst(g.ends, lo), searchsortedlast(g.starts, hi)
            a > b && continue
            rows = isnothing(window) ? g.rows : g.rows[a:b]
            starts = isnothing(window) ? g.starts : max.(g.starts[a:b], lo)
            ends = isnothing(window) ? g.ends : min.(g.ends[a:b], hi)
            push!(selected, WindowStepGroup(g.chain, g.walker, g.cycle, rows, starts, ends))
            if !isnothing(window)
                for k in eachindex(rows)
                    mass[rows[k]] = Base.Checked.checked_add(get(mass, rows[k], Int64(0)), ends[k] - starts[k] + 1)
                end
            end
        end
        isnothing(window) && return (data.samples, selected)
        rows = sort!(collect(keys(mass)))
        result = data.samples[rows]
        exact_weights = [mass[row] for row in rows]
        clipped_weights = convert.(eltype(result.weight), exact_weights)
        clipped_weights == exact_weights || throw(ArgumentError(
            "Selected step counts cannot be represented exactly by the sample weight type; use integer repetition weights"))
        result.weight .= clipped_weights
        return (result, selected)
    else
        isnothing(window) && return (data.samples, selected)
        lo, hi = window
        rows = Int[]
        for r in data.walker_ranges
            a, b = max(lo, 1), min(hi, length(r))
            a <= b && append!(rows, (first(r) - 1 + a):(first(r) - 1 + b))
        end
        return (data.samples[rows], selected)
    end
end

# Grouping/recency is shared by all cells; coordinates are gathered per marginal.
function _window_trace_groups(groups, nsteps)
    latest = Dict{Tuple{Int32,Int32},WindowStepGroup}()
    for g in groups
        key = (g.chain, g.walker)
        (!haskey(latest, key) || latest[key].cycle < g.cycle) && (latest[key] = g)
    end
    result = NamedTuple{(:chain, :walker, :rows, :recency),Tuple{Int32,Int32,Vector{Int},Vector{Float64}}}[]
    for key in sort!(collect(keys(latest)))
        g = latest[key]
        now = last(g.ends)
        cutoff = now - min(now - 1, nsteps - 1)
        a = searchsortedfirst(g.ends, cutoff)
        rows, recency = Int[], Float64[]
        previous_end = Int64(0)
        for k in a:length(g.rows)
            start = max(g.starts[k], cutoff)
            if !isempty(rows) && start - 1 > previous_end
                push!(result, (chain=g.chain, walker=g.walker, rows=rows, recency=recency))
                rows, recency = Int[], Float64[]
            end
            push!(rows, g.rows[k])
            push!(recency, nsteps == 1 ? 1.0 : clamp(1.0 - Float64(now - g.ends[k]) / (nsteps - 1), 0.0, 1.0))
            previous_end = g.ends[k]
        end
        isempty(rows) || push!(result, (chain=g.chain, walker=g.walker, rows=rows, recency=recency))
    end
    return result
end

# Initialization cursors own only integers, never candidate coordinate arrays.
struct InitStepSegment
    cycle::Int32
    before_step::Int64
    after_step::Int64
    before::Tuple{Int,Int64}
    after::Tuple{Int,Int64}
end
_sample_cursor(samples) = isempty(samples) ? (0, Int64(0)) : (length(samples), Int64(last(samples.weight)))

function BAT.prepare_visualizer_init!(vis::BATVisualizer{BATMakieVisualization}, algorithm)
    _claim_live_visualizer!(vis)
    try
        vis.content.capture_init[] = algorithm.store_burnin && algorithm.sample_weighting isa BAT.RepetitionWeighting
    catch err
        _fail_visualizer!(vis, err, catch_backtrace(), :prepare)
        BAT._safe_clear_visualizer_init!(vis; suppress_interrupt=true)
        BAT._safe_finish_visualizer!(vis; success=false)
        rethrow()
    end
    return nothing
end
BAT.capture_visualizer_init(vis::BATVisualizer{BATMakieVisualization}) = !isnothing(vis.content) && vis.content.capture_init[]
function BAT.clear_visualizer_init!(vis::BATVisualizer{BATMakieVisualization})
    isnothing(vis.content) && return nothing
    empty!(vis.content.init_segments)
    empty!(vis.content.invalid_init)
    vis.content.capture_init[] = false
    return nothing
end
function BAT.reset_visualizer_init!(vis::BATVisualizer{BATMakieVisualization}, chain, walkers)
    lock(vis.content.buffer_lock) do
        for j in walkers
            key = (chain.info.id, chain.current.x.info[j].walkerid)
            delete!(vis.content.init_segments, key)
            delete!(vis.content.invalid_init, key)
        end
    end
    return nothing
end
function BAT.begin_visualizer_init_segment(vis::BATVisualizer{BATMakieVisualization}, chain, output)
    return (cycle=chain.info.cycle, step=chain.stepno, cursors=_sample_cursor.(output), zero=iszero.(chain.current.x.weight))
end
function BAT.end_visualizer_init_segment!(vis::BATVisualizer{BATMakieVisualization}, before, chain, output)
    lock(vis.content.buffer_lock) do
        for j in eachindex(output)
            key = (chain.info.id, chain.current.x.info[j].walkerid)
            if !before.zero[j] || !iszero(chain.current.x.weight[j]) || before.cycle != chain.info.cycle
                push!(vis.content.invalid_init, key)
            else
                segments = get!(() -> InitStepSegment[], vis.content.init_segments, key)
                push!(segments, InitStepSegment(chain.info.cycle, before.step, chain.stepno, before.cursors[j], _sample_cursor(output[j])))
            end
        end
    end
    return nothing
end
function _initial_step_spans(vis, state, outputs)
    state.chain_state.weighting isa BAT.RepetitionWeighting || return nothing
    all(iszero, state.chain_state.current.x.weight) || return nothing
    result = _empty_walker_spans(outputs)
    for j in eachindex(outputs)
        key = (state.chain_state.info.id, state.chain_state.current.x.info[j].walkerid)
        key in vis.content.invalid_init && return nothing
        walker = outputs[j]
        covered = zeros(Int64, length(walker))
        for segment in get(vis.content.init_segments, key, InitStepSegment[])
            r0, w0 = segment.before
            r1, w1 = segment.after
            0 <= r0 <= r1 <= length(walker) || return nothing
            step = segment.before_step
            for row in max(1, r0):r1
                upper = row == r1 ? w1 : walker.weight[row]
                lower = row == r0 ? w0 : 0
                mass = upper - lower
                0 <= mass <= segment.after_step - step || return nothing
                if mass > 0
                    _push_step_span!(result[j], SampleStepSpan(row, segment.cycle, step + 1, step + mass))
                    covered[row] += mass
                    step += mass
                end
            end
            step == segment.after_step || return nothing
        end
        covered == walker.weight || return nothing
    end
    return result
end
