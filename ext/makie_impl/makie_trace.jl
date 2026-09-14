# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Trace2D: always-live overlay of each chain's recent path through a 2D marginal, recency-colored. Grouped by (chainid, walkerid): walkers in one
# chain share stepno ranges. Recency is elapsed MCMC steps (stepno + weight - 1; a row's weight is the steps spent at that position), not stored-row count.
_empty_trace2d_primitives() = (x=Float64[], y=Float64[], chainids=Int32[], walkerids=Int32[], recency=Float64[], group_ranges=Tuple{Int32,Int32,UnitRange{Int}}[], all_chainids=Int32[])

# Whether `samples` has chain identity AND step numbers (MCMCSampleID does; AHMCSampleID lacks stepno) -- distinct from _samples_have_chain_ids.
_samples_have_trace_info(samples) =
    hasfield(eltype(samples.info), :chainid) && hasfield(eltype(samples.info), :stepno)

function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::AbstractVector,
    ::AbstractVector,
    ::AbstractVector,
    ::Trace2D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _empty_trace2d_primitives()
end

# Inputs are the ordinary windowed per-sample inputs: with the step window applied per walker, each group's rows are already its own chronological slice and its "now" is the window's end.
function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    chainids::AbstractVector,
    walkerids::AbstractVector,
    stepnos::AbstractVector,
    recipe::Trace2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    total_n = length(chainids)
    if total_n == 0 || isempty(stepnos)
        return _empty_trace2d_primitives()
    end
    (; trace_nsteps) = config
    x = view(marg_coords, 1, :)
    y = view(marg_coords, 2, :)
    last_steps = stepnos .+ Float64.(weights) .- 1

    # Indices grouped by (chainid, walkerid) via one forward scan, so each group's index vector comes out in chronological order.
    groups = Dict{Tuple{Int32,Int32},Vector{Int}}()
    for idx in eachindex(chainids)
        key = (chainids[idx], walkerids[idx])
        push!(get!(() -> Int[], groups, key), idx)
    end

    out_x = Float64[]
    out_y = Float64[]
    out_chainids = Int32[]
    out_walkerids = Int32[]
    out_recency = Float64[]
    # (chain_id, walker_id, range-into-the-flat-arrays) per surviving group, so compose_plotspecs doesn't rebuild the grouping.
    group_ranges = Tuple{Int32,Int32,UnitRange{Int}}[]

    # Sorted key order, not raw Dict order: hash-order iteration reshuffles the emitted spec order (draw/z-order, SpecApi reconciliation) mid-run.
    for key in sort!(collect(keys(groups)))
        idxs = groups[key]
        chain_id, walker_id = key

        current_step = last_steps[idxs[end]]
        min_step = current_step - trace_nsteps + 1
        # Walk backward from the group's most recent row, keeping rows still within the trace window; idxs is chronological, so stop at the first miss.
        keep = Int[]
        for k in length(idxs):-1:1
            idx = idxs[k]
            last_steps[idx] < min_step && break
            push!(keep, idx)
        end
        isempty(keep) && continue
        range_start = length(out_x) + 1
        for idx in Iterators.reverse(keep) # restore chronological (oldest-of-window-first) order
            push!(out_x, x[idx])
            push!(out_y, y[idx])
            push!(out_chainids, chain_id)
            push!(out_walkerids, walker_id)
            # Normalized by trace_nsteps - 1 so the newest point reaches exactly 1.0; trace_nsteps == 1 is 1.0 outright, <= 0 rejected at config construction.
            r = trace_nsteps == 1 ? 1.0 : clamp((last_steps[idx] - min_step) / (trace_nsteps - 1), 0.0, 1.0)
            push!(out_recency, r)
        end
        push!(group_ranges, (chain_id, walker_id, range_start:length(out_x)))
    end

    # all_chainids: sorted unique chain ids of the FULL (unwindowed) input, so compose's color ranking stays fixed and matches ChainScatter2D's colors.
    return (x=out_x, y=out_y, chainids=out_chainids, walkerids=out_walkerids, recency=out_recency, group_ranges=group_ranges, all_chainids=sort(unique(chainids)))
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Trace2D,
    config::NamedTuple;
    transposed::Bool = false
)
    (; x, y, recency, group_ranges, all_chainids) = primitives
    # Lower-triangle cells swap x/y at compose time -- see _init_gridlayout's invariant comment.
    transposed && ((x, y) = (y, x))

    if isempty(x)
        return PlotSpec[]
    end

    (; markersize) = config
    # Rank-based palette indexing over all_chainids -- see compute's comment above.
    rank_of = Dict(id => r for (r, id) in enumerate(all_chainids))

    specs = PlotSpec[]
    for (chain_id, _, range) in group_ranges
        base_color = _CHAIN_COLOR_PALETTE[mod1(rank_of[chain_id], length(_CHAIN_COLOR_PALETTE))]
        # 0.15 alpha floor: a fully transparent oldest point would be invisible instead of merely faint.
        colors = [RGBA(base_color.r, base_color.g, base_color.b, 0.15 + 0.85 * recency[idx]) for idx in range]
        pts = Point2f.(x[range], y[range])
        length(pts) > 1 && push!(specs, S.Lines(pts; color=colors))
        push!(specs, S.Scatter(pts; color=colors, markersize=markersize * 3))
    end

    return specs
end

function determine_recipe_status(subject::Trace2D, live_recipe_1::R1, live_recipe_2::R2) where {R1<:BATMakieRecipe,R2<:BATMakieRecipe}
    return LiveRecipe()
end

function get_trace_plotspecs(
    inputs::NamedTuple,
    vsel::Tuple{Integer,Integer},
    recipe::Trace2D,
    config::NamedTuple;
    # Forwarded so the trace overlay follows the cell's main-recipe orientation.
    transposed::Bool = false
)
    trace_primitives = getproperty(inputs, primitive_symbol(Trace2D(), vsel))
    return compose_plotspecs(trace_primitives, Trace2D(), config; transposed=transposed)
end
