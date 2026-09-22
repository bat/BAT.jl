# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Trace2D uses the shared cycle-aware timing selection.
_empty_trace2d_primitives() = (x=Float64[], y=Float64[], chainids=Int32[], walkerids=Int32[], recency=Float64[], group_ranges=Tuple{Int32,Int32,UnitRange{Int}}[], all_chainids=Int32[])

# Time filtering/grouping is computed once for the graph, not once per marginal.
function _timed_trace_primitives(data, groups, vsel)
    x, y, recency = Float64[], Float64[], Float64[]
    chainids, walkerids = Int32[], Int32[]
    ranges = Tuple{Int32,Int32,UnitRange{Int}}[]
    values = data.samples.v.data
    for g in groups
        start = length(x) + 1
        append!(x, view(values, vsel[1], g.rows))
        append!(y, view(values, vsel[2], g.rows))
        append!(recency, g.recency)
        append!(chainids, fill(g.chain, length(g.rows)))
        append!(walkerids, fill(g.walker, length(g.rows)))
        push!(ranges, (g.chain, g.walker, start:length(x)))
    end
    return (x=x, y=y, chainids=chainids, walkerids=walkerids, recency=recency,
        group_ranges=ranges, all_chainids=data.chainids)
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
