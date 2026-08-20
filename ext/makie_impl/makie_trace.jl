# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Trace2D: an always-live overlay (like Mean2D/Std2D/Cov2D -- see
# determine_recipe_status below) showing each MCMC chain's recent path
# through a 2D marginal, as a recency-colored line + markers -- the current
# position opaque, fading toward transparent further back in time. Recency
# is measured in real elapsed MCMC steps (stepno + weight - 1, since a
# stored row's weight is how many consecutive steps the chain spent at that
# position -- see :flat_stepnos' comment in makie_visualizer.jl), not
# stored-row count, so a position the chain dwelled at for a long run of
# rejections ages out of the trace at the correct rate instead of counting
# as a single recent point.
#
# Grouped by (chainid, walkerid) rather than chainid alone: a single BAT
# "chain" can have multiple concurrent walkers stepping in lock-step but
# accepting/rejecting independently, sharing overlapping stepno ranges --
# chainid alone can't disentangle one walker's own trajectory in that case.
# Each group's own "now" is its own most recent step, not a global max
# across all chains, so a slower chain's trace isn't falsely aged by a
# faster one running alongside it.
_empty_trace2d_primitives() = (x=Float64[], y=Float64[], chainids=Int32[], walkerids=Int32[], recency=Float64[], group_ranges=Tuple{Int32,Int32,UnitRange{Int}}[])

function compute_plotting_primitives(
        ::SubArray,
        ::SubArray,
        ::AbstractVector,
        ::AbstractVector,
        ::AbstractVector,
        ::Integer,
        ::Trace2D,
        ::RS,
        ::CS,
        ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
        return _empty_trace2d_primitives()
end

# marg_coords/weights/chainids/walkerids/stepnos here are the *full*,
# untruncated completed-dataset versions (marg_full_symbol/:flat_*_full --
# see their own comments in makie_visualizer.jl), not the shared
# current_idx-truncated ones every other 2D recipe receives. current_idx is
# threaded through separately so each chain's own reveal fraction --
# current_idx / (that chain's own total row count) -- can be computed and
# applied per group below, rather than truncating the flattened array at one
# shared row position (which would reveal one chain's entire
# chain-block-concatenated run before any of the next chain's -- see
# :flat_samples_full's comment for the full diagnosis of why that's wrong
# specifically for a completed, static multi-chain result).
function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        chainids::AbstractVector,
        walkerids::AbstractVector,
        stepnos::AbstractVector,
        current_idx::Integer,
        recipe::Trace2D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        total_n = length(chainids)
        if total_n == 0 || isempty(stepnos) || current_idx <= 0
                return _empty_trace2d_primitives()
        end
        (; trace_nsteps) = config
        x = marg_coords[1, :]
        y = marg_coords[2, :]
        # Last step actually spent at each row's position -- see this file's
        # header comment. weights are converted rather than passed through
        # (not that they're returned raw here anyway, but for the same
        # general reason as the fix in makie_scatter.jl/makie_hist.jl:
        # samples.weight's own concrete eltype isn't guaranteed Float64).
        last_steps = stepnos .+ Float64.(weights) .- 1

        # Overall reveal fraction (how far current_idx is into the *full*
        # dataset) -- applied identically to every chain's own full-length
        # group below, so all chains advance in lockstep regardless of which
        # chain-block they occupy in the underlying storage.
        reveal_frac = clamp(current_idx / total_n, 0.0, 1.0)

        # Indices grouped by (chainid, walkerid) -- built via one forward
        # scan, so each group's own index vector comes out in the same
        # ascending order as the scan, which is chronological order within
        # that group (the flat sample arrays preserve per-chain-walker
        # contiguity/order all the way from the sampler through to here,
        # confirmed during this feature's feasibility review -- never
        # reordered/sorted).
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
        # (chain_id, walker_id, index-range-into-the-above-flat-arrays) per
        # group that survived the window -- computed once here so
        # compose_plotspecs doesn't need to re-scan chainids/walkerids and
        # rebuild an identical grouping Dict from scratch on every call (it
        # used to; this was a real, measurable, entirely avoidable duplicate
        # O(window size) cost paid every time the overlay is drawn).
        group_ranges = Tuple{Int32,Int32,UnitRange{Int}}[]

        for (key, idxs) in groups
                chain_id, walker_id = key
                n_revealed = clamp(round(Int, reveal_frac * length(idxs)), 0, length(idxs))
                n_revealed == 0 && continue
                revealed = @view idxs[1:n_revealed]

                current_step = last_steps[revealed[end]]
                min_step = current_step - trace_nsteps + 1
                # Walk backward from the group's most recently *revealed* row,
                # keeping rows whose last-occupied step still falls within
                # the window; stop at the first row entirely before it
                # (revealed is chronological, so nothing earlier can still be
                # in-window either).
                keep = Int[]
                for k in length(revealed):-1:1
                        idx = revealed[k]
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
                        r = trace_nsteps <= 0 ? 1.0 : clamp((last_steps[idx] - min_step) / trace_nsteps, 0.0, 1.0)
                        push!(out_recency, r)
                end
                push!(group_ranges, (chain_id, walker_id, range_start:length(out_x)))
        end

        return (x=out_x, y=out_y, chainids=out_chainids, walkerids=out_walkerids, recency=out_recency, group_ranges=group_ranges)
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::Trace2D,
        config::NamedTuple;
        transposed::Bool=false
)
        (; x, y, chainids, recency, group_ranges) = primitives
        # Lower-triangle cells swap x/y at compose time -- see
        # _init_gridlayout's invariant comment. Swapped once up front so the
        # per-group Point2f construction below stays untouched.
        transposed && ((x, y) = (y, x))

        if isempty(x)
                return PlotSpec[]
        end

        (; markersize) = config
        # Rank-based palette indexing, consistent with ChainScatter2D (see
        # its own comment) -- ranked among *all* chain ids present in this
        # cell's primitives (not just ones with points surviving into the
        # trace window), so a chain's color matches ChainScatter2D's if both
        # happen to be visible in the same figure. This is a fresh, cheap
        # O(window size) scan over just the surviving points -- unlike the
        # (chainid, walkerid) grouping itself, which no longer needs to be
        # rebuilt here at all (group_ranges, computed once in
        # compute_plotting_primitives, replaces the Dict this function used
        # to reconstruct from scratch on every single call).
        unique_chains = sort(unique(chainids))
        rank_of = Dict(id => r for (r, id) in enumerate(unique_chains))

        specs = PlotSpec[]
        for (chain_id, _, range) in group_ranges
                base_color = _CHAIN_COLOR_PALETTE[mod1(rank_of[chain_id], length(_CHAIN_COLOR_PALETTE))]
                # 0.15 alpha floor (not 0) for the oldest point in the window
                # -- a fully transparent oldest point would make a
                # just-entered-the-window position invisible instead of
                # merely faint.
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
        # Forwarded so the trace overlay follows the same orientation as the
        # cell's main recipe -- see _init_gridlayout's invariant comment.
        transposed::Bool=false
)
        trace_primitives = getproperty(inputs, primitive_symbol(Trace2D(), vsel))
        return compose_plotspecs(trace_primitives, Trace2D(), config; transposed=transposed)
end
