# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function _init_gridlayout(
    graph::ComputeGraph,
    n::Int64
)
    # This used to be a plain Observables.lift(...) over just the 10 "control"
    # observables below, with everything else (every cell's own primitive,
    # the stats/trace overlay primitives, the per-dimension axis limits) read
    # reentrantly via graph[symbol][] *inside* the lift body. That reentrant
    # read pattern is what caused a real, reproducible bug: a real mouse
    # click's callback chain runs synchronously nested inside
    # ComputePipeline's own locked _update!/resolve! call (Observables'
    # notify() is synchronous, and ComputePipeline's graph.lock is a
    # ReentrantLock, so the nested graph[symbol][] read doesn't deadlock --
    # it just silently corrupts ComputePipeline's internal edge-resolution
    # bookkeeping instead). Confirmed directly and reproducibly: a
    # closure-captured, otherwise-immutable Integer parameter (this very `n`)
    # was observed reading back as 100 (matching a config's nbins) in one
    # capture and 93568 (matching the sample count) in another, immediately
    # after a real click -- two different *other* integers from the same
    # closure's environment, not random memory corruption, exactly what
    # stale/reentrant-corrupted edge state would produce. Never reproducible
    # by driving the identical Observable change programmatically (no active
    # GLFW event-poll context involved there) -- only by a real mouse click,
    # which is what let this hide for as long as it did.
    #
    # Fixed by making this a genuine ComputePipeline computation instead,
    # matching the pattern every *other* node in this graph already uses:
    # register_computation! requires a static input list (fixed at
    # registration time, since it can't vary by whichever recipe happens to
    # be selected right now), so this lists *every* symbol this cell matrix
    # could ever need -- every recipe, not just the currently-selected one --
    # rather than just the 10 control signals. ComputePipeline resolves all
    # of them (cheaply -- resolving an edge that isn't dirty is a checked
    # no-op) before invoking this callback, so the callback body itself does
    # zero graph reads, only NamedTuple field lookups on the already-resolved
    # `inputs`. No new work this adds: :diagonal_recipe/:upper_recipe/
    # :lower_recipe are already declared inputs to every per-recipe primitive
    # computation today, so switching recipes already wakes every recipe's
    # (cheap, dead-sentinel) computation regardless of this change.
    # Mean1D/Std1D are already part of BAT_MAKIE_RECIPES_1D (and
    # Mean2D/Cov2D/Std2D already part of BAT_MAKIE_RECIPES_2D) -- only
    # ChainScatter2D and Trace2D need adding, since those two are
    # deliberately excluded from BAT_MAKIE_RECIPES_2D itself (see their own
    # registration comments in makie_compute_graph.jl for why). Re-adding an
    # already-listed recipe here would register the same primitive_symbol
    # twice in register_computation!'s input list below, one of several
    # ways to a duplicate-field NamedTuple error.
    diag_recipes = BAT_MAKIE_RECIPES_1D
    pair_recipes = vcat(BAT_MAKIE_RECIPES_2D, [ChainScatter2D(), Trace2D()])

    primitive_inputs = Symbol[]
    for i in 1:n
        for recipe in diag_recipes
            push!(primitive_inputs, primitive_symbol(recipe, (i, i)))
        end
    end
    # (j, i) with j > i, matching the (bigger, smaller) convention every 2D
    # primitive is actually registered under in _init_compute_graph (shared
    # between the upper and lower cell at the same unordered pair).
    for i in 1:n, j in i+1:n
        for recipe in pair_recipes
            push!(primitive_inputs, primitive_symbol(recipe, (j, i)))
        end
    end
    axis_limit_inputs = [Symbol("axis_limits_$i") for i in 1:n]

    control_inputs = [
        :current_idx,
        :idxs, # re-render on vsel changes too, not just new sample batches --
        # otherwise toggling the picker only appears to work during live
        # sampling, as an accidental side effect of :current_idx also changing.
        :upper_recipe,
        :diagonal_recipe,
        :lower_recipe,
        :show_stats_upper,
        :show_stats_diag,
        :show_stats_lower,
        :show_trace_upper,
        :show_trace_lower,
    ]

    # :triagonal_config/:diagonal_config are never updated after construction
    # (no code anywhere calls update!(graph, triagonal_config=...) or
    # diagonal_config=...), so a single plain read here -- at setup time,
    # before this computation is ever registered/invoked, not reentrant -- is
    # exactly equivalent to re-reading them on every invocation, and avoids
    # needing them as declared inputs at all.
    triagonal_config = graph[:triagonal_config][]
    diagonal_config = graph[:diagonal_config][]

    register_computation!(
        graph,
        vcat(control_inputs, primitive_inputs, axis_limit_inputs),
        [:gridlayout],
    ) do inputs, changed, cached
        idx = inputs.current_idx
        _idxs = inputs.idxs
        upper_recipe = inputs.upper_recipe
        diagonal_recipe = inputs.diagonal_recipe
        lower_recipe = inputs.lower_recipe
        stats_upper = inputs.show_stats_upper
        stats_diag = inputs.show_stats_diag
        stats_lower = inputs.show_stats_lower
        trace_upper = inputs.show_trace_upper
        trace_lower = inputs.show_trace_lower

        matrix = Matrix{Any}(undef, n, n)

        # Deselecting a variable should visually remove its row/column and
        # let the remaining ones grow into the freed space. The naive way to
        # do that -- building an n_active x n_active S.GridLayout matrix
        # instead of always n x n -- hits a genuine Makie SpecApi
        # reconciliation bug: shrinking the matrix then growing it back to a
        # size it held *before* tries to reuse a previously-disconnected
        # block at that position instead of creating a fresh one, and it
        # never reappears (confirmed empirically via direct instrumentation
        # of both the compute graph and this very closure -- both correctly
        # recomputed n_active back to its original value, yet the rendered
        # scene stayed frozen at the smaller size).
        #
        # So the matrix itself always stays n x n (sidestepping that
        # reconciliation path entirely, since the *set of positions* never
        # changes) and inactive rows/columns are instead collapsed to
        # Fixed(0) via GridLayoutSpec's own rowsizes/colsizes -- the same
        # mechanism (and the same Fixed(0)-collapses-a-cell trick used
        # elsewhere in this file for the collapsible controls row) just
        # applied here to grid rows/columns instead of Figure rows.
        # GridLayoutBase then naturally redistributes the same total area
        # across only the n_active non-collapsed cells.
        n_active = length(_idxs)
        @assert n_active <= n "idxs has $(n_active) entries, exceeding the grid size N_max=$n"
        # Explicitly typed as Union{Auto,Fixed} (not left to infer as
        # Vector{Any}) -- GridLayoutBase.convert_contentsizes requires
        # Vector{<:ContentSize} and rejects a plain Vector{Any}, which is
        # what an untyped comprehension over two different concrete types
        # produces.
        cellsizes = Union{Auto,Fixed}[i <= n_active ? Auto() : Fixed(0) for i in 1:n]

        # Shared y-axis limit across all diagonal cells (0 to 1.1x the peak
        # value of any active diagonal's own recipe primitives -- see
        # _diag_y_extent's per-recipe methods), rather than each diagonal
        # auto-scaling to its own peak independently. nothing (Makie's usual
        # autolimits) until real data exists in at least one active diagonal
        # cell, matching the graceful-degradation-before-data pattern used
        # elsewhere in this function (e.g. axis_limits_i falling back to
        # (0,1)).
        # Filters out non-finite extents (e.g. a NaN from a zero-weight
        # diagonal cell -- see makie_hist.jl's _safe_normalize) before
        # taking the max: maximum()'s NaN-poisoning would otherwise let one
        # degenerate/zero-weight cell silently corrupt the shared y-axis
        # limit for every *other*, perfectly valid diagonal cell too.
        diag_y_max = maximum(
            Iterators.filter(isfinite, (_diag_y_extent(getproperty(inputs, primitive_symbol(diagonal_recipe, (i, i))), diagonal_recipe()) for i in 1:n_active));
            init=0.0
        )
        diag_ylims = diag_y_max > 0 ? (0.0, 1.1 * diag_y_max) : nothing

        for i in 1:n
            diagonal_primitives = getproperty(inputs, primitive_symbol(diagonal_recipe, (i, i)))
            diagonal_plotspecs = compose_plotspecs(diagonal_primitives, diagonal_recipe(), diagonal_config)
            stats_specs_1D = stats_diag ? get_stats_plotspecs(inputs, (i, i), Makie1DStats(), diagonal_config) : []
            append!(diagonal_plotspecs, stats_specs_1D)

            xlims = getproperty(inputs, Symbol("axis_limits_$i"))
            # A Fixed(0) row/column (cellsizes above) collapses the cell's
            # own plotting area to zero, but ticks/tick-labels/gridlines are
            # protrusion content drawn *outside* that area -- they don't
            # automatically disappear just because the cell they're attached
            # to has shrunk to nothing (confirmed empirically: leftover tick
            # marks/labels from deselected variables were still visible).
            # Explicitly forcing every decoration off for an inactive cell,
            # not just relying on it having zero size, is what actually
            # removes them. This grid's own alignmode=Outside(...) (see the
            # S.GridLayout call below) means its reported protrusion to its
            # parent (fig.layout, in _build_fig) is always a fixed margin,
            # regardless of which/how many rows are actually active, so
            # there's no size-jump risk from an inactive cell's decorations
            # being on/off the way there once was under the GridLayoutBase
            # default Inside() alignmode.
            cell_active = i <= n_active
            matrix[i, i] = S.Axis(
                plots=diagonal_plotspecs,
                # Matches the upper/lower 2D cells' aspect=1 below -- without
                # it, a diagonal cell has no fixed visual aspect ratio at all
                # (unlike a 2D cell, whose data-derived x/y limits happen to
                # somewhat constrain its shape) and stretches to fill whatever
                # rectangle the GridLayout/decorations leave it, typically
                # taller than wide for a 1D density/histogram.
                aspect=1,
                limits=(xlims, diag_ylims),
                # Every cell shows its own bottom/left tick labels + axis
                # labels now (per explicit request), with the tick *marks*
                # themselves removed everywhere to keep the added clutter in
                # check -- xticksvisible/yticksvisible=false rather than
                # tying them to visibility of the labels.
                xticklabelsvisible=cell_active, xticksvisible=false,
                yticklabelsvisible=cell_active, yticksvisible=false,
                yticklabelrotation=pi / 2,
                ytickformat="{:.1f}",
                xgridvisible=cell_active,
                ygridvisible=cell_active,
                leftspinevisible=cell_active, rightspinevisible=cell_active,
                topspinevisible=cell_active, bottomspinevisible=cell_active,
                # v_i on x, p_i on y -- every diagonal cell shows both,
                # unconditionally (same "every cell always" rule as ticks
                # above): each diagonal is the unique anchor identifying
                # which variable its row/column represents.
                # Plain "" (not L"") for the inactive/hidden case -- an empty
                # LaTeXString crashes Makie's glyph-collection computation
                # even when xlabelvisible=false (confirmed empirically: the
                # visibility flag doesn't skip glyph layout for the
                # underlying text, only its own rendering).
                xlabel=cell_active ? L"v_%$(_idxs[i])" : "",
                ylabel=cell_active ? L"p_%$(_idxs[i])" : "",
                xlabelvisible=cell_active,
                ylabelvisible=cell_active,
            )

            for j in i+1:n
                # NOTE: upper/lower cells at mirrored grid positions reuse the same
                # computed primitive (row1=larger-index var, row2=smaller-index var),
                # to avoid computing each variable pair twice. Upper cells' axis
                # limits (below) follow that layout, but the lower-triangle cells
                # (see the `for j in 1:i-1` loop) assign x/y limits the other way
                # around, so a lower cell's x-axis doesn't align with its column's
                # diagonal cell -- inconsistent with the standard corner-plot
                # convention (every cell in column c shares column c's x-range).
                # Fixing this properly needs a transpose flag threaded through
                # every 2D recipe's compose_plotspecs (Hist2D, KDE2D, QuantileHist2D,
                # QuantileKDE2D, Hexbin2D, Scatter2D, Cov2D, Std2D, Mean2D,
                # Errorbars2D), not a local change here -- deferred as its own pass.
                upper_primitives = getproperty(inputs, primitive_symbol(upper_recipe, (j, i)))
                upper_plotspecs = compose_plotspecs(upper_primitives, upper_recipe(), triagonal_config)
                stats_specs_2D = stats_upper ? get_stats_plotspecs(inputs, (j, i), Makie2DStats(), triagonal_config) : PlotSpec[]
                append!(upper_plotspecs, stats_specs_2D)
                trace_specs_upper = trace_upper ? get_trace_plotspecs(inputs, (j, i), Trace2D(), triagonal_config) : PlotSpec[]
                append!(upper_plotspecs, trace_specs_upper)

                ylims = getproperty(inputs, Symbol("axis_limits_$j"))
                # i < j always in this loop, so j <= n_active already implies
                # i <= n_active -- checking j alone is sufficient here.
                cell_active_upper = j <= n_active
                matrix[i, j] = S.Axis(
                    plots=upper_plotspecs,
                    aspect=1,
                    limits=(ylims, xlims),
                    # Every cell shows its own bottom/left ticks+labels now --
                    # see the diagonal cell's matching comment above. No
                    # xaxisposition/yaxisposition override either (was :top/
                    # :right), so this defaults to the same bottom/left as
                    # every other cell.
                    xticklabelsvisible=cell_active_upper, xticksvisible=false,
                    yticklabelsvisible=cell_active_upper, yticksvisible=false,
                    yticklabelrotation=pi / 2,
                    xgridvisible=cell_active_upper,
                    ygridvisible=cell_active_upper,
                    leftspinevisible=cell_active_upper, rightspinevisible=cell_active_upper,
                    topspinevisible=cell_active_upper, bottomspinevisible=cell_active_upper,
                    # See the diagonal cell's comment above re: plain "" vs L"".
                    xlabel=cell_active_upper ? L"v_%$(_idxs[j])" : "",
                    ylabel=cell_active_upper ? L"v_%$(_idxs[i])" : "",
                    xlabelvisible=cell_active_upper,
                    ylabelvisible=cell_active_upper,
                )
            end
            for j in 1:i-1
                lower_primitives = getproperty(inputs, primitive_symbol(lower_recipe, (i, j)))
                lower_plotspecs = compose_plotspecs(lower_primitives, lower_recipe(), triagonal_config)
                stats_specs_2D = stats_lower ? get_stats_plotspecs(inputs, (i, j), Makie2DStats(), triagonal_config) : PlotSpec[]
                append!(lower_plotspecs, stats_specs_2D)
                trace_specs_lower = trace_lower ? get_trace_plotspecs(inputs, (i, j), Trace2D(), triagonal_config) : PlotSpec[]
                append!(lower_plotspecs, trace_specs_lower)

                ylims = getproperty(inputs, Symbol("axis_limits_$j"))
                # j < i always in this loop, so i <= n_active already implies
                # j <= n_active -- checking i alone is sufficient here.
                cell_active_lower = i <= n_active
                matrix[i, j] = S.Axis(
                    plots=lower_plotspecs,
                    aspect=1,
                    limits=(xlims, ylims),
                    # Every cell shows its own bottom/left ticks+labels now --
                    # see the diagonal cell's matching comment above (this
                    # cell already defaulted to bottom/left, unlike upper).
                    xticklabelsvisible=cell_active_lower, xticksvisible=false,
                    yticklabelsvisible=cell_active_lower, yticksvisible=false,
                    yticklabelrotation=pi / 2,
                    xgridvisible=cell_active_lower,
                    ygridvisible=cell_active_lower,
                    leftspinevisible=cell_active_lower, rightspinevisible=cell_active_lower,
                    topspinevisible=cell_active_lower, bottomspinevisible=cell_active_lower,
                    # See the diagonal cell's comment above re: plain "" vs L"".
                    xlabel=cell_active_lower ? L"v_%$(_idxs[i])" : "",
                    ylabel=cell_active_lower ? L"v_%$(_idxs[j])" : "",
                    xlabelvisible=cell_active_lower,
                    ylabelvisible=cell_active_lower,
                )
            end
        end

        # alignmode=Outside(...) (rather than the GridLayoutBase default
        # Inside()) makes this grid absorb its own tick/axis-label protrusion
        # margin internally instead of reporting it upward to its parent --
        # see the cell_active comment above for why that's what actually
        # stops _build_fig's row 1 from resizing whenever the vsel selection
        # changes. Values match what the parent (fig.layout in _build_fig)
        # used to reserve for this grid's protrusion before that
        # responsibility moved here: 44px left/right comfortably covers
        # 2-digit variable indices' v_i/p_i labels (confirmed empirically
        # across N_max=3/5/15 and both full and partial vsel selections), 40px
        # top covers the diagonal's own p_i label sitting above its tick
        # labels, 16px bottom is Makie's own default margin (nothing is
        # positioned further out than that on this side). Passed as a
        # constructor kwarg, not set post-hoc (gl.alignmode[] = ...): this
        # grid is rebuilt fresh via S.GridLayout(...) on every reactive
        # update, and an alignmode assigned after construction gets silently
        # reset to Inside() on the next rebuild.
        return (S.GridLayout(matrix; rowsizes=cellsizes, colsizes=cellsizes, alignmode=Outside(44, 44, 16, 40)),)
    end

    return graph[:gridlayout]
end
