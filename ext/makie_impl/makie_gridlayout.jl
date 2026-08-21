# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function _init_gridlayout(
    graph::ComputeGraph,
    n::Int64
)
    # A real ComputePipeline computation, not an Observables.lift with reentrant
    # graph[sym][] reads inside the body (those corrupt edge-resolution state).
    # register_computation!'s input list is static, so it names every recipe's
    # primitives exactly once (a duplicate would make a duplicate-field NamedTuple).
    diag_recipes = BAT_MAKIE_RECIPES_1D
    pair_recipes = vcat(BAT_MAKIE_RECIPES_2D, [ChainScatter2D(), Trace2D()])

    primitive_inputs = Symbol[]
    for i in 1:n
        for recipe in diag_recipes
            push!(primitive_inputs, primitive_symbol(recipe, (i, i)))
        end
    end
    # (j, i) with j > i: the (bigger, smaller) key convention of _init_compute_graph.
    for i in 1:n, j in i+1:n
        for recipe in pair_recipes
            push!(primitive_inputs, primitive_symbol(recipe, (j, i)))
        end
    end
    axis_limit_inputs = [Symbol("axis_limits_$i") for i in 1:n]

    control_inputs = [
        :current_idx,
        :idxs, # re-render on vsel changes too, not just new sample batches
        :upper_recipe,
        :diagonal_recipe,
        :lower_recipe,
        :show_stats_upper,
        :show_stats_diag,
        :show_stats_lower,
        :show_trace_upper,
        :show_trace_lower,
    ]

    # Never updated after construction, so a setup-time read (not a declared
    # input) is sufficient.
    triagonal_config = graph[:triagonal_config][]
    diagonal_config = graph[:diagonal_config][]

    register_computation!(
        graph,
        vcat(control_inputs, primitive_inputs, axis_limit_inputs),
        [:gridlayout],
    ) do inputs, changed, cached
        @nospecialize(changed, cached)
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

        # The matrix always stays n x n, with inactive rows/columns collapsed to
        # Fixed(0) instead: shrinking and re-growing the matrix hits a Makie
        # SpecApi reconciliation bug where re-grown blocks never reappear.
        n_active = length(_idxs)
        n_active <= n || throw(ArgumentError("idxs has $(n_active) entries, exceeding the grid size N_max=$n"))
        # Explicit Union{Auto,Fixed} eltype: GridLayoutBase.convert_contentsizes
        # rejects a plain Vector{Any}.
        cellsizes = Union{Auto,Fixed}[i <= n_active ? Auto() : Fixed(0) for i in 1:n]

        for i in 1:n
            diagonal_primitives = getproperty(inputs, primitive_symbol(diagonal_recipe, (i, i)))
            diagonal_plotspecs = compose_plotspecs(diagonal_primitives, diagonal_recipe(), diagonal_config)
            stats_specs_1D = stats_diag ? get_stats_plotspecs(inputs, (i, i), Makie1DStats(), diagonal_config) : PlotSpec[]
            append!(diagonal_plotspecs, stats_specs_1D)

            xlims = getproperty(inputs, Symbol("axis_limits_$i"))
            # Per-cell y-limit, not shared across diagonals: densities have
            # per-variable units, so different scales aren't comparable on one axis.
            diag_y_ext = _diag_y_extent(diagonal_primitives, diagonal_recipe())
            diag_ylims = (isfinite(diag_y_ext) && diag_y_ext > 0) ? (0.0, 1.1 * diag_y_ext) : nothing
            # Decorations are protrusion content drawn outside the cell area --
            # they don't vanish with Fixed(0), so hide them explicitly when inactive.
            cell_active = i <= n_active
            matrix[i, i] = S.Axis(
                plots=diagonal_plotspecs,
                # Matches the 2D cells' aspect=1; otherwise a diagonal cell stretches.
                aspect=1,
                limits=(xlims, diag_ylims),
                xticklabelsvisible=cell_active, xticksvisible=false,
                yticklabelsvisible=cell_active, yticksvisible=false,
                # No fixed ytickformat: Makie's adaptive default handles all scales.
                yticklabelrotation=pi / 2,
                xgridvisible=cell_active,
                ygridvisible=cell_active,
                leftspinevisible=cell_active, rightspinevisible=cell_active,
                topspinevisible=cell_active, bottomspinevisible=cell_active,
                # Plain "" (not L"") when inactive: an empty LaTeXString crashes
                # Makie's glyph-collection even with the label invisible.
                xlabel=cell_active ? L"v_%$(_idxs[i])" : "",
                ylabel=cell_active ? L"p_%$(_idxs[i])" : "",
                xlabelvisible=cell_active,
                ylabelvisible=cell_active,
            )

            for j in i+1:n
                # Orientation invariant: every off-diagonal cell shows x = column
                # variable, y = row variable. Both mirrored cells share the
                # (bigger, smaller)-keyed primitive, oriented for the upper cell.
                upper_primitives = getproperty(inputs, primitive_symbol(upper_recipe, (j, i)))
                upper_plotspecs = compose_plotspecs(upper_primitives, upper_recipe(), triagonal_config)
                stats_specs_2D = stats_upper ? get_stats_plotspecs(inputs, (j, i), Makie2DStats(), triagonal_config) : PlotSpec[]
                append!(upper_plotspecs, stats_specs_2D)
                trace_specs_upper = trace_upper ? get_trace_plotspecs(inputs, (j, i), Trace2D(), triagonal_config) : PlotSpec[]
                append!(upper_plotspecs, trace_specs_upper)

                # xlims (the row variable's limits) serves as this cell's y-limits.
                col_lims = getproperty(inputs, Symbol("axis_limits_$j"))
                cell_active_upper = j <= n_active
                matrix[i, j] = S.Axis(
                    plots=upper_plotspecs,
                    aspect=1,
                    limits=(col_lims, xlims),
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
                # transposed=true swaps x/y at compose time so this mirrored cell's
                # x-axis is its column's variable; overlays get the same flag.
                lower_primitives = getproperty(inputs, primitive_symbol(lower_recipe, (i, j)))
                lower_plotspecs = compose_plotspecs(lower_primitives, lower_recipe(), triagonal_config; transposed=true)
                stats_specs_2D = stats_lower ? get_stats_plotspecs(inputs, (i, j), Makie2DStats(), triagonal_config; transposed=true) : PlotSpec[]
                append!(lower_plotspecs, stats_specs_2D)
                trace_specs_lower = trace_lower ? get_trace_plotspecs(inputs, (i, j), Trace2D(), triagonal_config; transposed=true) : PlotSpec[]
                append!(lower_plotspecs, trace_specs_lower)

                col_lims = getproperty(inputs, Symbol("axis_limits_$j"))
                cell_active_lower = i <= n_active
                matrix[i, j] = S.Axis(
                    plots=lower_plotspecs,
                    aspect=1,
                    limits=(col_lims, xlims),
                    xticklabelsvisible=cell_active_lower, xticksvisible=false,
                    yticklabelsvisible=cell_active_lower, yticksvisible=false,
                    yticklabelrotation=pi / 2,
                    xgridvisible=cell_active_lower,
                    ygridvisible=cell_active_lower,
                    leftspinevisible=cell_active_lower, rightspinevisible=cell_active_lower,
                    topspinevisible=cell_active_lower, bottomspinevisible=cell_active_lower,
                    # See the diagonal cell's comment above re: plain "" vs L"".
                    xlabel=cell_active_lower ? L"v_%$(_idxs[j])" : "",
                    ylabel=cell_active_lower ? L"v_%$(_idxs[i])" : "",
                    xlabelvisible=cell_active_lower,
                    ylabelvisible=cell_active_lower,
                )
            end
        end

        # alignmode=Outside absorbs tick/label protrusions internally so the
        # parent layout never resizes on vsel changes; must be a constructor
        # kwarg -- an alignmode set post-hoc is reset to Inside() on rebuild.
        return (S.GridLayout(matrix; rowsizes=cellsizes, colsizes=cellsizes, alignmode=Outside(44, 44, 16, 40)),)
    end

    return graph[:gridlayout]
end
