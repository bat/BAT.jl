# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function _init_gridlayout(
    graph::ComputeGraph,
    n::Integer
)
    n = Int(n)
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

        # Keep a fixed n x n grid, collapsing inactive tracks to Fixed(0). The
        # interactive adapter owns one axis per slot; ordinary composition also
        # receives this same stable layout shape.
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


# This adapter consumes only _init_gridlayout's fixed matrix of Axis specs.
# Keep axes in their cells: Makie 0.24's whole-grid reconciler may disconnect a
# block and later reuse it without restoring its Figure membership.
function _bind_interactive_grid!(fig::Figure, gridlayout; live_vis = nothing)
    # Match Makie's existing grid bridge: published plot geometry must not be
    # deep-copied merely to subscribe to the computed layout.
    source = Makie.ComputePipeline.get_observable!(gridlayout; use_deepcopy=false)
    initial = source[]
    grid_size, grid_offsets = initial.size, initial.offsets
    positions = first.(initial.content)
    function validate(spec)
        (spec.size == grid_size && spec.offsets == grid_offsets &&
            first.(spec.content) == positions &&
            all(p -> last(p) isa Makie.BlockSpec && last(p).type === :Axis, spec.content)) ||
            throw(ArgumentError("Interactive BAT grids require fixed Axis positions"))
        return nothing
    end
    validate(initial)
    layout = fig[1, 1] = GridLayout(initial.size...;
        rowsizes=copy(initial.rowsizes), colsizes=copy(initial.colsizes),
        addedrowgaps=copy(initial.rowgaps), addedcolgaps=copy(initial.colgaps),
        alignmode=initial.alignmode, tellwidth=initial.tellwidth,
        tellheight=initial.tellheight, halign=initial.halign, valign=initial.valign)
    axes = [Axis(layout[position...]; spec.kwargs...) for (position, spec) in initial.content]
    plots = [Observable(PlotSpec[]) for _ in axes]
    # Use the same scene-owned updater as Makie's grid reconciler. The public
    # PlotList wrapper in 0.24.15 fails automatic bounds queries when empty.
    # Keep this compatibility dependency here; do not patch Makie globally.
    foreach((axis, specs) -> Makie.update_plotspecs!(axis.scene, specs), axes, plots)
    # Retain only the last generated axis attributes, not old plot generations.
    previous = Union{Nothing,Dict{Symbol,Any}}[nothing for _ in axes]
    track_sizes = Ref((initial.rowsizes, initial.colsizes))
    update_grid! = function(spec)
        validate(spec) # Reject a changed shape before touching any cell.
        sizes = (spec.rowsizes, spec.colsizes)
        if !isequal(track_sizes[], sizes)
            Makie.GridLayoutBase.with_updates_suspended(layout) do
                for (i, size) in enumerate(spec.rowsizes)
                    rowsize!(layout, i, size)
                end
                for (i, size) in enumerate(spec.colsizes)
                    colsize!(layout, i, size)
                end
            end
            track_sizes[] = sizes
        end
        for i in eachindex(axes)
            cell = last(spec.content[i])
            _update_grid_axis!(axes[i], plots[i], cell, previous[i])
            previous[i] = cell.kwargs
        end
        return nothing
    end
    # Register before initial execution, so startup rollback owns the handle.
    if isnothing(live_vis)
        on(update_grid!, fig.scene, source)
    else
        _view_on(update_grid!, live_vis, source)
    end
    update_grid!(initial)
    return nothing
end

function _update_grid_axis!(axis::Axis, plots, spec::Makie.BlockSpec, previous)
    attributes = spec.kwargs
    # Generated v_i/p_i labels identify the ordered variables and diagonal role.
    # Compare generated attributes, never labels a caller may have overwritten.
    semantic_keys = (:xlabel, :ylabel, :xlabelvisible, :ylabelvisible)
    reset = isnothing(previous) ||
        any(key -> !isequal(attributes[key], previous[key]), semantic_keys) ||
        !isequal(attributes[:limits], previous[:limits]) ||
        any(isnothing, attributes[:limits])
    # Makie's matcher can reuse plots across missing/present keyword attributes,
    # even when the converted input type changes (e.g. scalar/vector markersize).
    # Rebuild only on structural changes; ordinary data updates still reuse plots.
    old_plots = plots[]
    compatible = length(old_plots) == length(spec.plots) &&
        all(_same_plot_structure(a, b) for (a, b) in zip(old_plots, spec.plots))
    compatible || (plots[] = PlotSpec[])
    plots[] = spec.plots
    for (key, value) in attributes
        key === :limits && continue
        isequal(to_value(getproperty(axis, key)), value) || setproperty!(axis, key, value)
    end
    if reset
        # Publish new plots before resolving automatic dimensions. Reinstall
        # computed limits even when equal: the caller may have overridden them.
        axis.limits = attributes[:limits]
        Makie.reset_limits!(axis)
    end
    return nothing
end

function _same_plot_structure(a::PlotSpec, b::PlotSpec)
    return a.type === b.type && length(a.args) == length(b.args) &&
        all(typeof(x) === typeof(y) for (x, y) in zip(a.args, b.args)) &&
        keys(a.kwargs) == keys(b.kwargs) &&
        all(typeof(a.kwargs[key]) === typeof(b.kwargs[key]) for key in keys(a.kwargs))
end
