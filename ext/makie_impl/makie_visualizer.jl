# This file is a part of BAT.jl, licensed under the MIT License (MIT).

const BAT_MAKIE_RECIPES_1D = [
    Hist1D(),
    QuantileHist1D(),
    KDE1D(),
    QuantileKDE1D(),
    Std1D(),
    Mean1D(),
    Errorbars1D(),
    PDF1D()
]

const BAT_MAKIE_RECIPES_2D = [
    Hist2D(),
    QuantileHist2D(),
    Hexbin2D(),
    KDE2D(),
    QuantileKDE2D(),
    Scatter2D(),
    Cov2D(),
    Std2D(),
    Mean2D(),
    Errorbars2D()
]

function recipe_symbol(recipe::R) where {R<:BATMakieRecipe}
    return Symbol(string(typeof(recipe)))
end

function primitive_symbol(recipe, vsel::Tuple{Int64,Int64})
    return Symbol(string(recipe) * "_prim" * "_" * "$(vsel[1])" * "$(vsel[2])")
end

function primitive_symbol(recipe::R, vsel::Tuple{Int64,Int64}) where {R<:BATMakieRecipe}
    return Symbol(string(typeof(recipe)) * "_prim" * "_" * "$(vsel[1])" * "$(vsel[2])")
end

function marg_symbol(vsel::Tuple{Int64,Int64})
    return Symbol("marg_$(vsel[1])$(vsel[2])")
end


function BATVisualizer(vis::BATMakieVisualization)
    (; recipes, N_max, triagonal_config, diagonal_config) = vis

    # TODO: Think about whether or not the more expressive vsel symbol approach is desired, or integers are sufficient
    # idxs = vsel isa AbstractVector{<:Int64} ? vsel : reduce(vcat, asindex.(Ref(vs), vsel))

    graph = _init_compute_graph(
        recipes,
        triagonal_config,
        diagonal_config,
        N_max,
    )

    buffer_lock = ReentrantLock()
    content = (
        graph=graph,
        buffer_lock=buffer_lock,
        # Tied to buffer_lock so sampling threads blocked on the high-watermark
        # (in update_visualizer_impl!) and the listener's flush-and-notify both
        # synchronize through the same lock, per Threads.Condition's contract.
        buffer_cond=Threads.Condition(buffer_lock),
        chain_ids=Vector{Integer}(),
        output_buffer=Vector{Vector{DensitySampleVector}}(),
        n_buffer_samples=Ref(0),
        # Current flush-trigger threshold. Starts at vis.n_batch; when
        # adaptive_batching is on, flush_buffer! grows it geometrically after
        # every flush. update_visualizer_impl!'s backpressure ceiling is always
        # derived from this (scaled by the configured max_buffered/n_batch
        # ratio) rather than a separately-growing value, so the two can never
        # drift out of the relationship where blocking would deadlock the
        # sampler against a flush trigger it's not allowed to reach.
        effective_batch_size=Ref(vis.n_batch),
        buffer_ratio=vis.max_buffered / vis.n_batch,
        is_live=Threads.Atomic{Bool}(true),
        listener_task=Ref{Union{Task,Nothing}}(nothing),
        # Set once in init_visualizer! (the model's dimensionality isn't known yet
        # here); read by _apply_vsel! to (re-)validate any requested vsel change.
        n_dof=Ref{Int}(0)
    )

    return BATVisualizer(vis, content)
end


function _init_compute_graph(
    recipes::NamedTuple,
    triagonal_config::NamedTuple,
    diagonal_config::NamedTuple,
    n::Integer,
)
    graph = ComputeGraph()

    smpls = Any[] #Vector{Vector{DensitySampleVector}}()

    add_input!(graph, :samples, smpls)

    curr_idxs = Vector{Vector{Integer}}()
    add_input!(graph, :current_idxs, curr_idxs)

    register_computation!(graph,
        [:samples, :current_idxs],
        [:flat_samples],
    ) do inputs, changed, cached
        samples = inputs.samples
        current_idxs = inputs.current_idxs

        walker_views = Any[] #Vector{DensitySampleVector}()
        for i in eachindex(samples)
            for j in eachindex(samples[i])
                push!(walker_views, view(samples[i][j], 1:current_idxs[i][j]))
            end
        end
        return (vcat(walker_views...),)
    end

    map!(smpls -> length(smpls),
        graph,
        :flat_samples,
        :current_idx
    )
    map!(
        # Empty view of the real weight type/eltype instead of a hardcoded Int64
        # placeholder, matching the empty marg_coords convention used elsewhere.
        (smpls, idx) -> view(smpls.weight, 1:idx),
        graph,
        [:flat_samples, :current_idx],
        :flat_weights
    )

    add_input!(graph, :idxs, Integer[])

    register_computation!(graph,
        [:idxs],
        [:vsel_map],
    ) do inputs, changed, cached
        idxs = inputs.idxs
        # idxs is meant to be user-changeable at runtime (via _apply_vsel!, which
        # enforces this), so this is a full stateless recompute on every change --
        # not a cache mutated in place -- rather than relying on any invariant
        # about how idxs evolves. Cheap regardless: n<=N_max is small and idxs
        # changes are rare (user-driven), not a per-sample-batch occurrence.
        @assert length(idxs) <= n "idxs has $(length(idxs)) entries, exceeding the grid size N_max=$n"

        n_active = length(idxs)
        vsel_map = Matrix{Tuple{Int,Int}}(undef, n, n)
        for i in 1:n, j in 1:n
            # (0, 0) is an unreachable sentinel for cells beyond the current
            # selection -- live_map already keeps these cells from being read,
            # but this way a hypothetical future bypass fails loudly (index 0)
            # instead of silently reusing a stale selection.
            vsel_map[i, j] = (i <= n_active && j <= n_active) ? (idxs[i], idxs[j]) : (0, 0)
        end

        return (vsel_map,)
    end

    register_computation!(graph,
        [:idxs],
        [:live_map],
    ) do inputs, changed, cached
        idxs = inputs.idxs
        @assert length(idxs) <= n "idxs has $(length(idxs)) entries, exceeding the grid size N_max=$n"

        live_map = fill(false, n, n)

        n_active = length(idxs)
        for i in 1:n_active
            for j in 1:n_active
                live_map[i, j] = true
            end
        end
        return (live_map,)
    end

    # Control nodes for what primitives are computed and what are set to emtpy
    add_input!(graph, :upper_recipe, recipes.upper)
    add_input!(graph, :diagonal_recipe, recipes.diagonal)
    add_input!(graph, :lower_recipe, recipes.lower)

    add_input!(graph, :show_stats_upper, false)
    add_input!(graph, :show_stats_diag, false)
    add_input!(graph, :show_stats_lower, false)

    add_input!(graph, :triagonal_config, triagonal_config)
    add_input!(graph, :diagonal_config, diagonal_config)

    for recipe in vcat(BAT_MAKIE_RECIPES_1D, BAT_MAKIE_RECIPES_2D)
        add_input!(graph, Symbol("$(typeof(recipe))"), recipe)
    end

    for i in 1:n
        #1D marginal views
        marg_sym = marg_symbol((i, i))

        map!(
            # Dead-branch view is over smpls.v.data itself (empty index sets,
            # so it's always safe regardless of current_idx/n_dof), not a
            # hardcoded ElasticMatrix placeholder: bat_makie_plot's static path
            # backs samples with a plain Matrix, and once vsel can change at
            # runtime there (via the picker), a live view (Matrix-backed) and a
            # dead view (previously hardcoded ElasticMatrix-backed) of the same
            # node would be incompatible types, erroring on the transition.
            (smpls, vsel_map, current_idx, live_map) -> (current_idx > 0 && live_map[i, i]) ? view(smpls.v.data, [vsel_map[i, i][1]], 1:current_idx) : view(smpls.v.data, Int[], 1:0),
            graph,
            [:flat_samples, :vsel_map, :current_idx, :live_map],
            marg_sym
        )

        map!(
            (marg, current_idx) -> isempty(marg) ? (0.0, 1.0) : (minimum(marg) - 0.1 * abs(minimum(marg)), maximum(marg) + 0.1 * abs(minimum(marg))),
            graph,
            [marg_sym, :current_idx],
            Symbol("axis_limits_$i")
        )

        primitive_symbols_1D = [primitive_symbol(recipe, (i, i)) for recipe in BAT_MAKIE_RECIPES_1D]

        for k in eachindex(primitive_symbols_1D)
            recipe = BAT_MAKIE_RECIPES_1D[k]
            # Persistent per-cell accumulator for incremental recipes (Mean1D,
            # Std1D) -- captured by the closure below, one independent instance
            # per (cell, recipe), so it survives across ticks instead of being
            # rebuilt from scratch on every recompute.
            running_state = is_incremental(recipe) ? _IncrementalUvState() : nothing

            register_computation!(graph,
                [marg_sym, :flat_weights, :diagonal_recipe, :live_map, :diagonal_config, :vsel_map],
                [primitive_symbols_1D[k]]
            ) do inputs, changed, cached
                coords, weights, live_recipe, live_map, config, vsel_map = inputs
                cell_status = live_map[i, i] ? LiveCell() : DeadCell()
                recipe_status = determine_recipe_status(recipe, live_recipe())
                primitives = if cell_status isa LiveCell && is_incremental(recipe)
                    _update_stats!(running_state, vec(coords), weights, vsel_map[i, i][1])
                    compute_stats_primitives(recipe, running_state, config)
                else
                    compute_plotting_primitives(coords, weights, recipe, recipe_status, cell_status, config)
                end

                return (primitives,)
            end
        end

        for j in i+1:n
            marg_sym_2D = marg_symbol((j, i))
            map!(
                # See the 1D case above for why this is an empty view over the
                # real smpls.v.data rather than a hardcoded ElasticMatrix.
                (smpls, vsel_map, current_idx, live_map) -> (current_idx > 0 && live_map[j, i]) ? view(smpls.v.data, [vsel_map[j, i]...], 1:current_idx) : view(smpls.v.data, Int[], 1:0),
                graph,
                [:flat_samples, :vsel_map, :current_idx, :live_map],
                marg_sym_2D
            )

            primitive_symbols_2D = [primitive_symbol(recipe, (j, i)) for recipe in BAT_MAKIE_RECIPES_2D]

            for k in eachindex(primitive_symbols_2D)
                recipe = BAT_MAKIE_RECIPES_2D[k]
                # Per-(cell, recipe) persistent accumulator for the incremental
                # 2D stats recipes (Mean2D, Cov2D, Std2D) -- see the 1D case above.
                running_state = is_incremental(recipe) ? _IncrementalMvState() : nothing

                register_computation!(graph,
                    [marg_sym_2D, :flat_weights, :upper_recipe, :lower_recipe, :live_map, :triagonal_config, :vsel_map],
                    [primitive_symbols_2D[k]]
                ) do inputs, changed, cached
                    coords, weights, live_recipe_upper, live_recipe_lower, live_map, config, vsel_map = inputs
                    cell_status = live_map[i, j] ? LiveCell() : DeadCell()
                    recipe_status = determine_recipe_status(recipe, live_recipe_upper(), live_recipe_lower())
                    primitives = if cell_status isa LiveCell && is_incremental(recipe)
                        _update_stats!(running_state, coords, weights, vsel_map[j, i])
                        compute_stats_primitives(recipe, running_state, config)
                    else
                        compute_plotting_primitives(coords, weights, recipe, recipe_status, cell_status, config)
                    end
                    return (primitives,)
                end
            end
        end
    end

    return graph
end

function _init_gridlayout(
    graph::ComputeGraph,
    n::Int64
)
    gridlayout = lift(
        graph[:current_idx],
        graph[:idxs], # re-render on vsel changes too, not just new sample batches --
        # otherwise toggling the picker only appears to work during live
        # sampling, as an accidental side effect of :current_idx also changing.
        graph[:upper_recipe],
        graph[:diagonal_recipe],
        graph[:lower_recipe],
        graph[:show_stats_upper],
        graph[:show_stats_diag],
        graph[:show_stats_lower]
    ) do idx, _idxs, upper_recipe, diagonal_recipe, lower_recipe, stats_upper, stats_diag, stats_lower
        matrix = Matrix{Any}(undef, n, n)
        triagonal_config = graph[:triagonal_config][]
        diagonal_config = graph[:diagonal_config][]

        for i in 1:n
            diagonal_primitives = graph[primitive_symbol(diagonal_recipe, (i, i))][]
            diagonal_plotspecs = compose_plotspecs(diagonal_primitives, diagonal_recipe(), diagonal_config)
            stats_specs_1D = stats_diag ? get_stats_plotspecs(graph, (i, i), Makie1DStats(), diagonal_config) : []
            append!(diagonal_plotspecs, stats_specs_1D)

            xlims = graph[Symbol("axis_limits_$i")][]
            show_x_cosmetics = (i == n) || (i == 1)
            matrix[i, i] = S.Axis(
                plots=diagonal_plotspecs,
                limits=(xlims, nothing),
                xticklabelsvisible=show_x_cosmetics, xticksvisible=show_x_cosmetics,
                yticklabelsvisible=true,
                yticklabelrotation=pi / 2,
                ytickformat="{:.1f}",
                xgridvisible=true,
                ygridvisible=true,
                xaxisposition=(i == 1) ? :top : :bottom
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
                upper_primitives = graph[primitive_symbol(upper_recipe, (j, i))][]
                upper_plotspecs = compose_plotspecs(upper_primitives, upper_recipe(), triagonal_config)
                stats_specs_2D = stats_upper ? get_stats_plotspecs(graph, (j, i), Makie2DStats(), triagonal_config) : PlotSpec[]
                append!(upper_plotspecs, stats_specs_2D)

                ylims = graph[Symbol("axis_limits_$j")][]
                show_y_cosmetics = (j == n)
                matrix[i, j] = S.Axis(
                    plots=upper_plotspecs,
                    aspect=1,
                    limits=(ylims, xlims),
                    xticklabelsvisible=show_x_cosmetics, xticksvisible=show_x_cosmetics,
                    yticklabelsvisible=show_y_cosmetics, yticksvisible=show_y_cosmetics,
                    yticklabelrotation=pi / 2,
                    xgridvisible=true,
                    ygridvisible=true,
                    xaxisposition=:top,
                    yaxisposition=:right,
                )
            end
            for j in 1:i-1
                lower_primitives = graph[primitive_symbol(lower_recipe, (i, j))][]
                lower_plotspecs = compose_plotspecs(lower_primitives, lower_recipe(), triagonal_config)
                stats_specs_2D = stats_lower ? get_stats_plotspecs(graph, (i, j), Makie2DStats(), triagonal_config) : PlotSpec[]
                append!(lower_plotspecs, stats_specs_2D)

                ylims = graph[Symbol("axis_limits_$j")][]
                show_y_cosmetics = (j == 1)
                matrix[i, j] = S.Axis(
                    plots=lower_plotspecs,
                    aspect=1,
                    limits=(xlims, ylims),
                    xticklabelsvisible=show_x_cosmetics, xticksvisible=show_x_cosmetics,
                    yticklabelsvisible=show_y_cosmetics, yticksvisible=show_y_cosmetics,
                    yticklabelrotation=pi / 2,
                    xgridvisible=true,
                    ygridvisible=true,
                )
            end
        end
        return S.GridLayout(matrix)
    end

    return gridlayout
end


function _build_fig(
    graph::ComputeGraph,
    gridlayout::Any,
    picker_info::Union{NamedTuple,Nothing}=nothing
)
    fig = Figure()

    plot(fig[1, 1], gridlayout)

    colsize!(fig.layout, 1, Aspect(1, 1))
    rowsize!(fig.layout, 1, Relative(0.8))

    ui_layout = fig[2, 1] = GridLayout()
    # A dedicated, fixed-height row for the "Adjust vsel" button -- not nested
    # inside ui_layout (which shares row 2's space with more and more menus/
    # toggles as it grows, squeezing everything in it) and not inside the
    # picker's own collapsible column (or the button would vanish along with
    # the panel it's meant to reveal).
    button_row = fig[3, 1] = GridLayout()
    rowsize!(fig.layout, 3, Fixed(40))

    options2D = [
        ("QuantileHist", QuantileHist2D),
        ("Hist", Hist2D),
        ("Scatter", Scatter2D),
        ("Hexbin", Hexbin2D),
        ("QuantileKDE", QuantileKDE2D),
        ("KDE", KDE2D),
    ]
    options1D = [
        ("QuantileHist", QuantileHist1D),
        ("Hist", Hist1D),
        ("KDE", KDE1D),
        ("QuantileKDE", QuantileKDE1D),
        ("PDF", PDF1D),
    ]

    default_upper = options2D[findfirst(x -> x[2] == graph[:upper_recipe][], options2D)][1]
    default_diag = options1D[findfirst(x -> x[2] == graph[:diagonal_recipe][], options1D)][1]
    default_lower = options2D[findfirst(x -> x[2] == graph[:lower_recipe][], options2D)][1]

    menu_upper = Menu(
        fig,
        options=options2D,
        default=default_upper
    )
    menu_diagonal = Menu(
        fig,
        options=options1D,
        default=default_diag
    )
    menu_lower = Menu(
        fig,
        options=options2D,
        default=default_lower
    )

    curr_idxs = graph[:current_idxs][]
    # curr_idxs[1] being non-empty only means the (single) chain has at least
    # one walker -- it says nothing about whether any samples have actually
    # been produced yet. Without also checking current_idx > 0, this builds a
    # Slider with an empty 1:0 range (crashes) whenever _build_fig runs before
    # any samples exist, which for the live path is always (it's called
    # synchronously at figure construction, before the first async flush).
    show_slider = length(curr_idxs) == 1 && !isempty(curr_idxs[1]) && graph[:current_idx][] > 0
    if show_slider
        slider_curr_idx = Slider(fig, range=1:graph[:current_idx][], startvalue=graph[:current_idx][])
    end

    ui_layout[2, 1] = Label(fig, "Upper")
    ui_layout[3, 1] = Label(fig, "Diagonal")
    ui_layout[4, 1] = Label(fig, "Lower")

    ui_layout[1, 2] = Label(fig, "Recipe")
    ui_layout[2, 2] = menu_upper
    ui_layout[3, 2] = menu_diagonal
    ui_layout[4, 2] = menu_lower

    ui_layout[1, 3] = Label(fig, "Stats overlay")
    toggle_upper = Toggle(ui_layout[2, 3], active=false)
    toggle_diag = Toggle(ui_layout[3, 3], active=false)
    toggle_lower = Toggle(ui_layout[4, 3], active=false)

    if show_slider
        ui_layout[5, 1] = Label(fig, "Current Idx")
        ui_layout[5, 2] = slider_curr_idx
    end

    colsize!(ui_layout, 1, Auto())
    colsize!(ui_layout, 2, 200)
    colsize!(ui_layout, 3, Auto())

    rowsize!(fig.layout, 2, Auto())

    if show_slider
        # show_slider guarantees a single chain, but that chain may still have
        # multiple walkers; pan all of them to the same position.
        on(slider_curr_idx.value) do curr_idx
            n_walkers_here = length(graph[:current_idxs][][1])
            update!(graph, current_idxs=[fill(curr_idx, n_walkers_here)])
        end
    end

    on(menu_upper.selection) do selected_recipe
        update!(graph, upper_recipe=selected_recipe)
    end
    on(menu_diagonal.selection) do selected_recipe
        update!(graph, diagonal_recipe=selected_recipe)
    end
    on(menu_lower.selection) do selected_recipe
        update!(graph, lower_recipe=selected_recipe)
    end

    on(toggle_upper.active) do is_live
        update!(graph, show_stats_upper=is_live)
    end

    on(toggle_diag.active) do is_live
        update!(graph, show_stats_diag=is_live)
    end

    on(toggle_lower.active) do is_live
        update!(graph, show_stats_lower=is_live)
    end

    if !isnothing(picker_info)
        (; N, N_max, initial_vsel, apply_vsel!) = picker_info
        _build_vsel_picker!(fig, button_row, graph, N, N_max, initial_vsel, apply_vsel!)
    end

    return fig
end

# Pure decision logic, kept separate from the widget wiring below so it can be
# tested directly without real mouse events (which CairoMakie can't produce).
# Checking cell (i,j) adds both i and j to the active set (a diagonal cell
# i==j just adds i); unchecking removes both. The active set -- not individual
# checkbox states -- is the source of truth, so removal is explicit rather
# than re-derived from sibling checkboxes that haven't been resynced yet
# (which would make e.g. unchecking a diagonal while some other still-checked
# pair references it snap the diagonal back on).
function _vsel_after_toggle(active_vars::Set{Integer}, i::Integer, j::Integer, is_on::Bool)
    return is_on ? union(active_vars, (i, j)) : setdiff(active_vars, (i, j))
end

# Whether cell (i,j) should display as checked given the active set: both
# endpoints must be selected (a diagonal cell i==j just needs i).
_checkbox_should_be_checked(active_vars::Set{Int}, i::Integer, j::Integer) = (i in active_vars) && (j in active_vars)

# Checkbox has no direct `visible` attribute (unlike Label/Box); it needs its
# underlying blockscene hidden instead.
_set_block_visible!(b::Checkbox, v::Bool) = (b.blockscene.visible[] = v)
_set_block_visible!(b::Union{Label,Box}, v::Bool) = (b.visible[] = v)

# Builds the "Adjust vsel" button (placed into button_row, its own dedicated
# fixed-height row so it never gets squeezed by ui_layout's growing content,
# and never collapses along with the panel it's meant to reveal) and its
# collapsible N x N variable picker panel (N = total model dimensionality, not
# N_max), placed in a column to the right of button_row/ui_layout, below the
# corner plot itself. The lower triangle including the diagonal (i >= j) is
# interactive -- cell (i,j) and (j,i) are the same 2D marginal, so only one
# needs a checkbox; a diagonal cell (i,i) selects variable i on its own. Only
# the strict upper triangle is decorative. Checking a cell on/off means
# "include/exclude these variable(s) in vsel"; the actual N_max x N_max
# corner-plot grid always shows every pairing among the currently selected
# variables, so there's no independent per-pair visibility beyond which
# variables are selected.
#
# Takes graph/N/N_max/initial_vsel/apply_vsel! directly rather than a
# BATVisualizer so it also works for the static bat_makie_plot path, which has
# no vis/content at all.
function _build_vsel_picker!(
    fig::Figure,
    button_row::GridLayout,
    graph::ComputeGraph,
    N::Integer,
    N_max::Integer,
    initial_vsel::AbstractVector{<:Integer},
    apply_vsel!::Function
)
    picker_layout = fig[2:3, 2] = GridLayout()
    colsize!(fig.layout, 2, Fixed(0)) # starts collapsed

    status_label = Label(fig, "", fontsize=12, color=:red)
    button_row[1, 2] = status_label

    initial_vsel_set = Set(initial_vsel)
    checkboxes = Dict{Tuple{Int,Int},Checkbox}()
    all_blocks = Union{Checkbox,Label,Box}[]
    updating_programmatically = Ref(false)

    for j in 1:N
        lbl = Label(fig, string(j), fontsize=12)
        picker_layout[1, j+1] = lbl
        push!(all_blocks, lbl)
    end
    for i in 1:N
        lbl = Label(fig, string(i), fontsize=12)
        picker_layout[i+1, 1] = lbl
        push!(all_blocks, lbl)
        for j in 1:N
            if i >= j
                cb = Checkbox(
                    picker_layout[i+1, j+1],
                    checked=(i in initial_vsel_set && j in initial_vsel_set),
                    roundness=0
                )
                checkboxes[(i, j)] = cb
                push!(all_blocks, cb)
            else
                bx = Box(fig, color=:gray70, width=20, height=20)
                picker_layout[i+1, j+1] = bx
                push!(all_blocks, bx)
            end
        end
    end

    active_vars = Ref(initial_vsel_set)

    for (i, j) in keys(checkboxes)
        cb = checkboxes[(i, j)]
        on(cb.checked) do is_on
            updating_programmatically[] && return

            new_vars = _vsel_after_toggle(active_vars[], i, j, is_on)

            if is_on && length(new_vars) > N_max
                updating_programmatically[] = true
                cb.checked[] = false
                updating_programmatically[] = false
                status_label.text[] = "Can't select more than $(N_max) variables at once -- deselect one first."
                return
            end

            status_label.text[] = ""
            active_vars[] = new_vars
            apply_vsel!(sort(collect(new_vars)))

            # Resync every checkbox to the new active set -- e.g. checking
            # (2,1) also checks (1,1) and (2,2) automatically, and checking
            # both (1,1) and (2,2) also checks (2,1) automatically.
            updating_programmatically[] = true
            for ((i2, j2), cb2) in checkboxes
                cb2.checked[] = _checkbox_should_be_checked(new_vars, i2, j2)
            end
            updating_programmatically[] = false
        end
    end

    adjust_button = Button(fig, label="Adjust vsel")
    button_row[1, 1] = adjust_button
    picker_visible = Observable(false)
    on(picker_visible) do is_visible
        colsize!(fig.layout, 2, is_visible ? Auto() : Fixed(0))
        for b in all_blocks
            _set_block_visible!(b, is_visible)
        end
    end
    for b in all_blocks
        _set_block_visible!(b, false) # start collapsed AND hidden
    end
    on(adjust_button.clicks) do _
        picker_visible[] = !picker_visible[]
    end

    return nothing
end

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

    with_theme(vis.backend.dark ? bat_theme_dark() : bat_theme()) do
        gridlayout = _init_gridlayout(graph, vis.backend.N_max)
        fig = _build_fig(graph, gridlayout, picker_info)
        display(fig)
    end

    # force=false: only flush once effective_batch_size[] samples are buffered
    # (the normal per-tick check). force=true: flush whatever is currently
    # buffered regardless of threshold, used once for the final drain below so
    # the last partial batch isn't silently dropped when is_live[] flips false.
    function flush_buffer!(; force::Bool=false)
        lock(buffer_lock)
        update_graph = force ? n_buffer_samples[] > 0 : n_buffer_samples[] >= effective_batch_size[]
        if update_graph
            # Shallow copy: output_buffer's slots get replaced (not mutated)
            # below, so the extracted inner vectors are never touched again --
            # no need to deep-copy the actual sample data out of them.
            extracted_output_buffer = copy(output_buffer)
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
        end
        unlock(buffer_lock)

        if update_graph
            fresh_batch_trafo = _transform_chain_outputs(f_pretransform, extracted_output_buffer)

            samples = graph[:samples][]
            samples_new = _append_chain_outputs(mcmc_states[1], samples, fresh_batch_trafo)

            update!(graph, samples=samples_new)

            # Derived from the actual merged length (not the raw batch length):
            # checked_push! inside _append_chain_outputs can collapse a sample at
            # the batch boundary into a weight increment instead of a new row.
            current_idxs_new = [length.(chain_samples) for chain_samples in samples_new]
            update!(graph, current_idxs=current_idxs_new)
        end
        return nothing
    end

    # Started only after the figure above has fully resolved its initial state --
    # the listener's first tick can mutate :idxs/:samples via _apply_vsel!, which
    # would otherwise race the still-in-progress initial construction (observed
    # in practice specifically on a cold/slow first JIT compile of the compute
    # graph's closures, where construction can take longer than poll_interval).
    listener_task[] = errormonitor(
        @async begin
            while is_live[]
                sleep(poll_interval)

                # Decoupled from the sample-batch flush below so a vsel change
                # (vis.backend.vsel mutated by a future UI widget) is picked up
                # promptly instead of waiting for the next full batch.
                _apply_vsel!(vis, vis.backend.vsel)

                flush_buffer!()
            end

            # is_live[] can flip false between two ticks (bat_sample_impl sets it
            # right before waiting on this task), which would otherwise drop
            # whatever's currently buffered -- below n_batch, or even a full
            # batch this tick hadn't gotten to yet -- leaving the display behind
            # the true final sample count. One last unconditional flush closes
            # that gap before the task returns.
            flush_buffer!(force=true)
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
    n_smpls_start = sum(length.(output_buffer[output_id]))
    get_samples!(output_buffer[output_id], chain_state, nonzero_weights)
    n_smpls_end = sum(length.(output_buffer[output_id]))

    lock(buffer_lock)
    n_new = n_smpls_end - n_smpls_start
    n_buffer_samples[] += n_new
    # Backpressure: block this chain's sampling task once the buffer has grown
    # far enough ahead of the listener that the display would otherwise lag
    # noticeably behind the true latest samples. Bounded, not per-sample --
    # sampling still runs in free bursts up to max_buffered before it has to
    # wait -- so the listener remains the sole bottleneck only when it's
    # genuinely falling behind, not on every single step.
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
    unlock(buffer_lock)
end


function _marginal_view_dist(
    locations::SubArray,
    weights::SubArray,
    filter::Bool,
    bins::Union{Tuple{Vararg{Int64}},Int64},
    closed::Symbol,
    normalization::Symbol
)
    if filter
        mask = _low_weight_mask(weights)
        locations = view(locations, :, mask)
        weights = view(weights, mask)
    end

    cols = Tuple(eachrow(locations))
    edges = if isa(bins, Integer)
        _get_edges(cols, (bins,), closed)
    elseif bins isa Tuple
        Tuple(_get_edges(cols[i], bins[i], closed) for i in 1:length(bins))
    else
        (_get_edges(cols, bins, closed),)
    end

    hist = fit(Histogram, cols, FrequencyWeights(weights), edges, closed=closed)
    h_norm = normalization == :none ? hist : StatsBase.normalize(hist, mode=normalization)
    return h_norm
end

# Mirrors BAT.drop_low_weight_samples, but returns a mask over a bare weight
# vector instead of indexing a DensitySampleVector (locations/weights are
# already split apart into separate views by the time they get here).
function _low_weight_mask(weights::AbstractVector, fraction::Real=10^-5, threshold::Real=10^-2)
    W = float(weights)
    if minimum(W) / maximum(W) > threshold
        return trues(length(W))
    end
    W_s = sort(W)
    Q = cumsum(W_s)
    Q ./= maximum(Q)
    ind = searchsortedlast(Q, fraction)
    ind == 0 && return trues(length(W))
    thresh = W_s[ind]
    return W .>= thresh
end

function _get_bin_centers(hist::Histogram)
    edges = hist.edges
    dims = ndims(hist.weights)

    centers = [[edges[d][i] + 0.5 * (edges[d][i+1] - edges[d][i]) for i in 1:length(edges[d])-1] for d in 1:dims]

    return centers
end


function warmup_makie_shaders()
    @info "Warming up Makie shaders"
    fig = Figure()
    ax = Axis(fig[1, 1])

    barplot!(ax, [0.0], [0.0])
    stairs!(ax, [0.0, 1.0], [0.0, 0.0])
    vlines!(ax, [0.0])
    hlines!(ax, [0.0])
    errorbars!(ax, [0.0], [0.0], [0.1])

    scatter!(ax, [0.0], [0.0])
    lines!(ax, [0.0, 1.0], [0.0, 1.0])
    linesegments!(ax, [0.0, 1.0], [0.0, 1.0])

    heatmap!(ax, [0.0, 1.0], [0.0, 1.0], [0.0 1.0; 1.0 0.0])
    contourf!(ax, [0.0, 1.0], [0.0, 1.0], [0.0 1.0; 1.0 0.0])
    hexbin!(ax, [0.0], [0.0]; bins=2)
    poly!(ax, Point2f[(0, 0), (1, 0), (0, 1)])

    Makie.colorbuffer(fig)
    return nothing
end
