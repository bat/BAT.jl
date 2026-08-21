# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Empirically measured natural label widths at the design theme/fontsize --
# re-measure if any label text or the ambient fontsize changes.
const _LBL_ROW_WIDTH = 80.0        # max("Upper", "Diagonal", "Lower")
const _LBL_STATS_WIDTH = 116.0     # "Stats overlay"
const _LBL_TRACE_WIDTH = 122.0     # "Trace overlay"
const _LBL_MARGINALS_WIDTH = 182.0 # "Displayed Marginals"
const _LBL_MARGINALS_CAPPED_WIDTH = 292.0 # "Displayed Marginals (1-10 of NN)"
const _UI_COL2_MENU_WIDTH = 200.0  # fixed recipe-menu column width (colsize! below)
# The vsel picker shows at most this many variables (matrix cells scale down
# with N, but past 10 the matrix stops being usable UI; further dims stay
# selectable via the vsel argument).
const _PICKER_MAX_N = 10
# Deterministic picker geometry, purely N-derived (no reactive rescaling):
# cells shrink and pack tighter as N grows so the matrix fits the panel for
# every N up to _PICKER_MAX_N.
_picker_cell_size(N::Integer) = clamp(176.0 / (N + 1), 14.0, 24.0)
_picker_gap(N::Integer) = N <= 5 ? 5.0 : 3.0
_picker_fontsize(N::Integer) = clamp(0.55 * _picker_cell_size(N), 7.0, 12.0)
# Matrix width incl. the index-label column and the N inter-column gaps.
_picker_matrix_width(N::Integer) = (N + 1) * _picker_cell_size(N) + N * _picker_gap(N)
# GridLayoutBase's default inter-column/row gaps, applied between ui_layout's tracks.
const _UI_LAYOUT_COL_GAP = 16.0
const _UI_LAYOUT_ROW_GAP = 16.0
# Natural height of a menu/toggle control row (measured; re-measure if the
# theme fontsize or Menu styling changes).
const _UI_CONTROL_ROW_HEIGHT = 36.0

# Panel width computed by plain arithmetic, NOT GridLayoutBase bottom-up sizing: a nothing-deferring GridLayout anywhere in the chain breaks determinability.
function _controls_panel_width(picker_info::Union{PickerInfo,Nothing}, ui_box_pad::Real; has_trace_col::Bool = true)
    fixed_cols = _LBL_ROW_WIDTH + _UI_COL2_MENU_WIDTH + _LBL_STATS_WIDTH +
                 (has_trace_col ? _LBL_TRACE_WIDTH : 0.0)
    picker_col_width = if isnothing(picker_info)
        0.0
    else
        # The capped "(1-10 of N)" title variant is wider than the matrix.
        title_w = picker_info.N > _PICKER_MAX_N ? _LBL_MARGINALS_CAPPED_WIDTH : _LBL_MARGINALS_WIDTH
        max(title_w, _picker_matrix_width(min(picker_info.N, _PICKER_MAX_N)))
    end
    n_cols = (isnothing(picker_info) ? 3 : 4) + (has_trace_col ? 1 : 0)
    return 2 * ui_box_pad + fixed_cols + picker_col_width + (n_cols - 1) * _UI_LAYOUT_COL_GAP
end

# Fixed width (not Auto()) so a panel piece never depends on the grid's Aspect-driven column; halign=:center keeps it centered under the grid.
_fix_panel_size!(x, panel_width::Real) = (x.width[] = Fixed(panel_width); x.halign[] = :center)

# Looks up `current_recipe`'s dropdown label; raises a clear ArgumentError instead of an opaque "invalid index: nothing" if it isn't one of `options`.
function _default_recipe_label(options::Vector, current_recipe)
    idx = findfirst(x -> x[2] == current_recipe, options)
    isnothing(idx) && throw(ArgumentError(
        "Recipe $current_recipe is not a valid choice here (e.g. ChainScatter2D " *
        "requires samples with chain identity -- has_chain_info must be true) -- " *
        "check the `recipes` argument passed to bat_makie_plot/Makie.plot."
    ))
    return options[idx][1]
end

function _build_fig(
    graph::ComputeGraph,
    gridlayout::Any,
    picker_info::Union{PickerInfo,Nothing} = nothing;
    has_chain_info::Bool = false,
    has_trace_info::Bool = false,
)
    # Panel rows get their own Fixed widths (_fix_panel_size!) so the grid's Aspect/Auto column is never capped by the panel's width need.
    n_grid = size(graph[:live_map][], 1)
    grid_growth = 170 * max(n_grid - 3, 0)
    ui_box_pad = to_value(Makie.theme(:fontsize)) / 3
    panel_width = _controls_panel_width(picker_info, ui_box_pad; has_trace_col=has_trace_info)
    # The figure must be at least as wide as the fixed-width panel (plus a
    # small margin), or a wide picker matrix gets clipped at the figure edge.
    fig_w = max(min(665 + grid_growth, 1600), ceil(Int, panel_width) + 24)
    fig = Figure(size=(fig_w, min(850 + grid_growth, 1785)))

    plot(fig[1, 1], gridlayout)

    colsize!(fig.layout, 1, Aspect(1, 1))
    # Auto(), not Relative: the axis-only grid reports no natural size, so this row gets the remainder after rows 2/3 claim their content heights.
    rowsize!(fig.layout, 1, Auto())

    # toggle_row: a 3x3 wrapper of Fixed(ui_box_pad) margins around one content cell, with a Box at fig[2,1] as a rounded background; explicit dims are required for rowsize!/colsize! on row/col 3.
    toggle_row = fig[2, 1] = GridLayout(3, 3)
    rowgap!(toggle_row, 0)
    colgap!(toggle_row, 0)
    # A Box at its default width would fill the whole grid-derived column.
    toggle_box = Box(fig[2, 1], color=_panel_bg_color(fig.scene.backgroundcolor[]), cornerradius=10, strokewidth=0)
    _fix_panel_size!(toggle_box, panel_width)
    toggle_row_content = toggle_row[2, 2] = GridLayout()
    rowsize!(toggle_row, 1, Fixed(ui_box_pad))
    rowsize!(toggle_row, 3, Fixed(ui_box_pad))
    colsize!(toggle_row, 1, Fixed(ui_box_pad))
    colsize!(toggle_row, 3, Fixed(ui_box_pad))
    _fix_panel_size!(toggle_row, panel_width)
    rowsize!(fig.layout, 2, Auto())

    # Collapsible block holding everything else; same panel treatment and panel_width as toggle_row, so both bars match by construction.
    controls_layout = fig[3, 1] = GridLayout(3, 3)
    rowgap!(controls_layout, 0)
    colgap!(controls_layout, 0)
    _fix_panel_size!(controls_layout, panel_width)
    controls_box = Box(fig[3, 1], color=_panel_bg_color(fig.scene.backgroundcolor[]), cornerradius=10, strokewidth=0)
    _fix_panel_size!(controls_box, panel_width)
    rowsize!(controls_layout, 1, Fixed(ui_box_pad))
    rowsize!(controls_layout, 3, Fixed(ui_box_pad))
    colsize!(controls_layout, 1, Fixed(ui_box_pad))
    colsize!(controls_layout, 3, Fixed(ui_box_pad))
    # Only valid after the fig[3,1] assignment above -- rowgap!(_, 2, _) throws until GridLayoutBase has grown fig.layout to 3 rows.
    rowgap!(fig.layout, 2, ui_box_pad)

    # One GridLayout for labels/menus (cols 1-4) AND the vsel picker (col 5), so they share grid rows and align pixel-exactly by construction.
    ui_layout = controls_layout[2, 2] = GridLayout()

    # The default Auto() width means content-sized, NOT fill-the-cell; width = nothing is what defers to the assigned cell's width.
    for gl in (toggle_row_content, ui_layout)
        gl.width[] = nothing
    end

    options2D = [
        ("QuantileHist", QuantileHist2D),
        ("Hist", Hist2D),
        ("Scatter", Scatter2D),
        ("Hexbin", Hexbin2D),
        ("QuantileKDE", QuantileKDE2D),
        ("KDE", KDE2D),
        ("Errorbars", Errorbars2D),
    ]
    has_chain_info && push!(options2D, ("Scatter (by chain)", ChainScatter2D))
    options1D = [
        ("QuantileHist", QuantileHist1D),
        ("Hist", Hist1D),
        ("KDE", KDE1D),
        ("QuantileKDE", QuantileKDE1D),
        # "Normal fit", not "PDF": it overlays a fitted Gaussian, not the density.
        ("Normal fit", PDF1D),
        ("Errorbars", Errorbars1D),
    ]

    default_upper = _default_recipe_label(options2D, graph[:upper_recipe][])
    default_diag = _default_recipe_label(options1D, graph[:diagonal_recipe][])
    default_lower = _default_recipe_label(options2D, graph[:lower_recipe][])

    # Constructed BOTTOM-FIRST with direction pinned :up -- click dispatch, not
    # looks: all Menus share one event-handler priority (registration order), and
    # bottom-first + :up guarantees an open list (which only covers menus above it) is checked first.
    menu_lower = Menu(
        fig,
        options=options2D,
        default=default_lower,
        direction=:up
    )
    menu_diagonal = Menu(
        fig,
        options=options1D,
        default=default_diag,
        direction=:up
    )
    menu_upper = Menu(
        fig,
        options=options2D,
        default=default_upper,
        direction=:up
    )

    # Samples must already exist (an empty slider range crashes): true at build time for the static path, never for the live path, which retrofits post-run.
    show_slider = graph[:max_step][] > 0

    # Placed assuming no slider: the button's column claims the full row width so halign=:left is flush left -- add_index_slider! undoes both.
    collapse_button = Button(fig, label="☰", halign=:left, valign=:top)
    toggle_row_content[1, 1] = collapse_button
    colsize!(toggle_row_content, 1, Relative(1))

    # Builds the "Step Range" slider row into toggle_row -- immediately on the static path, or retrofitted post-run on the live path. Idempotent.
    slider_added = Ref(false)
    function add_index_slider!()
        slider_added[] && return nothing
        slider_added[] = true

        # Undo the no-slider button layout; height=Relative(1) makes the button fill its two-row span instead of centering its natural height in it.
        colsize!(toggle_row_content, 1, Auto())
        collapse_button.height[] = Relative(1)
        toggle_row_content[1:2, 1] = collapse_button

        # Spans cols 2:3 so its centered position doesn't shift with the value display's text width.
        lbl_idx_title = Label(toggle_row_content[1, 2:3], "Step Range")
        # Real MCMC steps, not stored rows (repetition weighting makes steps exceed rows); 0-based since step 0 is the initial position.
        slider_curr_idx = IntervalSlider(toggle_row_content[2, 2], range=0:graph[:max_step][], startvalues=(0, graph[:max_step][]))
        # Pinned outward (:top/:bottom) so the label's top and the slider's bottom land exactly on the button's edges.
        rowgap!(toggle_row_content, 1, 0)
        lbl_idx_title.valign[] = :top
        slider_curr_idx.valign[] = :bottom
        # width = nothing so the slider fills column 2 (Auto() would size the column to the slider's own natural width instead).
        slider_curr_idx.width[] = nothing
        # tellwidth=false, NOT width=nothing: Auto width would shrink column 2, while width=nothing would left-align instead of centering the text.
        lbl_idx_title.tellwidth[] = false
        lbl_idx_value = Label(toggle_row_content[2, 3], lift(iv -> "$(iv[1]):$(iv[2])", slider_curr_idx.interval))

        # Applied per walker inside :flat_samples (_step_window_rows); the slider must not write :current_idxs, which registration/flush owns.
        on(slider_curr_idx.interval) do (start_step, end_step)
            update!(graph, window_steps=(Int(start_step), Int(end_step)))
        end
        return nothing
    end
    show_slider && add_index_slider!()

    lbl_upper = Label(fig, "Upper")
    ui_layout[2, 1] = lbl_upper
    lbl_diag = Label(fig, "Diagonal")
    ui_layout[3, 1] = lbl_diag
    lbl_lower = Label(fig, "Lower")
    ui_layout[4, 1] = lbl_lower

    lbl_recipe = Label(fig, "Recipe")
    ui_layout[1, 2] = lbl_recipe
    ui_layout[2, 2] = menu_upper
    ui_layout[3, 2] = menu_diagonal
    ui_layout[4, 2] = menu_lower

    lbl_stats = Label(fig, "Stats overlay")
    ui_layout[1, 3] = lbl_stats
    toggle_upper = Toggle(ui_layout[2, 3], active=false)
    toggle_diag = Toggle(ui_layout[3, 3], active=false)
    toggle_lower = Toggle(ui_layout[4, 3], active=false)

    ui_blocks = Any[
        lbl_upper, lbl_diag, lbl_lower, lbl_recipe,
        menu_upper, menu_diagonal, menu_lower,
        lbl_stats, toggle_upper, toggle_diag, toggle_lower,
    ]

    # Only for sources that can drive Trace2D (chainid AND stepno); Row 3 (Diagonal) stays unassigned -- a trace is inherently 2D.
    if has_trace_info
        lbl_trace = Label(fig, "Trace overlay")
        ui_layout[1, 4] = lbl_trace
        toggle_trace_upper = Toggle(ui_layout[2, 4], active=false)
        toggle_trace_lower = Toggle(ui_layout[4, 4], active=false)
        append!(ui_blocks, Any[lbl_trace, toggle_trace_upper, toggle_trace_lower])
        on(toggle_trace_upper.active) do is_live
            update!(graph, show_trace_upper=is_live)
        end
        on(toggle_trace_lower.active) do is_live
            update!(graph, show_trace_lower=is_live)
        end
    end

    colsize!(ui_layout, 1, Auto())
    # Same constant the width formula uses -- a bare literal could drift.
    colsize!(ui_layout, 2, _UI_COL2_MENU_WIDTH)
    colsize!(ui_layout, 3, Auto())
    has_trace_info && colsize!(ui_layout, 4, Auto())

    rowsize!(controls_layout, 2, Auto())

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
        picker_blocks = _build_vsel_picker!(
            fig, ui_layout, graph, N, N_max, initial_vsel, apply_vsel!,
            has_trace_info ? 5 : 4,
        )
        append!(ui_blocks, picker_blocks)
    end

    # Starts collapsed; update=true applies that at construction, since `on` only fires on future notifications.
    controls_visible = Observable(false)
    on(controls_visible; update=true) do vis
        rowsize!(fig.layout, 3, vis ? Auto() : Fixed(0))
        # Not in ui_blocks, so hidden directly: row 3's Fixed(0) alone leaves a ~1px rounding sliver of the Box visible.
        controls_box.visible[] = vis
        for b in ui_blocks
            _set_block_visible!(b, vis)
        end
    end
    on(collapse_button.clicks) do _
        controls_visible[] = !controls_visible[]
    end

    # Exposed so the live path can retrofit the index slider post-run.
    return (fig=fig, add_index_slider! = add_index_slider!)
end

# Pure decision logic, kept separate from widget wiring for direct testing. The active set, not sibling checkbox states, is the source of truth.
function _vsel_after_toggle(active_vars::Set{<:Integer}, i::Integer, j::Integer, is_on::Bool)
    return is_on ? union(active_vars, (i, j)) : setdiff(active_vars, (i, j))
end

# Both endpoints must be selected (a diagonal cell i==j just needs i).
_checkbox_should_be_checked(active_vars::Set{<:Integer}, i::Integer, j::Integer) = (i in active_vars) && (j in active_vars)

# Click hit-tests are purely geometric against the block's own bbox, which a Fixed(0) cell only *suggests*, so width/height must be forced to literal 0 to make a hidden block unclickable; restore must use the remembered per-block original, not Auto() (Menu defaults to width === nothing, which Auto() would shrink-wrap).
const _BLOCK_NATURAL_SIZE = WeakKeyDict{Any,Tuple{Any,Any}}()
_natural_size!(b) = get!(() -> (b.width[], b.height[]), _BLOCK_NATURAL_SIZE, b)

# Sets an Observable without notifying -- collapses one widget's attribute writes into a single relayout; each widget's last write must still notify.
_silent_set!(observable, val) = setexcludinghandlers!(observe(observable), val)

function _set_block_visible!(b, v::Bool)
    _silent_set!(b.blockscene.visible, v)
    w, h = _natural_size!(b)
    _silent_set!(b.width, v ? w : 0)
    b.height[] = v ? h : 0 # final, notifying write
    return nothing
end
_set_block_visible!(b::Union{Label,Box}, v::Bool) = (b.visible[] = v)

# Checkbox's clickable area is a fixed-size square (`size`), independent of its bbox, so `size` must be zeroed/restored too (captured weakly, for GC).
const _CHECKBOX_NATURAL_SIZE = WeakKeyDict{Checkbox,Float64}()
function _set_block_visible!(b::Checkbox, v::Bool)
    _silent_set!(b.blockscene.visible, v)
    w, h = _natural_size!(b)
    _silent_set!(b.width, v ? w : 0)
    _silent_set!(b.height, v ? h : 0)
    if v
        b.size[] = get(_CHECKBOX_NATURAL_SIZE, b, b.size[]) # final, notifying write
    else
        haskey(_CHECKBOX_NATURAL_SIZE, b) || (_CHECKBOX_NATURAL_SIZE[b] = b.size[])
        b.size[] = 0 # final, notifying write
    end
    return nothing
end

# Builds the "Displayed Marginals" title + picker matrix over the first
# min(N, _PICKER_MAX_N) variables; only the lower triangle incl. diagonal
# (i >= j) is interactive, since (i,j)/(j,i) are the same marginal. All cell/
# gap/font sizes are fixed, N-derived values (_picker_cell_size and friends)
# set once at construction -- deterministic square geometry at every N, and
# exactly what _controls_panel_width budgets for.
function _build_vsel_picker!(
    fig::Figure,
    ui_layout::GridLayout,
    graph::ComputeGraph,
    N::Integer,
    N_max::Integer,
    initial_vsel::AbstractVector{<:Integer},
    apply_vsel!::Function,
    picker_col::Integer,
)
    N_shown = min(N, _PICKER_MAX_N)
    cell = _picker_cell_size(N_shown)
    gap = _picker_gap(N_shown)
    lbl_fontsize = _picker_fontsize(N_shown)
    cb_size = round(0.72 * cell)

    title = N > _PICKER_MAX_N ?
        "Displayed Marginals (1-$(_PICKER_MAX_N) of $N)" : "Displayed Marginals"
    lbl_marginals = Label(fig, title)
    ui_layout[1, picker_col] = lbl_marginals

    picker_layout = ui_layout[2:4, picker_col] = GridLayout()

    # Color derived from the actual background (see _status_text_color) -- a hardcoded :red is illegible on the dark theme.
    status_label = Label(fig, "", fontsize=12, color=_status_text_color(fig.scene.backgroundcolor[]))

    initial_vsel_set = Set(initial_vsel)
    checkboxes = Dict{Tuple{Int,Int},Checkbox}()
    all_blocks = Union{Checkbox,Label,Box}[status_label, lbl_marginals]
    updating_programmatically = Ref(false)

    for j in 1:N_shown
        lbl = Label(fig, string(j), fontsize=lbl_fontsize)
        picker_layout[1, j+1] = lbl
        push!(all_blocks, lbl)
    end
    for i in 1:N_shown
        lbl = Label(fig, string(i), fontsize=lbl_fontsize)
        picker_layout[i+1, 1] = lbl
        push!(all_blocks, lbl)
        for j in 1:N_shown
            if i >= j
                cb = Checkbox(
                    picker_layout[i+1, j+1],
                    checked=(i in initial_vsel_set && j in initial_vsel_set),
                    roundness=0,
                    size=cb_size
                )
                checkboxes[(i, j)] = cb
                push!(all_blocks, cb)
            else
                # Widget shade from the actual background (two ladder steps), not a hardcoded gray that clashes with the dark theme.
                bx = Box(fig, color=_panel_bg_color(_panel_bg_color(fig.scene.backgroundcolor[])), width=cb_size, height=cb_size)
                picker_layout[i+1, j+1] = bx
                push!(all_blocks, bx)
            end
        end
    end
    picker_layout[N_shown+2, 1:(N_shown+1)] = status_label

    # Fixed square tracks and tight gaps, set only now that the content has
    # created the tracks (gap/size calls on a still-empty GridLayout don't
    # extend to tracks added later): cells stay square no matter how the
    # surrounding rows/columns stretch.
    rowgap!(picker_layout, gap)
    colgap!(picker_layout, gap)
    for k in 1:N_shown+1
        colsize!(picker_layout, k, Fixed(cell))
        rowsize!(picker_layout, k, Fixed(cell))
    end

    # ui_layout's rows 2:4 do NOT auto-grow for the row-spanning matrix
    # (GridLayoutBase sizes Auto tracks from single-span content only, so an
    # oversized span silently overflows into the neighboring rows) -- give
    # all three rows explicit equal heights whenever the matrix needs more
    # than the natural control-row height.
    status_h = 16.0
    matrix_h = (N_shown + 1) * cell + (N_shown + 1) * gap + status_h
    row_h = max(_UI_CONTROL_ROW_HEIGHT, (matrix_h - 2 * _UI_LAYOUT_ROW_GAP) / 3)
    for r in 2:4
        rowsize!(ui_layout, r, Fixed(row_h))
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

            # Resync every checkbox as a total function of the new active set (unchecking an off-diagonal deselects a still-referenced diagonal by design).
            updating_programmatically[] = true
            for ((i2, j2), cb2) in checkboxes
                cb2.checked[] = _checkbox_should_be_checked(new_vars, i2, j2)
            end
            updating_programmatically[] = false
        end
    end

    return all_blocks
end
