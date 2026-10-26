# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# Measured label widths; update when labels or theme font size change.
const _LBL_ROW_WIDTH = 80.0        # max("Upper", "Diagonal", "Lower")
const _LBL_STATS_WIDTH = 116.0     # "Stats overlay"
const _LBL_TRACE_WIDTH = 122.0     # "Trace overlay"
const _LBL_MARGINALS_WIDTH = 182.0 # "Displayed Marginals"
const _LBL_MARGINALS_CAPPED_WIDTH = 292.0 # "Displayed Marginals (1-10 of NN)"
const _UI_COL2_MENU_WIDTH = 200.0  # fixed recipe-menu column width (colsize! below)
# Limit the picker to ten variables; further coordinates remain selectable through vsel.
const _PICKER_MAX_N = 10
# Size picker cells from N so the matrix fits the panel without reactive rescaling.
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

# Compute panel width explicitly: a GridLayout with deferred width prevents bottom-up sizing.
function _controls_panel_width(
    picker_info::Union{PickerInfo,Nothing}, ui_box_pad::Real; has_trace_col::Bool = true
)
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


# Fixed widths keep panel sizing independent of the grid's aspect ratio.
_fix_panel_size!(x, panel_width::Real) = (x.width[] = Fixed(panel_width); x.halign[] = :center)

function _default_recipe_label(options::Vector, current_recipe)
    idx = findfirst(x -> x[2] == current_recipe, options)
    isnothing(idx) && throw(ArgumentError(
        "Recipe $current_recipe is not a valid choice here (e.g. ChainScatter2D " *
        "requires samples with chain identity -- has_chain_info must be true) -- " *
        "check the `recipes` argument passed to bat_makie_plot/Makie.plot."
    ))
    return options[idx][1]
end


# Preserve the original construction/display error if best-effort cleanup fails.
function _empty_failed_static_figure!(fig)
    try
        empty!(fig)
    catch err
        BAT._visualizer_cleanup_diagnostic(:static_figure, (err, catch_backtrace()))
    end
    return nothing
end


function _build_fig(
    graph::ComputeGraph,
    gridlayout::Any,
    picker_info::Union{PickerInfo,Nothing} = nothing;
    has_chain_info::Bool = false,
    has_trace_info::Bool = false,
    live_vis = nothing,
)
    # Panel rows have independent fixed widths; the plot grid can still expand.
    n_grid = size(graph[:live_map][], 1)
    grid_growth = 170 * max(n_grid - 3, 0)
    ui_box_pad = to_value(Makie.theme(:fontsize)) / 3
    panel_width = _controls_panel_width(picker_info, ui_box_pad; has_trace_col = has_trace_info)
    # Leave enough figure width for the panel and its margin.
    fig_w = max(min(665 + grid_growth, 1600), ceil(Int, panel_width) + 24)
    fig = Figure(size = (fig_w, min(850 + grid_growth, 1785)))
    try
        return _populate_fig!(fig, graph, gridlayout, picker_info, ui_box_pad, panel_width;
            has_chain_info, has_trace_info, live_vis)
    catch
        # Live startup already has a lifecycle owner that disposes its view.
        isnothing(live_vis) && _empty_failed_static_figure!(fig)
        rethrow()
    end
end


function _populate_fig!(fig::Figure, graph::ComputeGraph, gridlayout,
    picker_info::Union{PickerInfo,Nothing}, ui_box_pad, panel_width;
    has_chain_info::Bool, has_trace_info::Bool, live_vis)
    subscription_owner = isnothing(live_vis) ? fig.scene : live_vis
    isnothing(live_vis) || _attach_live_view!(live_vis, fig)

    _bind_interactive_grid!(fig, gridlayout; live_vis = live_vis)

    colsize!(fig.layout, 1, Aspect(1, 1))
    # Auto assigns the plot grid the space left by the two control rows.
    rowsize!(fig.layout, 1, Auto())

    # Reserve fixed padding around the central content cell.
    toggle_row = fig[2, 1] = GridLayout(3, 3)
    rowgap!(toggle_row, 0)
    colgap!(toggle_row, 0)
    # A Box at its default width would fill the whole grid-derived column.
    toggle_box = Box(fig[2, 1],
        color = _panel_bg_color(fig.scene.backgroundcolor[]), cornerradius = 10, strokewidth = 0)
    _fix_panel_size!(toggle_box, panel_width)
    toggle_row_content = toggle_row[2, 2] = GridLayout()
    rowsize!(toggle_row, 1, Fixed(ui_box_pad))
    rowsize!(toggle_row, 3, Fixed(ui_box_pad))
    colsize!(toggle_row, 1, Fixed(ui_box_pad))
    colsize!(toggle_row, 3, Fixed(ui_box_pad))
    _fix_panel_size!(toggle_row, panel_width)
    rowsize!(fig.layout, 2, Auto())

    controls_layout = fig[3, 1] = GridLayout(3, 3)
    rowgap!(controls_layout, 0)
    colgap!(controls_layout, 0)
    _fix_panel_size!(controls_layout, panel_width)
    controls_box = Box(fig[3, 1],
        color = _panel_bg_color(fig.scene.backgroundcolor[]), cornerradius = 10, strokewidth = 0)
    _fix_panel_size!(controls_box, panel_width)
    rowsize!(controls_layout, 1, Fixed(ui_box_pad))
    rowsize!(controls_layout, 3, Fixed(ui_box_pad))
    colsize!(controls_layout, 1, Fixed(ui_box_pad))
    colsize!(controls_layout, 3, Fixed(ui_box_pad))
    # Set gaps after assigning content, once all three tracks exist.
    rowgap!(fig.layout, 2, ui_box_pad)

    # Share tracks between labels, menus and the variable picker.
    ui_layout = controls_layout[2, 2] = GridLayout()

    # A width of nothing fills the assigned cell; Auto would size to content.
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

    # Construct upward-opening menus bottom-first so an open list receives clicks before the
    # menus it covers. Equal-priority handlers run in registration order.
    menu_lower = Menu(
        fig,
        options = options2D,
        default = default_lower,
        direction = :up
    )
    menu_diagonal = Menu(
        fig,
        options = options1D,
        default = default_diag,
        direction = :up
    )
    menu_upper = Menu(
        fig,
        options = options2D,
        default = default_upper,
        direction = :up
    )

    # Keep interaction cheap: pending range changes never invalidate recipes.
    collapse_button = Button(fig, label = "☰", halign = :left, valign = :top)
    toggle_row_content[1, 1] = collapse_button
    colsize!(toggle_row_content, 1, Relative(1))
    range_controls = Ref{Any}(nothing)
    limit_label = Label(toggle_row_content[1, 2], "All")
    function add_index_slider!()
        limit = graph[:max_step][]
        limit_label.text[] = limit == 0 ? "No samples" : limit == 1 ? "All (single step/sample)" : ""
        limit > 1 || return nothing
        if isnothing(range_controls[])
            # A blank placeholder must not report zero as column 2's natural
            # width: the slider needs that column's remaining panel space.
            limit_label.tellwidth[] = false
            colsize!(toggle_row_content, 1, Auto())
            collapse_button.height[] = Relative(1)
            toggle_row_content[1:3, 1] = collapse_button
            title = Label(
                toggle_row_content[1, 2:3],
                graph[:has_step_timing][] ? "Step in cycle" : "Sample range (per walker)";
                tellwidth = false
            )
            slider = IntervalSlider(toggle_row_content[2, 2:3],
                range = SampleRangeGrid(limit), startvalues = (1, limit), width = nothing)
            draft = Observable((Int64(1), limit))
            syncing = Ref(false)
            value_entry = Textbox(toggle_row_content[3, 2:4]; stored_string = "1:$limit",
                placeholder = "start:end", width = nothing, tellwidth = false,
                validator = text -> !isnothing(_parse_sample_window(text, graph[:max_step][])))
            _view_on(subscription_owner, slider.interval) do interval
                syncing[] || (draft[] = Int64.(interval))
            end
            _view_on(subscription_owner, value_entry.stored_string) do text
                if !syncing[]
                    interval = _parse_sample_window(text, graph[:max_step][])
                    isnothing(interval) || (draft[] = interval)
                end
            end
            _view_on(subscription_owner, draft) do interval
                syncing[] = true
                try
                    set_close_to!(slider, interval...)
                    Makie.set!(value_entry, "$(interval[1]):$(interval[2])")
                finally
                    syncing[] = false
                end
            end
            apply = Button(toggle_row_content[1, 4], label = "Apply")
            all_button = Button(toggle_row_content[2, 4], label = "All")
            dirty = Observable(true)
            _view_onany(subscription_owner, graph[:window_steps], draft) do applied, pending
                dirty[] = applied != pending
                apply.label[] = dirty[] ? "Apply" : "Applied"
            end
            _view_on(subscription_owner, apply.clicks) do _
                dirty[] && _apply_sample_window!(graph, draft[]; live_vis = live_vis)
            end
            _view_on(subscription_owner, all_button.clicks) do _
                _apply_sample_window!(graph, nothing; live_vis = live_vis)
            end
            range_controls[] = (slider = slider, draft = draft, syncing = syncing, apply = apply,
                all = all_button, dirty = dirty, title = title, value_entry = value_entry)
            rowgap!(toggle_row_content, 1, 0)
        end
        # Growing data must not move the scale under a pressed pointer. Preserve
        # draft values as well as the applied range; All remains a separate state.
        if !(Mouse.left in events(fig.scene).mousebuttonstate)
            slider = range_controls[].slider
            if last(slider.range[]) != limit
                controls = range_controls[]
                controls.syncing[] = true
                try
                    slider.range[] = SampleRangeGrid(limit)
                    set_close_to!(slider, controls.draft[]...)
                finally
                    controls.syncing[] = false
                end
            end
        end
        return nothing
    end
    _view_on(subscription_owner, graph[:max_step]) do _
        add_index_slider!()
    end
    # Observe release before widgets consume it (native mouse handlers use 1).
    # Makie's button-state tracker has already removed the released button.
    _view_on(subscription_owner, events(fig.scene).mousebutton; priority = 2) do event
        event.button == Mouse.left && event.action == Mouse.release && add_index_slider!()
        return Consume(false)
    end
    add_index_slider!()

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
    toggle_upper = Toggle(ui_layout[2, 3], active = false)
    toggle_diag = Toggle(ui_layout[3, 3], active = false)
    toggle_lower = Toggle(ui_layout[4, 3], active = false)

    ui_blocks = Any[
        lbl_upper, lbl_diag, lbl_lower, lbl_recipe,
        menu_upper, menu_diagonal, menu_lower,
        lbl_stats, toggle_upper, toggle_diag, toggle_lower,
    ]

    # Trace controls require step timing and only apply to off-diagonal cells.
    if has_trace_info
        lbl_trace = Label(fig, "Trace overlay")
        ui_layout[1, 4] = lbl_trace
        toggle_trace_upper = Toggle(ui_layout[2, 4], active = false)
        toggle_trace_lower = Toggle(ui_layout[4, 4], active = false)
        append!(ui_blocks, Any[lbl_trace, toggle_trace_upper, toggle_trace_lower])
        _view_on(subscription_owner, toggle_trace_upper.active) do is_live
            _view_graph_update!(graph, live_vis; show_trace_upper = is_live)
        end
        _view_on(subscription_owner, toggle_trace_lower.active) do is_live
            _view_graph_update!(graph, live_vis; show_trace_lower = is_live)
        end
    end

    colsize!(ui_layout, 1, Auto())
    colsize!(ui_layout, 2, _UI_COL2_MENU_WIDTH)
    colsize!(ui_layout, 3, Auto())
    has_trace_info && colsize!(ui_layout, 4, Auto())

    rowsize!(controls_layout, 2, Auto())

    _view_on(subscription_owner, menu_upper.selection) do selected_recipe
        _view_graph_update!(graph, live_vis; upper_recipe = selected_recipe)
    end
    _view_on(subscription_owner, menu_diagonal.selection) do selected_recipe
        _view_graph_update!(graph, live_vis; diagonal_recipe = selected_recipe)
    end
    _view_on(subscription_owner, menu_lower.selection) do selected_recipe
        _view_graph_update!(graph, live_vis; lower_recipe = selected_recipe)
    end

    _view_on(subscription_owner, toggle_upper.active) do is_live
        _view_graph_update!(graph, live_vis; show_stats_upper = is_live)
    end

    _view_on(subscription_owner, toggle_diag.active) do is_live
        _view_graph_update!(graph, live_vis; show_stats_diag = is_live)
    end

    _view_on(subscription_owner, toggle_lower.active) do is_live
        _view_graph_update!(graph, live_vis; show_stats_lower = is_live)
    end

    if !isnothing(picker_info)
        (; N, N_max, initial_vsel, apply_vsel!) = picker_info
        picker_blocks = _build_vsel_picker!(
            fig, ui_layout, graph, N, N_max, initial_vsel, apply_vsel!,
            has_trace_info ? 5 : 4; live_vis = live_vis,
        )
        append!(ui_blocks, picker_blocks)
    end

    # Apply the initial collapsed state when registering the callback.
    controls_visible = Observable(false)
    _view_on(subscription_owner, controls_visible; update = true) do vis
        rowsize!(fig.layout, 3, vis ? Auto() : Fixed(0))
        # Hide the background explicitly; Fixed(0) alone can leave a rounding sliver.
        controls_box.visible[] = vis
        for b in ui_blocks
            _set_block_visible!(b, vis)
        end
    end
    _view_on(subscription_owner, collapse_button.clicks) do _
        controls_visible[] = !controls_visible[]
    end

    if !isnothing(live_vis)
        # Keep the observed-population label visible independently of the controls and selected
        # window.
        Label(fig[4, 1], "Observed sampling history\nMay include initialization and burn-in";
            tellwidth = false)
    end

    guarded_slider! = isnothing(live_vis) ? add_index_slider! :
        () -> _with_live_view(add_index_slider!, live_vis)
    return (fig = fig, add_index_slider! = guarded_slider!, range_controls = range_controls)
end


# Native IntervalSlider scans its range to locate values and uses floating-point
# index arithmetic while dragging. Bound that work independently of elapsed
# steps; exact integer endpoints remain editable in the accompanying textbox.
struct SampleRangeGrid <: AbstractVector{Int64}
    limit::Int64
end
Base.size(grid::SampleRangeGrid) = (Int(min(grid.limit, 10_001)),)
Base.IndexStyle(::Type{SampleRangeGrid}) = IndexLinear()
function Base.getindex(grid::SampleRangeGrid, i::Int)
    @boundscheck checkbounds(grid, i)
    n = length(grid)
    return n == 1 ? Int64(1) : Int64(1 + div(Int128(i - 1) * (grid.limit - 1), n - 1))
end


function _parse_sample_window(text, limit)
    text isa AbstractString || return nothing
    endpoints = split(text, ':')
    length(endpoints) == 2 || return nothing
    lo, hi = tryparse.(Int64, strip.(endpoints))
    (isnothing(lo) || isnothing(hi)) && return nothing
    return 1 <= lo <= hi <= limit ? (lo, hi) : nothing
end


# Derive selection from the active set rather than sibling checkbox state.
function _vsel_after_toggle(active_vars::Set{<:Integer}, i::Integer, j::Integer, is_on::Bool)
    return is_on ? union(active_vars, (i, j)) : setdiff(active_vars, (i, j))
end


_checkbox_should_be_checked(active_vars::Set{<:Integer}, i::Integer, j::Integer) =
    (i in active_vars) && (j in active_vars)

# A Fixed(0) cell alone does not disable geometric hit-testing: hidden blocks
# need zero width and height. Restore each block's original dimensions; using
# Auto() would shrink-wrap a Menu whose default width is nothing.
const _BLOCK_NATURAL_SIZE = WeakKeyDict{Any,Tuple{Any,Any}}()
_natural_size!(b) = get!(() -> (b.width[], b.height[]), _BLOCK_NATURAL_SIZE, b)

# Batch attribute writes without notification; the last write must notify to trigger relayout.
_silent_set!(observable, val) = setexcludinghandlers!(observe(observable), val)

function _set_block_visible!(b, v::Bool)
    _silent_set!(b.blockscene.visible, v)
    w, h = _natural_size!(b)
    _silent_set!(b.width, v ? w : 0)
    b.height[] = v ? h : 0 # final, notifying write
    return nothing
end
_set_block_visible!(b::Union{Label,Box}, v::Bool) = (b.visible[] = v)

# Checkbox hit testing uses size independently of its bounding box. Capture the widget weakly
# when hiding/restoring both.
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


# Build the capped variable picker with fixed square cells. Only the lower triangle is
# interactive because mirrored cells select the same marginal.
function _build_vsel_picker!(
    fig::Figure,
    ui_layout::GridLayout,
    graph::ComputeGraph,
    N::Integer,
    N_max::Integer,
    initial_vsel::AbstractVector{<:Integer},
    apply_vsel!::Function,
    picker_col::Integer;
    live_vis = nothing,
)
    subscription_owner = isnothing(live_vis) ? fig.scene : live_vis
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

    # Choose a status color legible against the current background.
    status_label = Label(fig, "", fontsize = 12, color = _status_text_color(fig.scene.backgroundcolor[]))

    initial_vsel_set = Set(_clamp_vsel(initial_vsel, N, N_max))
    checkboxes = Dict{Tuple{Int,Int},Checkbox}()
    all_blocks = Union{Checkbox,Label,Box}[status_label, lbl_marginals]
    updating_programmatically = Ref(false)

    for j in 1:N_shown
        lbl = Label(fig, string(j), fontsize = lbl_fontsize)
        picker_layout[1, j+1] = lbl
        push!(all_blocks, lbl)
    end
    for i in 1:N_shown
        lbl = Label(fig, string(i), fontsize = lbl_fontsize)
        picker_layout[i+1, 1] = lbl
        push!(all_blocks, lbl)
        for j in 1:N_shown
            if i >= j
                cb = Checkbox(
                    picker_layout[i+1, j+1],
                    checked = (i in initial_vsel_set && j in initial_vsel_set),
                    roundness = 0,
                    size = cb_size
                )
                checkboxes[(i, j)] = cb
                push!(all_blocks, cb)
            else
                bx = Box(fig,
                    color = _panel_bg_color(_panel_bg_color(fig.scene.backgroundcolor[])),
                    width = cb_size, height = cb_size)
                picker_layout[i+1, j+1] = bx
                push!(all_blocks, bx)
            end
        end
    end
    picker_layout[N_shown+2, 1:(N_shown+1)] = status_label

    # Set fixed track sizes after content has created the tracks; later layout growth must not
    # stretch the square cells.
    rowgap!(picker_layout, gap)
    colgap!(picker_layout, gap)
    for k in 1:N_shown+1
        colsize!(picker_layout, k, Fixed(cell))
        rowsize!(picker_layout, k, Fixed(cell))
    end

    # Auto tracks ignore spanning content. Enlarge all three rows equally when the picker
    # exceeds their natural height.
    status_h = 16.0
    matrix_h = (N_shown + 1) * cell + (N_shown + 1) * gap + status_h
    row_h = max(_UI_CONTROL_ROW_HEIGHT, (matrix_h - 2 * _UI_LAYOUT_ROW_GAP) / 3)
    for r in 2:4
        rowsize!(ui_layout, r, Fixed(row_h))
    end

    active_vars = Ref(initial_vsel_set)

    for (i, j) in keys(checkboxes)
        cb = checkboxes[(i, j)]
        _view_on(subscription_owner, cb.checked) do is_on
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

            # Resync all checkboxes: deselecting a pair also deselects its diagonal variables.
            updating_programmatically[] = true
            for ((i2, j2), cb2) in checkboxes
                cb2.checked[] = _checkbox_should_be_checked(new_vars, i2, j2)
            end
            updating_programmatically[] = false
        end
    end

    return all_blocks
end


# Programmatic and widget changes use the same single graph transaction.
function _apply_sample_window!(graph, window; live_vis = nothing)
    if !isnothing(window)
        lo, hi = window
        1 <= lo <= hi || throw(ArgumentError("Require 1 <= window start <= window end"))
    end
    # Live intents may replace an earlier deferred value even when this value
    # equals the currently displayed one (Apply followed by All, for example).
    isnothing(live_vis) && graph[:window_steps][] == window && return false
    _view_graph_update!(graph, live_vis; window_steps = window)
    return true
end


# Static controls preserve direct graph updates; live controls use the view owner.
_view_graph_update!(graph, ::Nothing; kwargs...) = update!(graph; kwargs...)
_view_graph_update!(graph, vis; kwargs...) = _live_graph_update!(vis; kwargs...)
