# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# Stable native layout: reserve a minimum figure size instead of clipping or scrolling.
const _LBL_ROW_WIDTH = 28.0
const _LBL_STATS_WIDTH = 98.0
const _LBL_TRACE_WIDTH = 92.0
const _UI_COL2_MENU_WIDTH = 180.0
const _UI_PANEL_PADDING = 12.0
const _UI_DIVIDER_WIDTH = 1.0
const _UI_LAYOUT_COL_GAP = 12.0
const _UI_LAYOUT_ROW_GAP = 12.0
const _LIVE_DRAG_INTERVAL_NS = UInt64(100_000_000)

function _picker_body_size(N::Integer)
    slots = _marginal_slot_count(N)
    return Float64(max(N > 5 ? 220 : 156, (slots + 1) * max(26, 7 * ndigits(N) + 10) + slots * 4))
end

function _controls_panel_dimensions(picker_info::Union{PickerInfo,Nothing}, padding::Real;
    has_trace_col::Bool = true)
    settings = _LBL_ROW_WIDTH + _UI_COL2_MENU_WIDTH + _LBL_STATS_WIDTH +
        (has_trace_col ? _LBL_TRACE_WIDTH : 0)
    columns = 3 + has_trace_col + 2 * !isnothing(picker_info)
    picker = isnothing(picker_info) ? 0 : _picker_body_size(picker_info.N)
    divider = isnothing(picker_info) ? 0 : _UI_DIVIDER_WIDTH
    natural_width = 2padding + settings + picker + divider + (columns - 1) * _UI_LAYOUT_COL_GAP
    width = max(620.0, natural_width)
    # Give spare width to the menus, preserving exact outer/divider padding.
    return (; width, menu_width = _UI_COL2_MENU_WIDTH + width - natural_width)
end

mutable struct _FigureControls
    range::Base.RefValue{Any}
    picker::Any
    expanded::Observable{Bool}
    collapse::Button
    notice::Any
    minimum_size::Tuple{Int,Int}
end

# Weak keys AND values: the figure's callbacks own its controls, never this test/debug lookup.
const _FIGURE_CONTROLS = WeakKeyDict{Scene,WeakRef}()
_figure_controls(fig::Figure) = get(_FIGURE_CONTROLS, fig.scene, WeakRef(nothing)).value

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
    graph::ComputeGraph, gridlayout::Any, picker_info::Union{PickerInfo,Nothing} = nothing;
    has_chain_info::Bool = false,
    has_trace_info::Bool = false,
    live_vis = nothing,
    message_duration::Real = 3.0,
)
    # Panel rows have independent fixed widths; the plot grid can still expand.
    n_grid = size(graph[:live_map][], 1)
    grid_growth = 170 * max(n_grid - 3, 0)
    ui_box_pad = _UI_PANEL_PADDING
    dimensions = _controls_panel_dimensions(picker_info, ui_box_pad; has_trace_col = has_trace_info)
    panel_width = dimensions.width
    fig_w = max(min(665 + grid_growth, 1600), ceil(Int, panel_width) + 40)
    fig = Figure(size = (fig_w, min(850 + grid_growth, 1785)))
    try
        return _populate_fig!(fig, graph, gridlayout, picker_info, ui_box_pad, panel_width;
            has_chain_info, has_trace_info, live_vis, message_duration, menu_width = dimensions.menu_width)
    catch
        # Live startup already has a lifecycle owner that disposes its view.
        isnothing(live_vis) && _empty_failed_static_figure!(fig)
        rethrow()
    end
end


function _populate_fig!(fig::Figure, graph::ComputeGraph, gridlayout,
    picker_info::Union{PickerInfo,Nothing}, ui_box_pad, panel_width;
    has_chain_info::Bool, has_trace_info::Bool, live_vis, message_duration, menu_width)
    subscription_owner = isnothing(live_vis) ? fig.scene : live_vis
    isnothing(live_vis) || _attach_live_view!(live_vis, fig)

    _bind_interactive_grid!(fig, gridlayout; live_vis = live_vis)

    colsize!(fig.layout, 1, Aspect(1, 1))
    # Auto assigns the plot grid the space left by the two control rows.
    rowsize!(fig.layout, 1, Auto())

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
        # Label the fitted Gaussian separately from the posterior density.
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

    # Manual range selection remains a draft until explicitly applied.
    collapse_button = Button(fig, label = "☰", width = 40, halign = :left, valign = :top)
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
            toggle_row_content[1:2, 1] = collapse_button
            summary_layout = toggle_row_content[1, 2] = GridLayout(; width = Auto(), halign = :center)
            colgap!(summary_layout, 8)
            title = Label(summary_layout[1, 1],
                graph[:has_step_timing][] ? "Step in cycle:" : "Sample range per walker:")
            slider = IntervalSlider(toggle_row_content[2, 2],
                range = SampleRangeGrid(limit), startvalues = (1, limit), width = nothing)
            draft = Observable((Int64(1), limit))
            syncing = Ref(false)
            text_editing = Ref(false)
            touched = Ref(false)
            value_control = _control_textbox(fig, summary_layout[1, 2]; owner = subscription_owner,
                limit = 48, text = "1:$limit", width = 126, height = 30, fontsize = 18,
                textpadding = (6, 6, 3, 3),
                on_submit = () -> commit_range_text!(), on_cancel = () -> begin
                    value_control.replace!("$(draft[][1]):$(draft[][2])")
                    value_control.blur!()
                end,
                validator = text -> !isnothing(_parse_sample_window(text, graph[:max_step][])))
            value_entry = value_control.box
            # Resolve the native theme through the existing button, preserving
            # alpha. The border must have one concrete color type from creation:
            # Makie's cached Poly computation cannot switch from Symbol to RGB.
            apply_stroke = RGBAf(Makie.to_color(collapse_button.strokecolor[]))
            apply = Button(toggle_row_content[1, 3], label = "Apply range", height = 30,
                width = 132, strokecolor = apply_stroke)
            apply_color = apply.buttoncolor[]
            live_layout = toggle_row_content[2, 3] = GridLayout()
            colgap!(live_layout, 6)
            live_drag_toggle = Toggle(live_layout[1, 1]; active = false, length = 27, markersize = 13)
            live_drag = live_drag_toggle.active
            live_drag_button = Button(live_layout[1, 2], label = "Live drag", fontsize = 18,
                height = 30, padding = (0, 0, 0, 0), buttoncolor = :transparent, strokewidth = 0)
            dirty = Observable(false)
            function commit_range_text!()
                interval = _parse_sample_window(_control_text(value_entry), graph[:max_step][])
                if isnothing(interval)
                    value_entry.bordercolor[] = _status_text_color(fig.scene.backgroundcolor[])
                    return false
                end
                value_entry.bordercolor[] = _panel_bg_color(fig.scene.backgroundcolor[], 0.3)
                touched[] = true
                draft[] = interval
                return true
            end
            pending = Ref(false)
            applying = Ref(false)
            last_apply = Ref(zero(UInt64))
            function apply_range!()
                (!pending[] || applying[]) && return nothing
                pending[] = false
                applying[] = true
                last_apply[] = time_ns()
                try
                    _apply_sample_window!(graph, draft[]; live_vis = live_vis)
                finally
                    applying[] = false
                end
                return nothing
            end
            # Keep only the latest draft; render ticks and release flush it.
            function apply_drag!(; force = false)
                live_drag[] || return nothing
                if force || time_ns() - last_apply[] >= _LIVE_DRAG_INTERVAL_NS
                    apply_range!()
                end
                return nothing
            end
            _view_onany(subscription_owner, graph[:window_steps], draft) do applied, pending_range
                dirty[] = (touched[] || !isnothing(applied)) && applied != pending_range
            end
            _view_on(subscription_owner, slider.interval) do interval
                if !syncing[]
                    touched[] = true
                    draft[] = Int64.(interval)
                end
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
                    text_editing[] || value_control.replace!("$(interval[1]):$(interval[2])"; stored = true)
                finally
                    syncing[] = false
                end
                if live_drag[]
                    pending[] = true
                    apply_drag!(; force = !(Mouse.left in events(fig.scene).mousebuttonstate))
                end
            end
            _view_on(subscription_owner, value_entry.displayed_string) do _
                syncing[] && return
                touched[] = true
                dirty[] = true
                if live_drag[]
                    text_editing[] = true
                    try
                        commit_range_text!()
                    finally
                        text_editing[] = false
                    end
                end
            end
            _view_on(subscription_owner, dirty) do needs_apply
                active = needs_apply && !live_drag[]
                base = RGB(Makie.to_color(apply_color))
                accent = RGB(Makie.to_color(slider.color_active[]))
                apply.buttoncolor[] = active ? 0.82base + 0.18accent : apply_color
                apply.strokecolor[] = active ? RGBAf(accent) : apply_stroke
            end
            _view_on(subscription_owner, apply.clicks) do _
                commit_range_text!() || return
                if dirty[] || pending[]
                    pending[] = true
                    apply_range!()
                end
            end
            _view_on(subscription_owner, live_drag_button.clicks) do _
                live_drag[] = !live_drag[]
            end
            _view_on(subscription_owner, live_drag) do active
                Makie.notify(dirty)
                pending[] = active
                active && apply_drag!(; force = true)
            end
            _view_on(subscription_owner, events(fig.scene).tick) do _
                apply_drag!()
            end
            range_controls[] = (slider = slider, draft = draft, syncing = syncing, apply = apply,
                live_drag = live_drag, live_drag_button = live_drag_button, live_drag_toggle = live_drag_toggle,
                value_control = value_control, apply_drag! = apply_drag!,
                dirty = dirty, title = title, value_entry = value_entry)
            rowgap!(toggle_row_content, 1, 10)
            colgap!(toggle_row_content, 12)
            colsize!(toggle_row_content, 1, Fixed(40))
            colsize!(toggle_row_content, 2, Auto())
            colsize!(toggle_row_content, 3, Fixed(132))
        end
        # Growing data must not move the scale under a pressed pointer. Preserve
        # draft values as well as the applied range.
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
        if event.button == Mouse.left && event.action == Mouse.release
            isnothing(range_controls[]) || range_controls[].apply_drag!(; force = true)
            add_index_slider!()
        end
        return Consume(false)
    end

    notice = _control_notice(fig, toggle_box; owner = subscription_owner, duration = message_duration)
    lbl_upper = Box(ui_layout[2, 1]; width = 28, height = 28, color = :transparent, strokewidth = 0)
    lbl_diag = Box(ui_layout[3, 1]; width = 28, height = 28, color = :transparent, strokewidth = 0)
    lbl_lower = Box(ui_layout[4, 1]; width = 28, height = 28, color = :transparent, strokewidth = 0)
    _control_icon!(lbl_upper, :upper; filled = true)
    _control_icon!(lbl_diag, :diagonal)
    _control_icon!(lbl_lower, :lower; filled = true)
    for (block, text) in ((lbl_upper, "Upper triangle"), (lbl_diag, "Diagonal"), (lbl_lower, "Lower triangle"))
        _control_tooltip!(fig, block, text; owner = subscription_owner)
    end

    lbl_recipe = Label(fig, "Plot type")
    ui_layout[1, 2] = lbl_recipe
    ui_layout[2, 2] = menu_upper
    ui_layout[3, 2] = menu_diagonal
    ui_layout[4, 2] = menu_lower

    lbl_stats = Label(fig, "Statistics")
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
        lbl_trace = Label(fig, "Trace")
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

    colsize!(ui_layout, 1, Fixed(_LBL_ROW_WIDTH))
    colsize!(ui_layout, 2, Fixed(menu_width))
    colsize!(ui_layout, 3, Fixed(_LBL_STATS_WIDTH))
    has_trace_info && colsize!(ui_layout, 4, Fixed(_LBL_TRACE_WIDTH))

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

    divider_column = has_trace_info ? 5 : 4
    if !isnothing(picker_info)
        divider = Box(ui_layout[1:4, divider_column]; width = _UI_DIVIDER_WIDTH,
            height = nothing, tellheight = false, strokewidth = 0,
            color = _panel_bg_color(fig.scene.backgroundcolor[], 0.3))
        push!(ui_blocks, divider)
        colsize!(ui_layout, divider_column, Fixed(_UI_DIVIDER_WIDTH))
    end
    picker = isnothing(picker_info) ? nothing : _build_marginal_controls!(fig, ui_layout,
        graph, picker_info, divider_column + 1, notice, collapse_button; live_vis)
    body_size = isnothing(picker_info) ? 156.0 : _picker_body_size(picker_info.N)
    rowgap!(ui_layout, _UI_LAYOUT_ROW_GAP)
    colgap!(ui_layout, _UI_LAYOUT_COL_GAP)
    rowsize!(ui_layout, 1, Fixed(28))
    for row in 2:4
        rowsize!(ui_layout, row, Fixed((body_size - 2 * _UI_LAYOUT_ROW_GAP) / 3))
    end
    isnothing(picker_info) || colsize!(ui_layout, divider_column + 1, Fixed(body_size))
    controls_visible = Observable(false)
    minimum_size = (ceil(Int, panel_width) + 40, ceil(Int, body_size) + 430 + (isnothing(live_vis) ? 0 : 48))
    controls = _FigureControls(range_controls, picker, controls_visible, collapse_button, notice, minimum_size)
    _FIGURE_CONTROLS[fig.scene] = WeakRef(controls)
    _view_on(subscription_owner, controls_visible; update = true) do vis
        rowsize!(fig.layout, 3, vis ? Auto() : Fixed(0))
        controls_box.visible[] = vis
        for block in ui_blocks
            _set_block_visible!(block, vis)
        end
        isnothing(controls.picker) || controls.picker.set_visible!(vis)
        vis || controls.notice.dismiss!()
    end
    _view_on(subscription_owner, collapse_button.clicks) do _
        controls.expanded[] = !controls.expanded[]
    end
    _install_controls_minimum!(fig, controls.minimum_size; owner = subscription_owner)

    if !isnothing(live_vis)
        # The population label stays visible when controls collapse or the window changes.
        Label(fig[4, 1], "Observed sampling history\nMay include initialization and burn-in";
            tellwidth = false)
    end

    add_index_slider!()
    guarded_slider! = isnothing(live_vis) ? add_index_slider! :
        () -> _with_live_view(add_index_slider!, live_vis)
    return (fig = fig, add_index_slider! = guarded_slider!, range_controls = range_controls, controls = controls)
end


# Bound IntervalSlider's range scans and floating-point index arithmetic.
# The textbox still accepts exact integer endpoints.
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
# need zero width and height. Restore each block's desired dimensions; using
# Auto() would shrink-wrap a Menu whose default width is nothing.
const _BLOCK_NATURAL_SIZE = WeakKeyDict{Any,Tuple{Any,Any}}()
_natural_size!(b) = get!(() -> (b.width[], b.height[]), _BLOCK_NATURAL_SIZE, b)
const _BLOCK_VISIBILITY = WeakKeyDict{Any,Bool}()

function _set_block_geometry!(b, width, height)
    _BLOCK_NATURAL_SIZE[b] = (width, height)
    if get(_BLOCK_VISIBILITY, b, true)
        _silent_set!(b.width, width)
        b.height[] = height
    end
    return nothing
end

# Batch attribute writes without notification; the last write must notify to trigger relayout.
_silent_set!(observable, val) = setexcludinghandlers!(observe(observable), val)

function _set_block_visible!(b, v::Bool)
    _BLOCK_VISIBILITY[b] = v
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
    _BLOCK_VISIBILITY[b] = v
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


function _install_controls_minimum!(fig, minimum_size; owner = fig.scene)
    size = Tuple(Makie.viewport(fig.scene)[].widths)
    enlarged = max.(size, minimum_size)
    size == enlarged || Makie.resize!(fig, enlarged...)
    function apply_native_minimum!()
        screen = Makie.getscreen(fig.scene)
        isnothing(screen) && return nothing
        backend = parentmodule(typeof(screen))
        _is_gl_backend(backend) || return nothing
        isopen(screen) && screen.scene === fig.scene || return nothing
        current = Tuple(Makie.viewport(fig.scene)[].widths)
        target = max.(current, minimum_size)
        current == target || Makie.resize!(fig, target...)
        return nothing
    end
    # Clamp through Makie's scale-aware resize path after native resize events.
    # Persistent GLFW limits would leak into another figure when GLMakie reuses
    # its screen/window. These callbacks instead end with this figure's owner.
    _view_on(owner, events(fig.scene).window_open) do _
        apply_native_minimum!()
    end
    _view_on(owner, events(fig.scene).tick) do _
        apply_native_minimum!()
    end
    return nothing
end


function _initial_live_vsel(vis)
    view = _live_view(vis)
    controls = isnothing(view.figure) ? nothing : _figure_controls(view.figure)
    return isnothing(controls) || isnothing(controls.picker) ? vis.backend.vsel :
        copy(controls.picker.state.selection)
end


# Programmatic and widget changes use the same single graph transaction.
function _apply_sample_window!(graph, window; live_vis = nothing)
    if !isnothing(window)
        lo, hi = window
        1 <= lo <= hi || throw(ArgumentError("Require 1 <= window start <= window end"))
    end
    # Live intents may replace an earlier deferred value even when this value
    # equals the currently displayed one after another deferred selection.
    isnothing(live_vis) && graph[:window_steps][] == window && return false
    _view_graph_update!(graph, live_vis; window_steps = window)
    return true
end


# Static controls preserve direct graph updates; live controls use the view owner.
_view_graph_update!(graph, ::Nothing; kwargs...) = update!(graph; kwargs...)
_view_graph_update!(graph, vis; kwargs...) = _live_graph_update!(vis; kwargs...)
