# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Empirically measured (bat_theme's fontsize=20/12 defaults, standalone/
# unconstrained figures) natural widths of the controls panel's fixed
# (N-independent) labels -- re-measure if any label text or the ambient
# fontsize changes. Toggle's own natural width (32px) never dominates its
# column over these header labels, so it doesn't need its own constant here.
const _LBL_ROW_WIDTH = 80.0        # max("Upper", "Diagonal", "Lower")
const _LBL_STATS_WIDTH = 116.0     # "Stats overlay"
const _LBL_TRACE_WIDTH = 122.0     # "Trace overlay"
const _LBL_MARGINALS_WIDTH = 182.0 # "Displayed Marginals"
const _UI_COL2_MENU_WIDTH = 200.0  # fixed recipe-menu column width (colsize! below)
# Vsel picker matrix: a fixed per-cell width (padded well above a checkbox's
# own ~11px default to leave headroom for rescale_picker!'s later, height-
# driven cell growth -- see its own comment) times N, plus its row-index
# label column (padded for 2-digit indices).
const _PICKER_LABEL_WIDTH = 14.0
const _PICKER_CELL_WIDTH = 22.0
# GridLayoutBase's own default inter-column gap, applied between every one
# of ui_layout's 5 columns (4 gaps) -- included so the width estimate below
# doesn't systematically undershoot by ignoring it.
const _UI_LAYOUT_COL_GAP = 16.0

# The controls panel's (and, matching it, the anchor row's) total needed
# width -- computed directly from known quantities (N, the label
# measurements above), not by asking GridLayoutBase to determine it
# bottom-up through several nested GridLayout levels: a GridLayout/Block
# reporting `nothing` (as ui_layout and picker_layout both do, deliberately,
# to fill their assigned cell -- see their own width[] comments below)
# contributes *nothing* toward an ancestor's own Auto() size determination,
# and a single `nothing` anywhere in a multi-level chain makes the whole
# chain's determination fail -- fragile in a way a direct formula isn't.
function _controls_panel_width(picker_info::Union{NamedTuple,Nothing}, ui_box_pad::Real; has_trace_col::Bool=true)
    fixed_cols = _LBL_ROW_WIDTH + _UI_COL2_MENU_WIDTH + _LBL_STATS_WIDTH +
                 (has_trace_col ? _LBL_TRACE_WIDTH : 0.0)
    picker_col_width = if isnothing(picker_info)
        0.0
    else
        max(_LBL_MARGINALS_WIDTH, _PICKER_LABEL_WIDTH + picker_info.N * _PICKER_CELL_WIDTH)
    end
    n_cols = (isnothing(picker_info) ? 3 : 4) + (has_trace_col ? 1 : 0)
    return 2 * ui_box_pad + fixed_cols + picker_col_width + (n_cols - 1) * _UI_LAYOUT_COL_GAP
end

# Gives a panel piece (a GridLayout or its background Box) this session's
# constant, grid-independent width: a literal Fixed size (not Auto()) so it
# never depends on the grid's own Aspect(1,1)-driven column, and
# halign=:center so it's centered under the grid regardless of whether that
# column ends up wider (extra space split evenly on both sides) or narrower
# (protrudes evenly past both edges -- see rowsize!(fig.layout,1,...) below
# for why that's an accepted tradeoff).
_fix_panel_size!(x, panel_width::Real) = (x.width[] = Fixed(panel_width); x.halign[] = :center)

# Looks up `current_recipe`'s dropdown label in `options` (a Vector of
# (label, RecipeType) tuples) -- raises a clear, actionable ArgumentError
# instead of an opaque "invalid index: nothing" if current_recipe isn't
# actually one of `options` (e.g. ChainScatter2D configured directly via the
# `recipes` argument against a sample set with no chain identity, bypassing
# the has_chain_info gating that only applies to this dropdown's own
# contents -- confirmed reachable via direct live repro).
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
    picker_info::Union{NamedTuple,Nothing}=nothing;
    has_chain_info::Bool=false,
    has_trace_info::Bool=false,
)
    # fig.layout's column 1 holds the grid (row 1), toggle_row (row 2), and
    # controls_layout (row 3), but only the *grid's own* width is driven by
    # this column's Aspect(1,1)/Auto() sizing below -- toggle_row and
    # controls_layout each get their own explicit, constant Fixed width
    # instead (_fix_panel_size! above), so the grid stays free to be as
    # large as the available viewport allows (particularly while the panel
    # is collapsed) without being capped by the panel's own width need.
    #
    # The starting size scales with the grid size (anchored to reproduce the
    # long-standing (665, 850) default exactly at N_max=3): a fixed size
    # left large grids (e.g. N_max=15) opening illegibly cramped, with axis
    # labels overlapping into unreadable clutter until a manual resize!.
    # Capped so very large grids don't request an absurd window -- resizing
    # beyond the cap is still up to the user/window manager.
    n_grid = size(graph[:live_map][], 1)
    grid_growth = 170 * max(n_grid - 3, 0)
    fig = Figure(size=(min(665 + grid_growth, 1600), min(850 + grid_growth, 1785)))

    plot(fig[1, 1], gridlayout)

    colsize!(fig.layout, 1, Aspect(1, 1))
    # Auto() (not a hardcoded Relative fraction): the corner grid has no
    # aspect- or content-determined natural size of its own (it's built
    # entirely from Axis cells, which fill whatever space they're given), so
    # GridLayoutBase classifies this row as undeterminable and hands it
    # whatever remains after the anchor row (2) and controls row (3) below
    # claim their own real, content-derived Auto heights -- see
    # _init_gridlayout's alignmode=Outside(...) for why this row's resolved
    # size also no longer depends on which/how many variables are selected.
    # Relative(x) would be the wrong tool here even at x=1.0: it resolves as
    # x * this grid's own total available space, not x * the remainder
    # after sibling rows (compute_col_row_sizes in GridLayoutBase's
    # gridlayout.jl), so it would silently overflow past the anchor/controls
    # rows instead of sharing space with them.
    #
    # This Aspect(1,1)/Auto() pair keeps the grid genuinely maximal (using
    # nearly the full figure) independent of toggle_row/controls_layout's
    # own Fixed width below. In a short window, the grid can end up
    # *smaller* than that fixed width -- the anchor/panel then visibly
    # extend a little past the grid's own edges, an accepted tradeoff
    # (maximal space is lower priority than a simple, robust mechanism once
    # the panel is expanded) rather than the grid growing non-square or the
    # figure overflowing.
    rowsize!(fig.layout, 1, Auto())

    # Always-visible row holding just the controls-collapse toggle (and the
    # current-index slider, see below) -- kept outside controls_layout so it
    # never disappears along with the block it's meant to reveal. Visually
    # separated from the rest of the figure via a rounded, subtly-shaded
    # panel behind it: toggle_row itself is a 3x3 wrapper of Fixed(ui_box_pad)
    # margin rows/cols around a single inner content cell (toggle_row_content,
    # holding the actual button/label/slider below) purely so the Box
    # (assigned to the same fig[2,1] span, and thus filling the exact same
    # resolved bbox as toggle_row itself -- confirmed empirically that a Box
    # and a GridLayout can coexist at the same grid position, the Box acting
    # as a background since it's created first and Makie draws blocks in
    # creation order) shows a visible inset margin around the content instead
    # of a border tightly hugging it. GridLayout(3, 3) (explicit dims, not
    # left to auto-expand from content) is required for rowsize!/colsize! to
    # accept row/col 3 at all -- nothing is ever assigned directly into it.
    # ui_box_pad is a third of the ambient fontsize (not a fixed pixel guess),
    # so it scales with the theme rather than needing separate re-tuning.
    # rowgap!/colgap!(..., 0): GridLayoutBase's own default inter-cell gap
    # (applied *between* every pair of the 3 rows/cols regardless of their
    # own sizes) would otherwise stack on top of these margins and make the
    # padding come out uneven/much larger than ui_box_pad -- confirmed
    # empirically (padding was ~3.7x the intended value with the default gap
    # left in place).
    ui_box_pad = fig.scene.theme[:fontsize][] / 3
    # The one constant this whole panel is built around -- computed once
    # from N alone (fixed for the life of this figure), never touched again:
    # this *is* what makes the anchor bar and the controls panel genuinely
    # constant-width, by construction. See _controls_panel_width's own
    # comment for why it's a direct formula rather than measured bottom-up.
    panel_width = _controls_panel_width(picker_info, ui_box_pad; has_trace_col=has_trace_info)
    toggle_row = fig[2, 1] = GridLayout(3, 3)
    rowgap!(toggle_row, 0)
    colgap!(toggle_row, 0)
    # A Box left at its own default width would fill the *whole*
    # grid-derived column instead of tightly matching toggle_row's content.
    toggle_box = Box(fig[2, 1], color=_panel_bg_color(fig.scene.backgroundcolor[]), cornerradius=10, strokewidth=0)
    _fix_panel_size!(toggle_box, panel_width)
    toggle_row_content = toggle_row[2, 2] = GridLayout()
    rowsize!(toggle_row, 1, Fixed(ui_box_pad))
    rowsize!(toggle_row, 3, Fixed(ui_box_pad))
    colsize!(toggle_row, 1, Fixed(ui_box_pad))
    colsize!(toggle_row, 3, Fixed(ui_box_pad))
    # Auto (not a hardcoded Fixed height guess) so this always matches
    # whatever toggle_row_content's actual content needs, regardless of
    # ui_box_pad or font-size changes -- toggle_row's own height is left at
    # its Block/GridLayout default, which is exactly the recursively-computed
    # natural height Auto needs here (Fixed(pad) + content + Fixed(pad));
    # width is set explicitly instead, via _fix_panel_size! above.
    _fix_panel_size!(toggle_row, panel_width)
    rowsize!(fig.layout, 2, Auto())

    # Everything else (recipe/stats menus and the vsel picker matrix) now
    # lives nested inside a single collapsible block. Same rounded-panel
    # treatment as toggle_row above -- see its comments for why each piece
    # is needed. Gets the identical _fix_panel_size!(panel_width) too, so
    # this and toggle_row are always the same width as each other by
    # construction, independent of whatever the grid's own column is.
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
    # Gap between the two boxes (toggle_row's above, controls_layout's below)
    # -- rowgap! with an explicit row index targets only *this* gap, unlike
    # the blanket rowgap!(gl, gap) form used to zero out the wrapper grids'
    # own internal gaps above. Only valid now that fig.layout actually has 3
    # rows (i.e. after controls_layout's own fig[3,1] assignment above) --
    # confirmed empirically that calling this any earlier throws ("invalid
    # row gap 2"), since GridLayoutBase only grows fig.layout's row count as
    # content gets assigned to new rows.
    rowgap!(fig.layout, 2, ui_box_pad)

    # ui_layout holds the row labels + recipe/stats/trace menus (columns
    # 1-4) *and* the vsel picker's title/matrix (column 5, added in
    # _build_vsel_picker! below) -- all in the same GridLayout, deliberately,
    # so the picker's title shares ui_layout's own row 1 (guaranteeing its
    # top/bottom edges exactly match the "Recipe"/"Stats overlay"/"Trace
    # overlay" header labels there) and the matrix spans
    # rows 2:4 (guaranteeing its bottom edge exactly matches menu_lower's
    # row), with no separate alignment bookkeeping needed -- two blocks in
    # the same grid row/rows share pixel-exact boundaries by construction.
    ui_layout = controls_layout[2, 2] = GridLayout()

    # A nested GridLayout's `width` attribute defaults to `Auto()`, just like
    # a Block's -- and just like Menu (see _set_block_visible!'s comments),
    # `Auto()` means "use my own bottom-up-computed size instead of whatever
    # the parent cell suggests", not "fill the parent cell". Confirmed
    # empirically: without this, these end up sized from their own content
    # instead of the intended width, overflowing past it in some cases and
    # auto-centering short of it in others. `width = nothing` (like Menu's
    # own default) makes a block/layout defer to the suggested bbox
    # unconditionally, which is the actual "fill this cell" behavior we want
    # here -- toggle_row_content fills toggle_row's own (now Fixed-width, see
    # above) inner content cell, and ui_layout fills controls_layout's
    # (likewise now Fixed-width) inner content cell. toggle_row/
    # controls_layout themselves are deliberately NOT included here -- they
    # have their own explicit Fixed width set above instead of deferring.
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
    # Only offered when the plotted samples actually carry chain identity
    # (see _samples_have_chain_ids in makie_scatter.jl) -- meaningless (and
    # would just show every point as a single color) for samples from a
    # non-MCMC sampler.
    has_chain_info && push!(options2D, ("Scatter (by chain)", ChainScatter2D))
    options1D = [
        ("QuantileHist", QuantileHist1D),
        ("Hist", Hist1D),
        ("KDE", KDE1D),
        ("QuantileKDE", QuantileKDE1D),
        # "Normal fit", not "PDF": the recipe overlays a Gaussian fitted by
        # weighted mean/std -- calling that "PDF" implied it was the actual
        # posterior density.
        ("Normal fit", PDF1D),
        ("Errorbars", Errorbars1D),
    ]

    # graph[:upper_recipe][]/etc. might not actually be present in `options`
    # -- e.g. ChainScatter2D configured directly via the `recipes` argument
    # (bypassing this dropdown, which is correctly gated by has_chain_info
    # above, but a caller can still pass recipes=(upper=ChainScatter2D,...)
    # straight through to bat_makie_plot/Makie.plot against a sample set with
    # no chain identity). findfirst then returns `nothing`, and indexing
    # options[nothing] previously threw an opaque "invalid index: nothing of
    # type Nothing" -- confirmed via direct live repro. Raising a clear,
    # actionable error here instead.
    default_upper = _default_recipe_label(options2D, graph[:upper_recipe][])
    default_diag = _default_recipe_label(options1D, graph[:diagonal_recipe][])
    default_lower = _default_recipe_label(options2D, graph[:lower_recipe][])

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

    # Samples must already exist for the slider to make sense (an empty 1:0
    # slider range crashes) -- true at build time for the static path, never
    # for the live path, whose figure is built before the first flush (the
    # listener's teardown retrofits the slider once sampling is done). No
    # chain/walker-count restriction: the window is defined in real MCMC
    # steps and applied per walker (:window_steps/_step_window_rows in
    # makie_compute_graph.jl), well-defined for any number of chains and
    # walkers.
    show_slider = graph[:max_step][] > 0

    # Collapse toggle for the whole controls block. Created here (before the
    # slider machinery below) since the slider spans the button's rows and
    # repositions it. Placed assuming no slider first: the button's column
    # claims the row's full width so its halign=:left is flush against the
    # true left edge instead of a shrink-wrapped, auto-centered one --
    # add_index_slider! (below) undoes both when/if the slider appears.
    collapse_button = Button(fig, label="☰", halign=:left, valign=:top)
    toggle_row_content[1, 1] = collapse_button
    colsize!(toggle_row_content, 1, Relative(1))

    # Builds the "Step Range" IntervalSlider row into toggle_row (always
    # visible, right of the collapse button, never hidden by the controls
    # collapse -- per explicit request). Invoked immediately for the static
    # path (samples already exist), or retroactively at the end of a live
    # run via the returned hook: the live figure is built before any samples
    # exist, so it can never qualify at build time -- previously that meant
    # a finished live run simply had no way to review its history in place.
    slider_added = Ref(false)
    function add_index_slider!()
        slider_added[] && return nothing
        slider_added[] = true

        # Undo the no-slider layout for the button (see its comment above):
        # column 1 shrinks back to the button's own width, and the button
        # spans both slider rows so its top/bottom edges land exactly on the
        # label's top and the slider's bottom. `height=Relative(1)` (not the
        # Auto() default) makes it *fill* that combined span exactly rather
        # than centering its own natural height within it.
        colsize!(toggle_row_content, 1, Auto())
        collapse_button.height[] = Relative(1)
        toggle_row_content[1:2, 1] = collapse_button

        # "Step Range" sits in its own row directly above the slider
        # (row 1), the slider itself in row 2. The label spans *both*
        # column 2 (the slider) and column 3 (the value display) rather than
        # just column 2 -- deliberately, so its centered position (halign
        # =:center, Label's default) is the midpoint of that whole 2:3 span,
        # which runs from the slider's left edge (immediately after the
        # fixed-width button) to the row's own right edge, both of which are
        # constant. Centering over column 2 alone instead would move the
        # label every time the value display's text width changes (e.g.
        # "500" vs "50000"), since that changes how the fixed remaining
        # width splits between columns 2 and 3 even though their *union*
        # doesn't move -- confirmed empirically.
        lbl_idx_title = Label(toggle_row_content[1, 2:3], "Step Range")
        # IntervalSlider over REAL MCMC STEPS (1:max_step, not stored-row
        # indices -- for repetition-weighted samples the step count exceeds
        # the row count): shows only the samples whose dwell intersects the
        # selected step window, time-aligned across all walkers and chains.
        # startvalues = the full range (everything shown). 0-based: MCMC
        # step numbering starts at 0 (the initial position), so a window
        # starting at step 1 would correctly-but-surprisingly exclude each
        # walker's initial sample. Range read at call time, so the post-run
        # retrofit sees the final step count.
        slider_curr_idx = IntervalSlider(toggle_row_content[2, 2], range=0:graph[:max_step][], startvalues=(0, graph[:max_step][]))
        # No gap between the label/slider rows -- and each pinned to the
        # outer edge of its own row (valign=:top / :bottom) rather than the
        # default :center, which would otherwise leave slack split above the
        # label and below the slider (confirmed empirically: with the
        # default :center, the slider's bottom sat ~5px short of the button's
        # bottom, even with the row gap zeroed). Pinning outward is what
        # makes the label's top and the slider's bottom land exactly on the
        # button's own top/bottom edges.
        rowgap!(toggle_row_content, 1, 0)
        lbl_idx_title.valign[] = :top
        slider_curr_idx.valign[] = :bottom
        # Deferring to whatever column 2 is given (rather than the slider's
        # own natural/Auto width) is what makes it actually span the row's
        # remaining width -- see the width[]=nothing comments above for why
        # Auto() wouldn't do this (Slider, like most Blocks, reports a real
        # non-nothing autosize, which would leave its column sized to just
        # that instead of expanding to fill whatever's left).
        slider_curr_idx.width[] = nothing
        # The label also can't be left reporting its own Auto() width to the
        # column: a column's Auto width is the max over every (single-span)
        # item placed in it across all rows, skipping only ones reporting
        # `nothing` -- confirmed empirically that leaving it in place made
        # column 2 "determined" at the label's own (much narrower) text
        # width, shrinking the slider to match instead of filling the row.
        # But unlike the slider, `width[]=nothing` is the wrong fix here: it
        # makes the label's *own* bbox span the full column too, and Label
        # always positions its text at `bbox.origin + 0.5*textwidth` (its own
        # halign attribute isn't consulted for single-line text placement,
        # only for word-wrap justification) -- so a full-width bbox renders
        # the text hard against the *left* edge instead of centered
        # (confirmed empirically). `tellwidth=false` instead removes it from
        # the column's width computation without touching its own width
        # attribute (still Auto(), i.e. sized tightly to its own text), so
        # its computed bbox stays text-sized and its halign=:center (Label's
        # own default) centers *that* box within column 2's full width.
        lbl_idx_title.tellwidth[] = false
        lbl_idx_value = Label(toggle_row_content[2, 3], lift(iv -> "$(iv[1]):$(iv[2])", slider_curr_idx.interval))
        # Column 1 (button) and 3 (value label) are left at their Auto()
        # default, so they size to their own content and column 2 (the only
        # one reporting no natural width) absorbs whatever's left over.

        # One tuple input in real MCMC steps, applied per walker inside
        # :flat_samples (see _step_window_rows). The slider no longer writes
        # :current_idxs -- that stays pure "rows that exist" bookkeeping
        # owned by registration/flush.
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

    # Collected here (rather than only living as local variables) so the
    # whole-controls collapse toggle below can show/hide every one of them
    # together -- one mechanism, not one per widget.
    ui_blocks = Any[
        lbl_upper, lbl_diag, lbl_lower, lbl_recipe,
        menu_upper, menu_diagonal, menu_lower,
        lbl_stats, toggle_upper, toggle_diag, toggle_lower,
    ]

    # Trace-overlay column only for sample sources that can actually drive
    # Trace2D (chainid AND stepno -- see _samples_have_trace_info in
    # makie_trace.jl): for anything else the toggles rendered but silently
    # did nothing. Row 3 (Diagonal) stays unassigned even when present -- a
    # trace is an inherently 2D concept. Without this column, the vsel
    # picker moves up to column 4 and the panel-width estimate drops the
    # trace column's share (_controls_panel_width's has_trace_col).
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
    # The same constant _controls_panel_width's estimate uses -- a bare
    # literal here could silently drift from the width formula.
    colsize!(ui_layout, 2, _UI_COL2_MENU_WIDTH)
    colsize!(ui_layout, 3, Auto())
    # Only when the trace widgets above actually created column 4: the vsel
    # picker (which otherwise occupies it) is built further below, so the
    # column doesn't exist yet here -- and its size is left at the Auto()
    # default anyway, exactly as the picker's column always was.
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

    rescale_picker! = nothing
    lbl_marginals = nothing
    if !isnothing(picker_info)
        (; N, N_max, initial_vsel, apply_vsel!) = picker_info
        picker_blocks, rescale_picker!, lbl_marginals = _build_vsel_picker!(
            fig, ui_layout, graph, N, N_max, initial_vsel, apply_vsel!,
            has_trace_info ? 5 : 4,
        )
        append!(ui_blocks, picker_blocks)
    end

    # Starts collapsed (false) per explicit request -- update=true is what
    # makes the handler actually apply that at construction time, since `on`
    # only fires on *future* notifications by default and every ui_block
    # starts out fully visible/expanded otherwise.
    #
    # No reactive whole-UI scale factor: every widget here renders at its one
    # true, unscaled design size (fontsize, ui_box_pad, the fixed 200px menu
    # column, widget width/height/size/markersize) always, whether expanded
    # or collapsed. An earlier version of this reactively rescaled everything
    # against the figure's own resolved bbox to avoid clipping on a smaller
    # screen -- removed as a deliberate simplification: the panel/anchor are
    # now treated as constant-sized, and an extremely small window clipping
    # the (unscaled) panel is an accepted, pathological edge case rather than
    # something worth the considerable reactive-scaling machinery it used to
    # take to avoid. Auto() below is what keeps the corner grid (row 1) from
    # being squeezed by this row's real size instead -- see its own comment
    # in this function.
    controls_visible = Observable(false)
    on(controls_visible; update=true) do vis
        # Auto() when expanded: the panel claims exactly whatever height its
        # own real, unscaled content needs; row 1's own Auto() sizing (see
        # above) is what gives the corner grid the rest.
        rowsize!(fig.layout, 3, vis ? Auto() : Fixed(0))
        # controls_box doesn't go through _set_block_visible! (it's not part
        # of ui_blocks -- it's the always-created panel background, not a
        # collapsible widget), so its own visibility needs setting directly:
        # deferring to fig.layout row 3's Fixed(0) alone left a ~1px sliver
        # visible at the collapsed height (floating-point rounding in the
        # nested GridLayout size resolution, confirmed empirically -- never
        # exactly 0), which a plain flat-color Box has no automatic occlusion
        # for the way zero-area content naturally would.
        controls_box.visible[] = vis
        for b in ui_blocks
            _set_block_visible!(b, vis)
        end
        # The per-widget restore loop above processes ui_layout's menus
        # before the picker matrix's own blocks, so mid-loop the matrix's
        # available height is transiently smaller than its final value --
        # _set_block_visible! would otherwise leave checkboxes/labels sized
        # to that transient value instead of the correct final one (confirmed
        # empirically). This forces one final, correct pass once everything
        # else has fully settled.
        vis && !isnothing(rescale_picker!) && rescale_picker!()
    end
    on(collapse_button.clicks) do _
        controls_visible[] = !controls_visible[]
    end

    # add_index_slider! is exposed so the live path can retrofit the index
    # slider once sampling finishes -- see its definition above. Idempotent
    # (latches on first call), and a single lexical closure site, so the
    # static path's build-time call warms the same compiled specialization
    # the live teardown call uses.
    return (fig=fig, add_index_slider! = add_index_slider!)
end

# Pure decision logic, kept separate from the widget wiring below so it can be
# tested directly without real mouse events (which CairoMakie can't produce).
# Checking cell (i,j) adds both i and j to the active set (a diagonal cell
# i==j just adds i); unchecking removes both. The active set -- not individual
# checkbox states -- is the source of truth, so removal is explicit rather
# than re-derived from sibling checkboxes that haven't been resynced yet
# (which would make e.g. unchecking a diagonal while some other still-checked
# pair references it snap the diagonal back on).
function _vsel_after_toggle(active_vars::Set{<:Integer}, i::Integer, j::Integer, is_on::Bool)
    return is_on ? union(active_vars, (i, j)) : setdiff(active_vars, (i, j))
end

# Whether cell (i,j) should display as checked given the active set: both
# endpoints must be selected (a diagonal cell i==j just needs i).
_checkbox_should_be_checked(active_vars::Set{<:Integer}, i::Integer, j::Integer) = (i in active_vars) && (j in active_vars)

# Most Blocks (Button/Menu/Toggle/Slider/Checkbox/...) have no direct
# `visible` attribute -- confirmed empirically (hasproperty(b, :visible) is
# false for all of these) -- so toggling their underlying blockscene's
# visibility is the general mechanism. Label/Box are the exception: they do
# have their own `visible` attribute directly, and use it instead.
#
# blockscene visibility alone isn't enough for these, though: an interactive
# block's own click handling (Menu's dropdown, Button's click, Toggle/Slider
# drag, Checkbox's toggle) is driven by a purely geometric hit-test against
# the block's *own computed bounding box* -- entirely independent of
# `visible` (confirmed empirically: a Menu still opens its dropdown from a
# click at its old screen position even with blockscene.visible[]=false).
# The reason the bbox doesn't shrink on its own: GridLayoutBase's `Fixed(0)`
# row/column size is only a *suggestion* to the block; whatever the block's
# own `width`/`height` attribute resolves to takes precedence, and computes
# its bbox from its natural/reported size regardless of how small the
# assigned cell actually is. Forcing width/height to a literal `0` collapses
# the computed bbox to zero area, which *is* geometrically unclickable.
#
# The original width/height must be *remembered per block* rather than
# restored to a hardcoded `Auto()`, though: not every block defaults to
# `Auto()` -- Menu defaults to `width === nothing`, which means "just use
# whatever the parent GridLayout cell suggests" (this is exactly what gives
# every recipe menu here its uniform, column-driven width, since ui_layout's
# column 2 is a fixed 200px). `Auto()` is a *different* value with different
# semantics: it means "use my own text-derived autosize if one is available",
# and Menu unconditionally always makes one available (it recomputes its
# autosize from its current selected-text bounding box on every change) --
# so overwriting `nothing` with `Auto()` on restore silently switches a Menu
# from column-width to shrink-wrapped-to-text width permanently, which is
# exactly the "Hist"-vs-"QuantileHist" width mismatch this once caused.
const _BLOCK_NATURAL_SIZE = WeakKeyDict{Any,Tuple{Any,Any}}()
_natural_size!(b) = get!(() -> (b.width[], b.height[]), _BLOCK_NATURAL_SIZE, b)

# Sets an Observable's value without notifying its listeners -- used below to
# collapse several attribute writes on the *same* widget into a single
# relayout instead of one per attribute. Confirmed empirically necessary,
# not just a micro-optimization: collapsing/expanding the controls panel for
# a realistic model (N=10 free variables, 55 vsel-picker cells) measured at
# 1-2.7s and 10-21M allocations *per click*, in steady state (not a JIT
# artifact -- 0% compilation time) -- traced to ~4,149 GridLayoutBase
# relayout notifications for a single click, each individual widget
# attribute write (width/height/size, 2-4 per widget x ~130 widgets)
# independently triggering a full relayout cascade up the nested GridLayout
# hierarchy. Also confirmed empirically that this can't be batched *across*
# widgets (notifying only the last of many silently-updated widgets leaves
# every other one showing its old, stale bbox) -- each widget still needs
# its own final notifying write; this only removes the *redundant extra*
# writes on top of that one.
_silent_set!(observable, val) = setexcludinghandlers!(observe(observable), val)

function _set_block_visible!(b, v::Bool)
    _silent_set!(b.blockscene.visible, v)
    w, h = _natural_size!(b)
    _silent_set!(b.width, v ? w : 0)
    b.height[] = v ? h : 0 # last write of this widget's batch -- the only one that actually notifies
    return nothing
end
_set_block_visible!(b::Union{Label,Box}, v::Bool) = (b.visible[] = v)

# Checkbox is a further exception to the generic method above: its clickable
# area is a *fixed*-size square (its `size` attribute) centered on
# computedbbox, independent of that bbox's own dimensions (confirmed in
# Makie's checkbox.jl source) -- so zeroing width/height alone still leaves a
# fully clickable checkbox sitting at the collapsed bbox's center point. Needs
# `size` zeroed/restored too; its natural size is remembered (weakly, so a
# checkbox that's later discarded can still be GC'd) the first time it's
# hidden, since that capture must happen before it's ever been zeroed.
const _CHECKBOX_NATURAL_SIZE = WeakKeyDict{Checkbox,Float64}()
function _set_block_visible!(b::Checkbox, v::Bool)
    _silent_set!(b.blockscene.visible, v)
    w, h = _natural_size!(b)
    _silent_set!(b.width, v ? w : 0)
    _silent_set!(b.height, v ? h : 0)
    if v
        b.size[] = get(_CHECKBOX_NATURAL_SIZE, b, b.size[]) # last write of this widget's batch -- the only one that actually notifies
    else
        haskey(_CHECKBOX_NATURAL_SIZE, b) || (_CHECKBOX_NATURAL_SIZE[b] = b.size[])
        b.size[] = 0 # last write of this widget's batch -- the only one that actually notifies
    end
    return nothing
end

# Builds the always-visible "Displayed Marginals" title + N x N variable
# picker matrix (N = total model dimensionality, not N_max), placed in
# `picker_col` (a column of ui_layout, see _build_fig -- column 5, after the
# recipe/stats/trace columns) : the title shares ui_layout's row 1 with the
# "Recipe"/"Stats overlay"/"Trace overlay" headers (so their top/bottom edges
# match exactly, being the same grid row) and the matrix spans rows 2:4 (so
# its bottom edge exactly matches menu_lower's row) -- both guaranteed by
# GridLayoutBase, not by manual pixel-matching. The lower triangle including
# the diagonal (i >= j) is interactive -- cell (i,j) and (j,i) are the same 2D
# marginal, so only one needs a checkbox; a diagonal cell (i,i) selects
# variable i on its own. Only the strict upper triangle is decorative.
# Checking a cell on/off means "include/exclude these variable(s) in vsel";
# the actual N_max x N_max corner-plot grid always shows every pairing among
# the currently selected variables, so there's no independent per-pair
# visibility beyond which variables are selected.
#
# Takes graph/N/N_max/initial_vsel/apply_vsel! directly rather than a
# BATVisualizer so it also works for the static bat_makie_plot path, which has
# no vis/content at all. Returns (blocks, rescale_picker!, lbl_marginals):
# every block it creates, so the caller's whole-controls collapse toggle can
# fold this matrix's visibility into its own (it's no longer independently
# collapsible); a callback the caller must invoke once after fully
# re-expanding the controls; and the "Displayed Marginals" title label itself,
# so the caller's own whole-UI rescale can keep its fontsize in step with the
# other column headers (deliberately excluded from rescale_picker!'s own
# cell-driven scaling below -- see its comment). That second (callback)
# return value is not redundant with the automatic bbox-driven rescaling
# above: collapsing/expanding processes ui_blocks (menus first, then this
# matrix's own blocks) one at a time, so
# mid-restore the matrix's available height is transiently smaller than its
# final value -- confirmed empirically to otherwise leave cells sized to that
# transient value instead of the correct final one.
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
    lbl_marginals = Label(fig, "Displayed Marginals")
    ui_layout[1, picker_col] = lbl_marginals

    picker_layout = ui_layout[2:4, picker_col] = GridLayout()
    # Deferring to the assigned row-span (rather than the matrix's own small
    # natural size) is what makes it actually *fill* that span all the way
    # down to row 4's bottom edge, instead of centering its small natural
    # size somewhere in the middle of it -- see the width[]=nothing comments
    # above for why a GridLayout's default Auto() sizing wouldn't do this.
    picker_layout.width[] = nothing
    picker_layout.height[] = nothing

    # Warning label (e.g. "can't select more than N_max...") sits below the
    # matrix, spanning its full width.
    status_label = Label(fig, "", fontsize=12, color=:red)

    initial_vsel_set = Set(initial_vsel)
    checkboxes = Dict{Tuple{Int,Int},Checkbox}()
    all_blocks = Union{Checkbox,Label,Box}[status_label, lbl_marginals]
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
                # Widget-shade from the figure's ACTUAL background (two ladder
                # steps, like the themed widgets) instead of a hardcoded
                # :gray70, which sat far outside the dark theme's shade ladder
                # and rendered as bright out-of-place squares there.
                bx = Box(fig, color=_panel_bg_color(_panel_bg_color(fig.scene.backgroundcolor[])), width=20, height=20)
                picker_layout[i+1, j+1] = bx
                push!(all_blocks, bx)
            end
        end
    end
    picker_layout[N+2, 1:(N+1)] = status_label

    # Scales the matrix's cells (checkbox click-target size, row/column
    # index label fontsize, decorative box size) to the space picker_layout
    # actually ends up filling, rather than leaving them at their small
    # default size with dead space around them now that the matrix fills the
    # whole label-to-bottom-edge span above. Unconditional (no dedup) so the
    # explicit external call the caller makes after re-expanding (see this
    # function's docstring) always forces a fresh, correct application.
    function rescale_picker!()
        h = picker_layout.layoutobservables.computedbbox[].widths[2]
        h < 10 && return
        cell = h / (N + 2) # matrix rows (N+1) plus the status-label row
        cb_size = max(4.0, 0.75 * cell)
        lbl_fontsize = clamp(0.45 * cell, 6.0, 16.0)
        for b in all_blocks
            if b isa Checkbox
                b.size[] = cb_size
            elseif b isa Box
                _silent_set!(b.width, cb_size)
                b.height[] = cb_size # last write of this widget's batch -- the only one that actually notifies
            elseif b !== lbl_marginals && b !== status_label
                b.fontsize[] = lbl_fontsize
            end
        end
        return nothing
    end
    # Reacts to picker_layout's own resolved bbox (not a one-off calculation
    # at construction time) so it also adapts to window resizes. last_h
    # dedupes re-entrant firings *here* specifically: changing a cell's size
    # above triggers its own relayout (Label/Checkbox both feed their size
    # back into the layout machinery), which unconditionally renotifies this
    # same computedbbox Observable even when its value comes out identical
    # (GridLayoutBase doesn't skip notification on equal values here) --
    # without this guard that's an infinite loop, confirmed empirically
    # (StackOverflowError). The explicit external call above bypasses this
    # dedup entirely (calling rescale_picker! directly, not through here),
    # which is required -- see this function's docstring for why.
    last_h = Ref(-1.0)
    on(picker_layout.layoutobservables.computedbbox) do bbox
        h = bbox.widths[2]
        (h < 10 || h == last_h[]) && return
        last_h[] = h
        rescale_picker!()
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

    return all_blocks, rescale_picker!, lbl_marginals
end
