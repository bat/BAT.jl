# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Shared setup of the static (all-samples-already-exist) entry points --
# Makie.convert_arguments, BAT.bat_makie_plot, and the precompile workload:
# builds the compute graph, registers the samples as a single chain/walker,
# sets the true data-range domain (strictly more accurate than the
# prior-based estimate the live path needs -- see _domain_from_samples's
# docs), and applies the initial, clamped variable selection -- all in one
# batched update!. Previously ~35 near-identical lines per entry point, a
# silent-drift hazard. Deliberately does NOT call warmup_makie_shaders():
# the precompile workload must stay backend-free (warmup rasterizes via
# colorbuffer), so the two user-facing entry points call it themselves.
function _setup_static_graph(
        samples::DensitySampleVector,
        recipes::NamedTuple,
        vsel::AbstractVector{<:Integer},
        N_max::Integer,
        triagonal_config::NamedTuple,
        diagonal_config::NamedTuple;
        # Hard prior-support bounds for KDE boundary reflection -- a completed
        # DensitySampleVector doesn't carry its measure, so the static path
        # can only know them if the caller says (a measure/prior/Distribution,
        # or explicit per-dim (lo, hi) pairs -- see _support_vectors). The
        # default means no correction, exactly the pre-existing rendering.
        support=nothing,
)
        graph = _init_compute_graph(recipes, triagonal_config, diagonal_config, N_max)

        unshaped_samples = unshaped.(samples)

        # Registered per (chainid, walkerid) -- the same chains-of-walkers
        # structure the live path uses -- rather than as one merged
        # pseudo-walker: BAT's merged multi-chain result is chain-block-
        # concatenated (all of chain A's rows, then chain B's, ...), so a
        # single shared row window used to reveal one chain's entire block
        # before the next chain's -- the root cause of Trace2D's historical
        # "frozen chains" bug (once worked around with a whole family of
        # untruncated _full graph nodes, since removed) and of the old
        # single-chain-only slider gate. With per-walker registration, the
        # step window (:window_steps) applies to each walker's own
        # chronologically-ordered rows, which is well-defined for any number
        # of chains and walkers. Groups are MATERIALIZED (getindex, one-time
        # O(n) at plot construction) rather than index-vector views, so each
        # registered walker has exactly the same concrete structure
        # (.v.data-backed DensitySampleVector) as a live walker. Samples
        # without chain identity register as one chain/one walker, with row
        # index standing in for the step number downstream.
        samples_graph = graph[:samples][]
        current_idxs_graph = graph[:current_idxs][]
        info_T = eltype(unshaped_samples.info)
        if hasfield(info_T, :chainid)
            wid_field = hasfield(info_T, :walkerid) ? :walkerid :
                        (hasfield(info_T, :walker) ? :walker : nothing)
            groups = Dict{Tuple{Int32,Int32},Vector{Int}}()
            for (k, inf) in pairs(unshaped_samples.info)
                key = (Int32(inf.chainid), isnothing(wid_field) ? Int32(0) : Int32(getfield(inf, wid_field)))
                push!(get!(() -> Int[], groups, key), k)
            end
            for cid in sort!(unique(map(first, collect(keys(groups)))))
                wids = sort!([w for (c, w) in keys(groups) if c == cid])
                walkers = [unshaped_samples[groups[(cid, w)]] for w in wids]
                push!(samples_graph, walkers)
                push!(current_idxs_graph, [length(w) for w in walkers])
            end
        else
            push!(samples_graph, [unshaped_samples])
            push!(current_idxs_graph, [length(samples)])
        end

        n_dof = totalndof(varshape(samples))
        domain_lo, domain_hi = _domain_from_samples(unshaped_samples.v.data, n_dof)
        support_lo, support_hi = _support_vectors(support, n_dof)

        update!(graph;
                samples=samples_graph,
                current_idxs=current_idxs_graph,
                domain_lo=domain_lo,
                domain_hi=domain_hi,
                support_lo=support_lo,
                support_hi=support_hi,
                idxs=_clamp_vsel(vsel, n_dof, N_max),
        )
        return graph, n_dof
end

function Makie.convert_arguments(
        ::Type{<:AbstractPlot},
        samples::DensitySampleVector;
        recipes::NamedTuple=(upper=QuantileHist2D, diagonal=Hist1D, lower=Hist2D),
        vsel::Vector{<:Integer}=[1, 2, 3],
        N_max::Integer=3,
        trace_nsteps::Integer=20,
        support=nothing,
)
        # Warms up backend-specific (CairoMakie/GLMakie) rendering of the raw
        # plot primitive types -- previously only called from the live path,
        # leaving this static path with no mitigation at all for the same
        # first-interaction latency. Complements (doesn't replace) the
        # PrecompileTools workload in makie_precompile.jl, which handles the
        # backend-agnostic recipe/SpecApi compilation ahead of time; this part
        # is backend-specific and can only happen once a concrete backend is
        # actually loaded, i.e. now.
        warmup_makie_shaders()

        graph, _ = _setup_static_graph(
                samples, recipes, vsel, N_max,
                _default_makie_triagonal_config(trace_nsteps=trace_nsteps),
                _default_makie_diagonal_config();
                support=support,
        )

        gridlayout = _init_gridlayout(graph, N_max)

        return gridlayout[]
end


function BAT.bat_makie_plot(
        samples::DensitySampleVector,
        recipes::NamedTuple=(upper=QuantileHist2D, diagonal=Hist1D, lower=Hist2D),
        vsel::Vector{<:Integer}=[1, 2, 3],
        N_max::Integer=3;
        dark::Bool=false,
        trace_nsteps::Integer=20,
        # Hard prior-support bounds for KDE boundary reflection: a measure/
        # prior/Distribution, or a vector of per-dim (lo, hi) pairs -- see
        # _support_vectors. Default: unknown, no correction.
        support=nothing,
)
        # TODO: MD, Discuss config handling and passing of user attribute overwrites

        # See Makie.convert_arguments' matching comment above for what this
        # warms up and why it lives at the entry point, not in
        # _setup_static_graph.
        warmup_makie_shaders()

        graph, n_dof = _setup_static_graph(
                samples, recipes, vsel, N_max,
                _default_makie_triagonal_config(trace_nsteps=trace_nsteps),
                _default_makie_diagonal_config();
                support=support,
        )

        picker_info = (
                N=n_dof,
                N_max=N_max,
                initial_vsel=vsel,
                apply_vsel! = new_vsel -> _apply_vsel_to_graph!(graph, n_dof, N_max, new_vsel),
        )

        with_theme(dark ? bat_theme_dark() : bat_theme()) do
                gridlayout = _init_gridlayout(graph, N_max)
                built = _build_fig(graph, gridlayout, picker_info;
                        has_chain_info=_samples_have_chain_ids(samples),
                        has_trace_info=_samples_have_trace_info(samples))
                display(built.fig)
        end
        return nothing
end

# Everything the light and dark themes share -- global font setup, the
# corner-grid Axis tick styling, and every per-plot-type recipe block that's
# identical in both. bat_theme()/bat_theme_dark() build on this via merge
# (Makie's Theme merge is recursive per attribute path with the override
# winning -- verified directly), each adding ONLY what genuinely differs, so
# a styling change made here can't silently drift between the two variants
# (previously ~60 duplicated lines per theme).
function _bat_base_theme()
        return Theme(
                fontsize=20,
                fonts=Attributes(
                        :bold => Makie.texfont(:bold),
                        :bolditalic => Makie.texfont(:bolditalic),
                        :italic => Makie.texfont(:italic),
                        :regular => Makie.texfont(:regular),
                ),
                Axis=(
                        xminorticksvisible=false,
                        yminorticksvisible=false,
                        xticksvisible=true,
                        yticksvisible=true,
                        xlabelpadding=3,
                        ylabelpadding=3,
                        # Small, tight ticks/tick labels -- the corner-grid cells
                        # (ext/makie_impl/makie_gridlayout.jl) are only ~100-150px
                        # square, so full-size (fontsize-inherited) tick labels
                        # overlap each other and eat into the plotting area via
                        # their protrusion. Independent of the global fontsize (used
                        # for titles, the v_i/p_i axis labels, etc.), which is left
                        # untouched.
                        xticklabelsize=10,
                        yticklabelsize=10,
                        xticklabelpad=1,
                        yticklabelpad=1,
                        xticksize=3,
                        yticksize=3,
                ),
                Legend=(
                        framevisible=false,
                        padding=(0, 0, 0, 0),
                ),
                Colorbar=(
                        ticksvisible=false,
                        spinewidth=0,
                        ticklabelpad=5,
                ),
                Heatmap=Theme(
                        colormap=:inferno,
                        alpha=1.0
                ),
                BarPlot=Theme(
                        color=RGB(0.898, 0.361, 0.188),
                        alpha=1.0,
                        gap=0.0
                ),
                # `visible` is variant-specific (the light theme hides the
                # Stairs outline by default, the dark one keeps it) -- see
                # the merge overrides in bat_theme()/bat_theme_dark().
                Stairs=Theme(
                        step=:post,
                        color=:darkblue,
                        linewidth=1.0,
                ),
                Lines=Theme(
                        color=RGB(0.741, 0.518, 0.02),
                        linewidth=1.0,
                        visible=true
                ),
                Poly=Theme(
                        color=RGB(0.898, 0.361, 0.188),
                        alpha=1.0,
                        visible=true
                ),
                Scatter=Theme(
                        color=RGB(0.898, 0.361, 0.188),
                        alpha=1.0
                ),
                # VLines/HLines/LineSegments's color here is NOT actually
                # applied to plots of these types -- confirmed directly: all
                # three have Makie's automatic color-*cycling* enabled by
                # default (a `cycle=[[:color]=>:color]` attribute), and cycling
                # silently overrides any theme-level `color` default rather
                # than the other way around. So the stats-overlay recipes'
                # black color (per explicit request) is set directly on each
                # PlotSpec instead (makie_stats.jl), not via this theme block --
                # left here anyway since linestyle/linewidth *do* still apply
                # via the theme (only `color` is affected by cycling).
                VLines=Theme(
                        color=:dodgerblue,
                        linestyle=:solid,
                        linewidth=2.0
                ),
                HLines=Theme(
                        color=:dodgerblue,
                        linestyle=:solid,
                        linewidth=2.0
                ),
                LineSegments=Theme(
                        color=:dodgerblue,
                        linestyle=:solid,
                        linewidth=2.0
                ),
                Errorbars=Theme(
                        color=:blue,
                        linewidth=2.0,
                        whiskerwidth=10
                ),
                Hexbin=Theme(
                        alpha=1.0
                )
        )
end

function BAT.bat_theme()
        # See _panel_bg_color's docstring (ext/makie_impl/makie_render_utils.jl)
        # for the background -> panel -> widget "shade ladder" this builds:
        # applying the same step twice keeps the panel-vs-page and
        # widget-vs-panel contrasts identical, rather than widgets sitting at
        # Makie's own near-white defaults (0.94/0.97) a single small step from
        # the panel color and barely standing out against it.
        panel_color = _panel_bg_color(RGB(1, 1, 1))
        widget_color = _panel_bg_color(panel_color)

        return merge(_bat_base_theme(), Theme(
                Button=(
                        buttoncolor=widget_color,
                ),
                Menu=(
                        cell_color_inactive_even=widget_color,
                        cell_color_inactive_odd=widget_color,
                        selection_cell_color_inactive=widget_color,
                ),
                Slider=(
                        color_inactive=widget_color,
                ),
                # IntervalSlider is a distinct type from Slider (Makie themes
                # key by exact type name), so it needs its own block even
                # though it shares the same attribute names -- confirmed via
                # propertynames(IntervalSlider(...)).
                IntervalSlider=(
                        color_inactive=widget_color,
                ),
                Toggle=(
                        framecolor_inactive=widget_color,
                ),
                # Checkbox previously had no theme block in either variant,
                # leaving Makie's defaults regardless of theme; the unchecked
                # box now follows the same widget shade ladder as every other
                # panel widget.
                Checkbox=(
                        checkboxcolor_unchecked=widget_color,
                ),
                # Hist1D (and any other recipe drawing a Stairs outline atop
                # its own fill, e.g. QuantileHist1D's stairs_data) doesn't
                # show this outline by default in the light theme -- purely
                # a light-theme cosmetic default, per explicit request; the
                # dark theme keeps visible=true.
                Stairs=(
                        visible=false,
                ),
        ))
end

function BAT.bat_theme_dark()
        # Nice dark purple:
        #color_inactive = RGBf(0.18, 0.039, 0.353)

        color_active = RGB(0.451, 0.102, 0.431)
        color_hover = RGB(0.714, 0.216, 0.322)
        text_color = RGB(0.80, 0.80, 0.80)
        # See bat_theme()'s matching comment / _panel_bg_color's docstring
        # (ext/makie_impl/makie_render_utils.jl) for the background -> panel ->
        # widget shade ladder this builds -- replaces the old hardcoded
        # RGB(0.15, 0.17, 0.20), which put idle widgets barely one step off
        # the panel's own color (too low-contrast for the same reason as the
        # light theme's Makie-default widget colors).
        panel_color = _panel_bg_color(Makie.to_color(:gray10))
        color_inactive = _panel_bg_color(panel_color)

        return merge(_bat_base_theme(), Theme(
                backgroundcolor=:gray10,
                textcolor=:gray80,
                linecolor=:gray70,
                palette=Makie.generate_default_palette(:gray10),
                Axis=(
                        backgroundcolor=:transparent,
                        xgridcolor=:gray50,
                        ygridcolor=:gray50,
                        leftspinecolor=:gray20,
                        rightspinecolor=:gray20,
                        bottomspinecolor=:gray20,
                        topspinecolor=:gray20,
                ),
                Button=(
                        buttoncolor=color_inactive,
                ),
                Menu=(
                        cell_color_active=color_active,
                        cell_color_hover=color_hover,
                        cell_color_inactive_even=RGBf(0.20, 0.20, 0.20),
                        cell_color_inactive_odd=RGBf(0.15, 0.15, 0.15),
                        selection_cell_color_inactive=color_inactive,
                        textcolor=text_color,
                        dropdown_arrow_color=:grey30
                ),
                Slider=(
                        color_active=color_active,
                        color_active_dimmed=color_hover,
                        color_inactive=color_inactive,
                ),
                # See bat_theme()'s matching comment -- IntervalSlider needs
                # its own theme block despite sharing Slider's attribute names.
                IntervalSlider=(
                        color_active=color_active,
                        color_active_dimmed=color_hover,
                        color_inactive=color_inactive,
                ),
                Toggle=(
                        buttoncolor=color_hover,
                        framecolor_active=color_active,
                        framecolor_inactive=color_inactive
                ),
                # Previously unthemed: every unchecked cell of the vsel
                # picker rendered as Makie's default white square against the
                # near-black panel. Colors from the same ladder/accents as
                # the other dark widgets.
                Checkbox=(
                        checkboxcolor_unchecked=color_inactive,
                        checkboxcolor_checked=color_active,
                        checkboxstrokecolor_unchecked=color_active,
                        checkboxstrokecolor_checked=color_active,
                        checkmarkcolor_checked=text_color,
                ),
                Stairs=(
                        visible=true,
                ),
        ))
end

