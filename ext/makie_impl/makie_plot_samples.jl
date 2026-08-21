# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Shared setup of the static (all-samples-already-exist) entry points:
# builds the compute graph, registers the samples, and applies the initial
# variable selection. Deliberately does NOT call warmup_makie_shaders() --
# the precompile workload must stay backend-free, so the entry points call it.
function _setup_static_graph(
    samples::DensitySampleVector,
    recipes::NamedTuple,
    vsel::AbstractVector{<:Integer},
    N_max::Integer,
    triagonal_config::NamedTuple,
    diagonal_config::NamedTuple;
    # Hard prior-support bounds for KDE boundary reflection: a measure/prior/
    # Distribution or per-dim (lo, hi) pairs (see _support_vectors); default: none.
    support = nothing,
)
    graph = _init_compute_graph(recipes, triagonal_config, diagonal_config, N_max)

    unshaped_samples = unshaped.(samples)

    # Registered per (chainid, walkerid): BAT's merged multi-chain result is
    # chain-block-concatenated, so a single shared row window would reveal one
    # chain's entire block before the next -- the step window must apply per walker.
    # Groups are MATERIALIZED (not index views) so each registered walker has the
    # same concrete .v.data-backed structure as a live walker.
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
    recipes::NamedTuple = (upper=QuantileHist2D, diagonal=Hist1D, lower=Hist2D),
    vsel::Vector{<:Integer} = [1, 2, 3],
    N_max::Integer = 3,
    trace_nsteps::Integer = 20,
    support = nothing,
    # This embed path draws under the caller's own theme; `dark` only informs the
    # few cycling-affected colors (config stats_color) that can't come from a theme.
    dark::Bool = false,
)
    # Backend-specific shader warmup -- can only happen once a concrete backend
    # is loaded, so it lives at the entry points, not in _setup_static_graph.
    warmup_makie_shaders()

    graph, _ = _setup_static_graph(
        samples, recipes, vsel, N_max,
        _default_makie_triagonal_config(trace_nsteps=trace_nsteps, dark=dark),
        _default_makie_diagonal_config(dark=dark);
        support=support,
    )

    gridlayout = _init_gridlayout(graph, N_max)

    return gridlayout[]
end


function BAT.bat_makie_plot(
    samples::DensitySampleVector,
    recipes::NamedTuple = (upper=QuantileHist2D, diagonal=Hist1D, lower=Hist2D),
    vsel::Vector{<:Integer} = [1, 2, 3],
    N_max::Integer = 3;
    dark::Bool = false,
    trace_nsteps::Integer = 20,
    # Hard prior-support bounds for KDE boundary reflection -- see _support_vectors.
    support = nothing,
)
    # TODO: MD, Discuss config handling and passing of user attribute overwrites

    # See convert_arguments' matching warmup comment above.
    warmup_makie_shaders()

    graph, n_dof = _setup_static_graph(
        samples, recipes, vsel, N_max,
        _default_makie_triagonal_config(trace_nsteps=trace_nsteps, dark=dark),
        _default_makie_diagonal_config(dark=dark);
        support=support,
    )

    # PickerInfo (concrete, type-erased callback) -- see makie_render_utils.jl.
    picker_info = PickerInfo(
        n_dof,
        N_max,
        vsel,
        new_vsel -> _apply_vsel_to_graph!(graph, n_dof, N_max, new_vsel),
    )

    with_theme(dark ? bat_theme_dark() : bat_theme()) do
        gridlayout = _init_gridlayout(graph, N_max)
        built = _build_fig(graph, gridlayout, picker_info;
            has_chain_info=_samples_have_chain_ids(samples),
            has_trace_info=_samples_have_trace_info(samples))
        _maybe_display(built.fig)
    end
    return nothing
end

# Everything the light and dark themes share; bat_theme()/bat_theme_dark() build
# on this via merge (recursive per attribute path, override wins), adding only what differs.
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
            # Small, tight ticks: corner-grid cells are only ~100-150px square.
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
        # `visible` is variant-specific -- see the merge overrides in the two themes.
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
        # `color` here is overridden by these types' automatic color-cycling;
        # the stats overlays set color per-PlotSpec instead (makie_stats.jl).
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
    # Background -> panel -> widget shade ladder, see _panel_bg_color.
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
        # IntervalSlider is a distinct type from Slider (Makie themes key by
        # exact type name), so it needs its own block.
        IntervalSlider=(
            color_inactive=widget_color,
        ),
        Toggle=(
            framecolor_inactive=widget_color,
        ),
        Checkbox=(
            checkboxcolor_unchecked=widget_color,
        ),
        # Light-theme cosmetic: hide the Stairs outline (the dark theme keeps it).
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
    # Same background -> panel -> widget shade ladder as bat_theme().
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
        # IntervalSlider needs its own block -- see bat_theme().
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

