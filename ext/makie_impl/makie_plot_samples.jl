# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function Makie.convert_arguments(
        ::Type{<:AbstractPlot},
        samples::DensitySampleVector;
        recipes::NamedTuple=(upper=QuantileHist2D, diagonal=Hist1D, lower=Hist2D),
        vsel::Vector{<:Integer}=[1, 2, 3],
        N_max::Integer=3,
        trace_nsteps::Integer=20,
)
        triagonal_config = (
                weights=nothing,
                nsigma=1.0,
                nbins=(100, 100),
                closed=:left,
                normalization=:pdf,
                levels=cdf.(Chi(2), 0:3),
                filter=false,
                colormap=:inferno,
                alpha=1.0,
                rev=false,
                threshold=nothing,
                markersize=2.0,
                trace_nsteps=trace_nsteps
        )

        diagonal_config = (
                weights=nothing,
                nsigma=1.0,
                nbins=100,
                closed=:left,
                normalization=:pdf,
                levels=cdf.(Chi(1), 0:3),
                filter=false,
                colormap=:inferno,
                alpha=1.0,
                y_ebars=0.0,
                filled_pdf=true,
                npoints_pdf=300,
                rev=false
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

        graph = _init_compute_graph(
                recipes,
                triagonal_config,
                diagonal_config,
                N_max,
        )

        unshaped_samples = unshaped.(samples)

        samples_graph = graph[:samples][]
        push!(samples_graph, [unshaped_samples])
        update!(graph, samples=samples_graph)

        current_idxs = [length(samples)]
        current_idxs_graph = graph[:current_idxs][]
        push!(current_idxs_graph, current_idxs)
        update!(graph, current_idxs=current_idxs_graph)

        # All samples already exist here (static plot, no further updates) --
        # use the true data range directly rather than the prior-based
        # estimate the live path needs (see _domain_from_samples's docs).
        n_dof = totalndof(varshape(samples))
        domain_lo, domain_hi = _domain_from_samples(unshaped_samples.v.data, n_dof)
        update!(graph, domain_lo=domain_lo, domain_hi=domain_hi)

        update!(graph, idxs=_clamp_vsel(vsel, samples, N_max))

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
)
        # TODO: MD, Discuss config handling and passing of user attribute overwrites
        triagonal_config = (
                weights=nothing,
                nsigma=1.0,
                nbins=(100, 100),
                closed=:left,
                normalization=:pdf,
                levels=cdf.(Chi(2), 0:3),
                filter=false,
                colormap=:inferno,
                alpha=1.0,
                rev=false,
                threshold=nothing,
                markersize=2.0,
                trace_nsteps=trace_nsteps
        )

        diagonal_config = (
                weights=nothing,
                nsigma=1.0,
                nbins=100,
                closed=:left,
                normalization=:pdf,
                levels=cdf.(Chi(1), 0:3),
                filter=false,
                colormap=:inferno,
                alpha=1.0,
                y_ebars=0.0,
                filled_pdf=true,
                npoints_pdf=300,
                rev=false
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

        graph = _init_compute_graph(
                recipes,
                triagonal_config,
                diagonal_config,
                N_max,
        )

        unshaped_samples = unshaped.(samples)

        samples_graph = graph[:samples][]
        push!(samples_graph, [unshaped_samples])
        update!(graph, samples=samples_graph)

        current_idxs = [length(samples)]
        current_idxs_graph = graph[:current_idxs][]
        push!(current_idxs_graph, current_idxs)
        update!(graph, current_idxs=current_idxs_graph)

        n_dof = totalndof(varshape(samples))

        # All samples already exist here (static plot, no further updates) --
        # use the true data range directly rather than the prior-based
        # estimate the live path needs (see _domain_from_samples's docs).
        domain_lo, domain_hi = _domain_from_samples(unshaped_samples.v.data, n_dof)
        update!(graph, domain_lo=domain_lo, domain_hi=domain_hi)

        update!(graph, idxs=_clamp_vsel(vsel, n_dof, N_max))

        picker_info = (
                N=n_dof,
                N_max=N_max,
                initial_vsel=vsel,
                apply_vsel! = new_vsel -> _apply_vsel_to_graph!(graph, n_dof, N_max, new_vsel),
        )

        with_theme(dark ? bat_theme_dark() : bat_theme()) do
                gridlayout = _init_gridlayout(graph, N_max)
                fig = _build_fig(graph, gridlayout, picker_info; has_chain_info=_samples_have_chain_ids(samples))
                display(fig)
        end
        return nothing
end

function BAT.bat_theme()
        # See _panel_bg_color's docstring (ext/makie_impl/makie_visualizer.jl)
        # for the background -> panel -> widget "shade ladder" this builds:
        # applying the same step twice keeps the panel-vs-page and
        # widget-vs-panel contrasts identical, rather than widgets sitting at
        # Makie's own near-white defaults (0.94/0.97) a single small step from
        # the panel color and barely standing out against it.
        panel_color = _panel_bg_color(RGB(1, 1, 1))
        widget_color = _panel_bg_color(panel_color)

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
                        # (ext/makie_impl/makie_visualizer.jl) are only ~100-150px
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
                Toggle=(
                        framecolor_inactive=widget_color,
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
                Stairs=Theme(
                        step=:post,
                        color=:darkblue,
                        linewidth=1.0,
                        visible=true
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

function BAT.bat_theme_dark()
        # Nice dark purple:
        #color_inactive = RGBf(0.18, 0.039, 0.353)

        color_active = RGB(0.451, 0.102, 0.431)
        color_hover = RGB(0.714, 0.216, 0.322)
        text_color = RGB(0.80, 0.80, 0.80)
        # See bat_theme()'s matching comment / _panel_bg_color's docstring
        # (ext/makie_impl/makie_visualizer.jl) for the background -> panel ->
        # widget shade ladder this builds -- replaces the old hardcoded
        # RGB(0.15, 0.17, 0.20), which put idle widgets barely one step off
        # the panel's own color (too low-contrast for the same reason as the
        # light theme's Makie-default widget colors).
        panel_color = _panel_bg_color(Makie.to_color(:gray10))
        color_inactive = _panel_bg_color(panel_color)

        return Theme(
                backgroundcolor=:gray10,
                textcolor=:gray80,
                linecolor=:gray70,
                palette=Makie.generate_default_palette(:gray10),
                fontsize=20,
                fonts=Attributes(
                        :bold => Makie.texfont(:bold),
                        :bolditalic => Makie.texfont(:bolditalic),
                        :italic => Makie.texfont(:italic),
                        :regular => Makie.texfont(:regular),
                ),
                Axis=(
                        backgroundcolor=:transparent,
                        xgridcolor=:gray50,
                        ygridcolor=:gray50,
                        # leftspinevisible=false,
                        # rightspinevisible=false,
                        # bottomspinevisible=false,
                        # topspinevisible=false,
                        leftspinecolor=:gray20,
                        rightspinecolor=:gray20,
                        bottomspinecolor=:gray20,
                        topspinecolor=:gray20,
                        xminorticksvisible=false,
                        yminorticksvisible=false,
                        xticksvisible=true,
                        yticksvisible=true,
                        xlabelpadding=3,
                        ylabelpadding=3,
                        # See bat_theme()'s matching comment.
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
                Toggle=(
                        buttoncolor=color_hover,
                        framecolor_active=color_active,
                        framecolor_inactive=color_inactive
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
                Stairs=Theme(
                        step=:post,
                        color=:darkblue,
                        linewidth=1.0,
                        visible=true
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

