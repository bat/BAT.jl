




function Makie.convert_arguments(
        ::Type{<:AbstractPlot},
        samples::DensitySampleVector;
        recipes::NamedTuple=(upper=QuantileHist2D, diagonal=Hist1D, lower=Hist2D),
        vsel::Vector{<:Integer}=[1, 2, 3],
        N_max::Integer=3,
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
                color=RGB(0.898, 0.361, 0.188),
                color_stats=:dodgerblue,
                alpha=1.0,
                rev=false,
                threshold=nothing,
                markersize=2.0,
                edge=false,
                strokecolor=RGB(0.741, 0.518, 0.02),
                strokewidth=1.0,
                strokestyle_stats=:solid,
                strokewidth_stats=2.0,
                color_mean=:white,
                strokestyle_mean=:dot,
                strokewidth_mean=2.0,
                color_ebars=:blue,
                whiskerwidth=10
        )

        diagonal_config = (
                weights=nothing,
                nsigma=1.0,
                nbins=100,
                closed=:left,
                normalization=:pdf,
                levels=cdf.(Chi(1), 0:3),
                filter=false,
                color=RGB(0.898, 0.361, 0.188),
                color_stats=:dodgerblue,
                colormap=:inferno,
                alpha=1.0,
                filled=true,
                edge=false,
                strokecolor=:orange,
                strokewidth=1.0,
                strokestyle_stats=:solid,
                strokewidth_stats=2.0,
                strokestyle_mean=:dot,
                strokewidth_mean=2.0,
                y_ebars=0.0,
                color_ebars=:blue,
                whiskerwidth=10,
                filled_pdf=true,
                npoints_pdf=300,
                rev=false
        )

        graph = _init_compute_graph(
                recipes,
                triagonal_config,
                diagonal_config,
                N_max,
        )

        samples_graph = graph[:samples][]
        push!(samples_graph, [unshaped.(samples)])
        update!(graph, samples=samples_graph)

        current_idxs = [length(samples)]
        current_idxs_graph = graph[:current_idxs][]
        push!(current_idxs_graph, current_idxs)
        update!(graph, current_idxs=current_idxs_graph)

        update!(graph, idxs=vsel)

        gridlayout = _init_gridlayout(graph, N_max)

        return gridlayout[]
end


function BAT.bat_makie_plot(
        samples::DensitySampleVector,
        recipes::NamedTuple=(upper=QuantileHist2D, diagonal=Hist1D, lower=Hist2D),
        vsel::Vector{<:Integer}=[1, 2, 3],
        N_max::Integer=3,
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
                color=RGB(0.898, 0.361, 0.188),
                color_stats=:dodgerblue,
                alpha=1.0,
                rev=false,
                threshold=nothing,
                markersize=2.0,
                edge=false,
                strokecolor=RGB(0.741, 0.518, 0.02),
                strokewidth=1.0,
                strokestyle_stats=:solid,
                strokewidth_stats=2.0,
                color_mean=:white,
                strokestyle_mean=:dot,
                strokewidth_mean=2.0,
                color_ebars=:blue,
                whiskerwidth=10
        )

        diagonal_config = (
                weights=nothing,
                nsigma=1.0,
                nbins=100,
                closed=:left,
                normalization=:pdf,
                levels=cdf.(Chi(1), 0:3),
                filter=false,
                color=RGB(0.898, 0.361, 0.188),
                color_stats=:dodgerblue,
                colormap=:inferno,
                alpha=1.0,
                filled=true,
                edge=false,
                strokecolor=:orange,
                strokewidth=1.0,
                strokestyle_stats=:solid,
                strokewidth_stats=2.0,
                strokestyle_mean=:dot,
                strokewidth_mean=2.0,
                y_ebars=0.0,
                color_ebars=:blue,
                whiskerwidth=10,
                filled_pdf=true,
                npoints_pdf=300,
                rev=false
        )

        graph = _init_compute_graph(
                recipes,
                triagonal_config,
                diagonal_config,
                N_max,
        )

        samples_graph = graph[:samples][]
        push!(samples_graph, [unshaped.(samples)])
        update!(graph, samples=samples_graph)

        current_idxs = [length(samples)]
        current_idxs_graph = graph[:current_idxs][]
        push!(current_idxs_graph, current_idxs)
        update!(graph, current_idxs=current_idxs_graph)

        update!(graph, idxs=vsel)

        gridlayout = _init_gridlayout(graph, N_max)
        fig = _build_fig(graph, gridlayout)
        display(fig)

        return nothing
end

function BAT.bat_theme()
        # Nice dark purple:
        #color_inactive = RGBf(0.18, 0.039, 0.353)

        color_active = RGB(0.451, 0.102, 0.431)
        color_inactive = RGB(0.15, 0.17, 0.20)
        color_hover = RGB(0.714, 0.216, 0.322)
        text_color = RGB(0.80, 0.80, 0.80)

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
                        colormap=:inferno
                )
        )
end

