# This file is a part of BAT.jl, licensed under the MIT License (MIT).
@recipe function f(
    marg::MarginalDist,
    vsel::NTuple{2,Union{Symbol, Expr, Integer}};
    intervals = default_credibilities,
    colors = default_colors,
    diagonal = Dict(),
    upper = Dict(),
    right = Dict(),
    interval_labels = [],
    normalize = true
)
    _plots_module() != nothing || throw(ErrorException("Package Plots not available, but required for this operation"))
    hist = convert(Histogram, marg.dist)
    seriestype = get(plotattributes, :seriestype, :histogram2d)

    xlabel = get(plotattributes, :xguide, "x$(vsel[1])")
    ylabel = get(plotattributes, :yguide, "x$(vsel[2])")

    # histogram / heatmap
    if seriestype == :histogram2d || seriestype == :histogram || seriestype == :hist
        @series begin
            seriestype := :bins2d
            xguide --> xlabel
            yguide --> ylabel
            colorbar --> true

            hist.edges[1], hist.edges[2], _plots_module().Surface(hist.weights)
        end


    # smallest interval contours
    elseif seriestype == :smallest_intervals_contour || seriestype == :smallest_intervals_contourf

        colors = colors[sortperm(intervals, rev=true)]

        if seriestype == :smallest_intervals_contour
            plotstyle = :contour
        else
            plotstyle = :contourf
        end

        lev = calculate_levels(hist, intervals)
        x, y = get_bin_centers(marg)
        m = hist.weights

        # quick fix: needed when plotting contour on top of histogram
        # otherwise scaling of histogram colorbar would change scaling
        lev = lev/10000
        m = m/10000

        colorbar --> false
        xguide --> xlabel
        yguide --> ylabel

        if _plots_module().backend() == _plots_module().PyPlotBackend()
            @series begin
                seriestype := plotstyle
                levels --> lev
                linewidth --> 2
                seriescolor --> colors # currently only works with pyplot
                (x, y, m')
            end
        else
             @series begin
                seriestype := plotstyle
                levels --> lev
                linewidth --> 2
                (x, y, m')
            end
        end


    # smallest intervals heatmap
    elseif seriestype == :smallest_intervals
        colors = colors[sortperm(intervals, rev=true)]

        hists, realintervals = get_smallest_intervals(hist, intervals)

        for (i, int) in enumerate(realintervals)
            @series begin
                seriestype := :bins2d
                seriescolor --> _plots_module().cgrad([colors[i], colors[i]])
                xguide --> xlabel
                yguide --> ylabel

                hists[i].edges[1], hists[i].edges[2], _plots_module().Surface(hists[i].weights)
            end

            # fake a legend
            interval_label = isempty(interval_labels) ? "smallest $(@sprintf("%.2f", realintervals[i]*100))% interval(s)" : interval_labels[i]

            @series begin
                seriestype := :shape
                fillcolor --> colors[i]
                linewidth --> 0
                label --> interval_label
                colorbar --> false
                [hists[i].edges[1][1], hists[i].edges[1][1]], [hists[i].edges[2][1], hists[i].edges[2][1]]
            end
        end


    # marginal histograms
    elseif seriestype == :marginal

        layout --> _plots_module().grid(2,2, widths=(0.8, 0.2), heights=(0.2, 0.8))
        link --> :both

        if get(diagonal, "seriestype", :histogram) != :histogram
            colorbar --> false
        end

        @series begin
            subplot := 1
            xguide := xlabel
            yguide := "p("*xlabel*")"
            seriestype := get(upper, "seriestype", :histogram)
            bins --> get(upper, "nbins", 200)
            normalize --> get(upper, "normalize", true)
            colors --> get(upper, "colors", default_colors)
            intervals --> get(upper, "intervals", default_credibilities)
            legend --> get(upper, "legend", true)

            marg, vsel[1]
        end

        # empty plot (needed since @layout macro not available)
        @series begin
            seriestype := :scatter
            subplot := 2
            grid := false
            xaxis := false
            yaxis := false
            markersize := 0.001
            markerstrokewidth := 0
            markeralpha := 1
            markerstrokealpha := 1
            legend := false
            label := ""
            xguide := ""
            yguide := ""
            [(0,0)]
        end

        @series begin
            subplot := 3
            seriestype := get(diagonal, "seriestype", :histogram)
            xguide --> xlabel
            yguide --> ylabel
            normalize --> get(diagonal, "normalize", true)
            bins --> get(diagonal, "nbins", 200)
            colors --> get(diagonal, "colors", default_colors)
            intervals --> get(diagonal, "intervals", default_credibilities)
            legend --> get(diagonal, "legend", false)

            marg, (vsel[1], vsel[2])
        end

        @series begin
            subplot := 4
            seriestype := get(right, "seriestype", :histogram)
            orientation := :horizontal
            xguide := ylabel
            yguide := "p("*ylabel*")"
            normalize --> get(right, "normalize", true)
            bins --> get(right, "nbins", 200)
            colors --> get(right, "colors", default_colors)
            intervals --> get(right, "intervals", default_credibilities)
            legend --> get(right, "legend", true)

            marg, vsel[2]
        end

    else
        error("seriestype $seriestype not supported")
    end

end



# rectangle bounds
@recipe function f(bounds::HyperRectBounds, vsel::NTuple{2,Integer})
    pi_x, pi_y = vsel

    vol = spatialvolume(bounds)
    vhi = vol.hi[[pi_x, pi_y]]; vlo = vol.lo[[pi_x, pi_y]]
    rect_xy = rectangle_path(vlo, vhi)
    bext = 0.1 * (vhi - vlo)
    xlims = (vlo[1] - bext[1], vhi[1] + bext[1])
    ylims = (vlo[2] - bext[2], vhi[2] + bext[2])

    @series begin
        seriestype := :path
        label --> "bounds"
        seriescolor --> :darkred
        seriesalpha --> 0.75
        linewidth --> 2
        xlims --> xlims
        ylims --> ylims
        (rect_xy[:,1], rect_xy[:,2])
    end

    nothing
end
