# This file is a part of BAT.jl, licensed under the MIT License (MIT).

@recipe function f(
    bathist::BATHistogram,
    parsel::NTuple{2,Integer};
    intervals = standard_confidence_vals,
    colors = standard_colors,
    diagonal = Dict(),
    upper = Dict(),
    right = Dict(),
    interval_labels = [],
    normalize = true
)
    _plots_module() != nothing || throw(ErrorException("Package Plots not available, but required for this operation"))

    pi_x, pi_y = parsel

    hist = subhistogram(bathist, [pi_x, pi_y])
    normalize ? hist.h=StatsBase.normalize(hist.h) : nothing

    seriestype = get(plotattributes, :seriestype, :histogram2d)

    xguide --> "x$(pi_x)"
    yguide --> "x$(pi_y)"

    # histogram / heatmap
    if seriestype == :histogram2d || seriestype == :histogram || seriestype == :hist
        @series begin
            seriestype := :bins2d
            colorbar --> true

            hist.h.edges[1], hist.h.edges[2], _plots_module().Surface(hist.h.weights)
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
        x, y = get_bin_centers(hist)
        m = hist.h.weights

        # quick fix: needed when plotting contour on top of histogram
        # otherwise scaling of histogram colorbar would change scaling
        lev = lev/10000
        m = m/10000

        colorbar --> false

        if _plots_module().backend() == _plots_module().PyPlotBackend()
            @series begin
                seriestype := plotstyle
                levels --> lev
                linewidth --> 2
                color --> colors # currently only works with pyplot
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
                color --> _plots_module().cgrad([colors[i], colors[i]])
                label --> "smallest $(@sprintf("%.2f", realintervals[i]*100))% interval(s)"
                hists[i].h.edges[1], hists[i].h.edges[2], _plots_module().Surface(hists[i].h.weights)
            end

            # fake a legend
            @series begin
                seriestype := :shape
                fillcolor --> colors[i]
                linewidth --> 0
                label --> "smallest $(@sprintf("%.2f", realintervals[i]*100))% interval(s)"
                colorbar --> false
                [hists[i].h.edges[1][1], hists[i].h.edges[1][1]], [hists[i].h.edges[2][1], hists[i].h.edges[2][1]]
            end
        end


    # with marginal histograms TODO: xyguides
    elseif seriestype == :marginal
        layout --> _plots_module().grid(2,2, widths=(0.8, 0.2), heights=(0.2, 0.8))
        link --> :both

        if get(diagonal, "seriestype", :histogram) != :histogram
            colorbar --> false
        end

        @series begin
            subplot := 1
            #xguide --> "v$(pi_x)"
            seriestype := get(upper, "seriestype", :histogram)
            bins --> get(upper, "nbins", 200)
            normalize --> get(upper, "normalize", true)
            colors --> get(upper, "colors", standard_colors)
            intervals --> get(upper, "intervals", standard_confidence_vals)
            legend --> get(upper, "legend", true)

            hist, 1
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

            normalize --> get(diagonal, "normalize", true)
            bins --> get(diagonal, "nbins", 200)
            colors --> get(diagonal, "colors", standard_colors)
            intervals --> get(diagonal, "intervals", standard_confidence_vals)
            legend --> get(diagonal, "legend", false)

            hist, (1, 2)
        end

        @series begin
            subplot := 4
            seriestype := get(right, "seriestype", :histogram)
            orientation := :horizontal
            #xguide --> "p(v$(pi_y))"
            normalize --> get(right, "normalize", true)
            bins --> get(right, "nbins", 200)
            colors --> get(right, "colors", standard_colors)
            intervals --> get(right, "intervals", standard_confidence_vals)
            legend --> get(right, "legend", true)

            hist, 2
        end

    else
        error("seriestype $seriestype not supported")
    end

end



# rectangle bounds
@recipe function f(bounds::HyperRectBounds, parsel::NTuple{2,Integer})
    pi_x, pi_y = parsel

    vol = spatialvolume(bounds)
    vhi = vol.hi[[pi_x, pi_y]]; vlo = vol.lo[[pi_x, pi_y]]
    rect_xy = rectangle_path(vlo, vhi)
    bext = 0.1 * (vhi - vlo)
    xlims = (vlo[1] - bext[1], vhi[1] + bext[1])
    ylims = (vlo[2] - bext[2], vhi[2] + bext[2])

    @series begin
        seriestype := :path
        label --> "bounds"
        color --> :darkred
        alpha --> 0.75
        linewidth --> 2
        xlims --> xlims
        ylims --> ylims
        (rect_xy[:,1], rect_xy[:,2])
    end

    nothing
end
