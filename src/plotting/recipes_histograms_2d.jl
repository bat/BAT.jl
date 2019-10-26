# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# ToDo: This is type piracy, find a cleaner solution!
@recipe function f(h::Histogram,
                parsel::NTuple{2,Integer}; 
                intervals = standard_confidence_vals, 
                colors = standard_colors,
                diagonal = Dict(),
                upper = Dict(),
                right = Dict())


    pi_x, pi_y = parsel

    seriestype = get(plotattributes, :seriestype, :histogram2d)

    
    if seriestype == :histogram2d || seriestype == :histogram || seriestype == :hist
        _plots_module() != nothing || throw(ErrorException("Package Plots not available, but required for this operation"))
        
        @series begin
            seriestype := :bins2d
            colorbar --> true
            h.edges[1], h.edges[2], _plots_module().Surface(h.weights)
        end

        
    elseif seriestype == :smallest_intervals_contour || seriestype == :smallest_intervals_contourf
        _plots_module() != nothing || throw(ErrorException("Package Plots not available, but required for this operation"))
        
        colors = colors[sortperm(intervals, rev=true)]

        if seriestype == :smallest_intervals_contour
            plotstyle = :contour
        else
            plotstyle = :contourf
        end
 
        lev = calculate_levels(h.weights, intervals)
        x, y, m = bin_centers(h)

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
                color --> colors # only works with pyplot
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


    elseif seriestype == :smallest_intervals
        _plots_module() != nothing || throw(ErrorException("Package Plots not available, but required for this operation"))

        colors = colors[sortperm(intervals, rev=true)]

        hists, orig_hist, realintervals = split_smallest(h, intervals)

       

        for (i, int) in enumerate(realintervals)
            @series begin
                seriestype := :bins2d
                color --> _plots_module().cgrad([colors[i], colors[i]])  
                label --> "smallest $(@sprintf("%.2f", realintervals[i]*100))% interval(s)"
                hists[i].edges[1], hists[i].edges[2], _plots_module().Surface(hists[i].weights)
            end
            # fake a legend
            @series begin
                seriestype := :shape
                fillcolor --> colors[i]
                linewidth --> 0
                label --> "smallest $(@sprintf("%.2f", realintervals[i]*100))% interval(s)"
                colorbar --> false
                [hists[i].edges[1][1], hists[i].edges[1][1]], [hists[i].edges[2][1], hists[i].edges[2][1]]
            end
        end
    

    elseif seriestype == :marginal
        _plots_module() != nothing || throw(ErrorException("Package Plots not available, but required for this operation"))

        size --> (900, 600)

        layout --> _plots_module().grid(2,2, widths=(0.8, 0.2), heights=(0.2, 0.8))
        link --> :both

        if get(diagonal, "seriestype", :histogram) != :histogram
            colorbar --> false
        end

        @series begin
            subplot := 1
            xguide --> "\$p(\\theta_$(pi_x))\$"
            seriestype := get(upper, "seriestype", :histogram)
            bins --> get(upper, "nbins", 200)
            normalize --> get(upper, "normalize", true)
            colors --> get(upper, "colors", standard_colors)
            intervals --> get(upper, "intervals", standard_confidence_vals)
            #mean --> get(upper, "mean", false)
            #std_dev --> get(upper, "std_dev", false)
            #globalmode --> get(upper, "globalmode", false)
            #localmode --> get(upper, "localmode", false)
            h, pi_x
        end
        
        # empty plot (needed since @layout macro not available)
        @series begin
            seriestype := :scatter
            subplot := 2
            grid := false
            xaxis := false
            yaxis := false
            markersize := 0.1
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
            legend --> false

            normalize --> get(diagonal, "normalize", true)
            bins --> get(diagonal, "nbins", 200)
            colors --> get(diagonal, "colors", standard_colors)
            intervals --> get(diagonal, "intervals", standard_confidence_vals)
            #mean --> get(diagonal, "mean", false)
            #std_dev --> get(diagonal, "std_dev", false)
            #globalmode --> get(diagonal, "globalmode", false)
            #localmode --> get(diagonal, "localmode", false)

            h, (pi_x, pi_y) 
        end

        @series begin
            subplot := 4
            seriestype := get(right, "seriestype", :histogram)
            orientation := :horizontal
            xguide --> "\$p(\\theta_$(pi_y))\$"
            normalize --> get(right, "normalize", true)
            bins --> get(right, "nbins", 200)
            colors --> get(right, "colors", standard_colors)
            intervals --> get(right, "intervals", standard_confidence_vals)
            #mean --> get(right, "mean", false)
            #std_dev --> get(right, "std_dev", false)
            #globalmode --> get(right, "globalmode", false)
            #localmode --> get(right, "localmode", false)
            
            h, pi_y
        end 


    else
        error("seriestype $seriestype not supported")
    end

end


#--- rectangle bounds ------------------------
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
