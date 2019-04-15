# This file is a part of BAT.jl, licensed under the MIT License (MIT).

@recipe function f(samples::DensitySampleVector, 
                parsel::NTuple{2,Integer}; 
                intervals = standard_confidence_vals, 
                colors = standard_colors,
                mean = false,
                std_dev = false,
                globalmode = false,
                localmode = true,
                diagonal = Dict(),
                upper = Dict(),
                right = Dict())

    pi_x, pi_y = parsel
    bins = get(plotattributes, :bins, "default")
    seriestype = get(plotattributes, :seriestype, :smallest_intervals)

    xguide --> "\$\\theta_$(pi_x)\$"
    yguide --> "\$\\theta_$(pi_y)\$"

    if bins=="default"
            h = fit(Histogram, (flatview(samples.params)[pi_x, :], flatview(samples.params)[pi_y, :]), FrequencyWeights(samples.weight), closed=:left)
        else
            h = fit(Histogram, (flatview(samples.params)[pi_x, :], flatview(samples.params)[pi_y, :]), FrequencyWeights(samples.weight), closed=:left, nbins=bins)
        end


    if seriestype == :scatter
        base_markersize = get(plotattributes, :markersize, 1.5)

        acc = findall(x -> x > 0, samples.weight)
        rej = findall(x -> x <= 0, samples.weight)
    
        color = parse(RGBA{Float64}, get(plotattributes, :seriescolor, :green))
        label = get(plotattributes, :label, isempty(rej) ? "samples" : "accepted")

        @series begin
            seriestype := :scatter
            label := label
            markersize := [w < 1 ? base_markersize : base_markersize * sqrt(w) for w in samples.weight[acc]]
            markerstrokewidth := 0
            color := [w >= 1 ? color : RGBA(convert(RGB, color), color.alpha * w) for w in samples.weight[acc]]
            (flatview(samples.params)[pi_x, acc], flatview(samples.params)[pi_y, acc])
        end

        if !isempty(rej)
            @series begin
                seriestype := :scatter
                label := "rejected"
                markersize := base_markersize
                markerstrokewidth := 0
                color := :red
                (flatview(samples.params)[pi_x, rej], flatview(samples.params)[pi_y, rej])
            end
        end

    
    elseif seriestype == :histogram2d || seriestype == :histogram
        _plots_module() != nothing || throw(ErrorException("Package Plots not available, but required for this operation"))
        
        @series begin
            seriestype := :bins2d
            colorbar --> true
            h.edges[1], h.edges[2], _plots_module().Surface(h.weights)
        end

    elseif seriestype == :smallest_intervals_contour || seriestype == :smallest_intervals_contourf
        
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

        @series begin
            seriestype := plotstyle
            levels --> lev
            linewidth --> 2
            color --> colors
            (x, y, m')
        end


    elseif seriestype == :smallest_intervals
        _plots_module() != nothing || throw(ErrorException("Package Plots not available, but required for this operation"))

        colors = colors[sortperm(intervals, rev=true)]

        hists, orig_hist, realintervals = split_smallest(h, intervals)

        colorbar --> false

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
            mean --> get(upper, "mean", false)
            std_dev --> get(upper, "std_dev", false)
            globalmode --> get(upper, "globalmode", globalmode)
            localmode --> get(upper, "localmode", false)

            samples, pi_x
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
            mean --> get(diagonal, "mean", false)
            std_dev --> get(diagonal, "std_dev", false)
            globalmode --> get(diagonal, "globalmode", false)
            localmode --> get(diagonal, "localmode", false)

            samples, (pi_x, pi_y) 
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
            mean --> get(right, "mean", false)
            std_dev --> get(right, "std_dev", false)
            globalmode --> get(right, "globalmode", false)
            localmode --> get(right, "localmode", false)
            samples, pi_y
        end 


    else
        error("seriestype $seriestype not supported")


    end


    #------ stats -----------------------------
    stats = MCMCBasicStats(samples) 

    mean_options = convert_to_options(mean)
    globalmode_options = convert_to_options(globalmode)
    localmode_options = convert_to_options(localmode)
    stddev_options = convert_to_options(std_dev)


    if mean_options != ()  
        mx= stats.param_stats.mean[pi_x]
        my = stats.param_stats.mean[pi_y]

        Σ_all = stats.param_stats.cov
        Σx = Σ_all[pi_x, pi_x]
        Σy = Σ_all[pi_y, pi_y]

        @series begin
            seriestype := :scatter
            label := get(mean_options, "label", "mean") #: ($(@sprintf("%.2f", mx)), $(@sprintf("%.2f", my)))
            seriestype==:marginal ? subplot := 3 : 
            markeralpha := get(mean_options, "markeralpha", 1)
            markercolor := get(mean_options, "markercolor", :black)
            markersize := get(mean_options, "markersize", 4)
            markershape := get(mean_options, "markershape", :circle)
            markerstrokealpha := get(mean_options, "markerstrokealpha", 1)
            markerstrokecolor := get(mean_options, "markerstrokecolor", :black)
            markerstrokestyle := get(mean_options, "markerstrokestyle", :solid)
            markerstrokewidth := get(mean_options, "markerstrokewidth", 1)
            if(std_dev)
                xerror := sqrt(Σx)
                yerror := sqrt(Σy)
            end
           ([mx], [my])
        end
    end


   if globalmode_options != ()  
        globalmode_x = stats.mode[pi_x]
        globalmode_y = stats.mode[pi_y]

        @series begin
            seriestype := :scatter
            label := get(globalmode_options, "label", "global mode") #: ($(@sprintf("%.2f", globalmode_x)), $(@sprintf("%.2f", globalmode_y)))
            seriestype==:marginal ? subplot := 3 : 
            markeralpha := get(globalmode_options, "markeralpha", 1)
            markercolor := get(globalmode_options, "markercolor", :black)
            markersize := get(globalmode_options, "markersize", 4)
            markershape := get(globalmode_options, "markershape", :rect)
            markerstrokealpha := get(globalmode_options, "markerstrokealpha", 1)
            markerstrokecolor := get(globalmode_options, "markerstrokecolor", :black)
            markerstrokestyle := get(globalmode_options, "markerstrokestyle", :solid)
            markerstrokewidth := get(globalmode_options, "markerstrokewidth", 1)
           ([globalmode_x], [globalmode_y])
        end
    end


    if localmode_options != ()  
        localmode_values = calculate_localmode_2d(h)
        for (i, l) in enumerate(localmode_values)
         @series begin
            seriestype := :scatter
            if i==1 && length(localmode_values)==1
                label := get(localmode_options, "label", "local mode") #: ($(@sprintf("%.2f", l[1])), $(@sprintf("%.2f", l[2])))
            elseif i ==1
                label := get(localmode_options, "label", "local modes")
            else 
                label :=""
            end

            seriestype == :marginal ? subplot := 3 : 
            markeralpha := get(localmode_options, "markeralpha", 0)
            markercolor := get(localmode_options, "markercolor", :black)
            markersize := get(localmode_options, "markersize", 4)
            markershape := get(localmode_options, "markershape", :rect)
            markerstrokealpha := get(localmode_options, "markerstrokealpha", 1)
            markerstrokecolor := get(localmode_options, "markerstrokecolor", :black)
            markerstrokestyle := get(localmode_options, "markerstrokestyle", :solid)
            markerstrokewidth := get(localmode_options, "markerstrokewidth", 1)
            ([localmode_values[i][1]], [localmode_values[i][2]])
            end
        end
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
