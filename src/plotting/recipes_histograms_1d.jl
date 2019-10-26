# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function plot_histogram(h::AbstractHistogram, swap::Bool)
    if swap
      return h.weights, h.edges[1][1:end-1]
    else
        h.edges[1][1:end-1], h.weights
    end 
end


# ToDo: This is type piracy, find a cleaner solution!
@recipe function f(hist::Histogram,
                    param::Integer;
                    intervals = standard_confidence_vals, 
                    bins=200,
                    normalize = true, 
                    colors = standard_colors,
                    intervallabels = [])
    
    seriestype = get(plotattributes, :seriestype, :stephist)
    
    orientation = get(plotattributes, :orientation, :vertical)
    (orientation != :vertical) ? swap=true : swap = false
    plotattributes[:orientation] = :vertical # without: auto-scaling of axes not correct

    dims = collect(1:ndims(hist.weights))
    if length(dims) > 1
        weights = vec(sum(hist.weights, dims=setdiff(dims, param)))
        hist = Histogram(hist.edges[param], weights, :left) 
    end

    normalize ? hist=StatsBase.normalize(hist) : nothing
    weights = hist.weights


    if seriestype == :histogram

         @series begin
            seriestype := :steppost
            label --> ""
            fillrange --> 0
            fillcolor --> :dodgerblue
            linewidth --> 0
            plot_histogram(hist, swap)
        end


    elseif seriestype == :stephist || seriestype == :steps #TODO: :steps not working -> "seriestype steppost not supported"
        @series begin
            seriestype := :steppost
            label --> ""
            linecolor --> :dodgerblue
            plot_histogram(hist, swap)
        end


    elseif seriestype == :smallest_intervals      
        hists, orig_hist, realintervals = split_smallest(hist, intervals)

        colors = colors[sortperm(intervals, rev=true)]

        for i in 1:length(realintervals)
            @series begin
                seriestype := :steppost
                fillcolor --> colors[i]
                linewidth --> 0
                fillrange --> 0

                if length(intervallabels) > 0
                    label := intervallabels[i]
                else
                    label := "smallest $(@sprintf("%.2f", realintervals[i]*100))% interval(s)"
                end
                plot_histogram(hists[i], swap)
            end
        end

        @series begin
            seriestype := :steppost
            linecolor --> :black
            linewidth --> 0.7
            label --> ""
            plot_histogram(orig_hist, swap)
        end 


    elseif seriestype == :central_intervals      
        hists, orig_hist, realintervals = split_central(hist, intervals)
         
        colors = colors[sortperm(intervals, rev=true)]

        for i in 1:length(realintervals)
            @series begin
                seriestype := :steppost
                fillcolor --> colors[i]
                linewidth --> 0
                fillrange --> 0
                label --> "central $(@sprintf("%.2f", realintervals[i]*100))% interval(s)"
                plot_histogram(hists[i], swap)
            end
        end
        
         @series begin
            seriestype := :steppost
            linecolor --> :black
            linewidth --> 0.7
            label --> ""
            plot_histogram(orig_hist, swap)
        end 

    else
        error("seriestype $seriestype not supported")
    end 


end






