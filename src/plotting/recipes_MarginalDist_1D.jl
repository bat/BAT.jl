# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# TODO: add plot without Int for overview?

function plothistogram(h::StatsBase.Histogram, swap::Bool)
    if swap
        return h.weights, h.edges[1][1:end-1]
    else
        return h.edges[1][1:end-1], h.weights
    end
end


@recipe function f(
    origmarg::MarginalDist,
    vsel::Union{Nothing, Integer, Symbol, Expr} = nothing;
    intervals = default_credibilities,
    colors = default_colors,
    interval_labels = []
)   
    if vsel === nothing
        marg = origmarg
    else 
        marg = MarginalDist(origmarg, vsel)
    end
    hist = convert(Histogram, marg.dist.dist)

    orientation = get(plotattributes, :orientation, :vertical)
    (orientation != :vertical) ? swap = true : swap = false
    plotattributes[:orientation] = :vertical # without: auto-scaling of axes not correct

    seriestype = get(plotattributes, :seriestype, :stephist)

    xl = vsel === nothing ? "x" : vsel isa Integer ? "x$(vsel)" : "$vsel"
    yl = vsel === nothing ? "p(x)" : vsel isa Integer ? "p(x$(vsel))" : "p($vsel)"

    xlabel = get(plotattributes, :xguide, xl)
    ylabel = get(plotattributes, :yguide, yl)

    if swap
        xguide := ylabel
        yguide := xlabel
    else
        xguide := xlabel
        yguide := ylabel
    end

    # step histogram
    if seriestype == :stephist || seriestype == :steppost
        @series begin
            seriestype := :steppost
            label --> ""
            linecolor --> :dodgerblue
            plothistogram(hist, swap)
        end

    # filled histogram
    elseif seriestype == :histogram
        @series begin
            seriestype := :steppost
            label --> ""
            fillrange --> 0
            fillcolor --> :dodgerblue
            linewidth --> 0
            plothistogram(hist, swap)
        end


    # smallest intervals aka highest density region (HDR)
    elseif seriestype == :smallest_intervals || seriestype == :HDR
        hists, realintervals = get_smallest_intervals(hist, intervals)
        colors = colors[sortperm(intervals, rev=true)]

        # colored histogram for each interval
        for i in 1:length(realintervals)
            @series begin
                seriestype := :steppost
                fillcolor --> colors[i]
                linewidth --> 0
                fillrange --> 0

                if length(interval_labels) > 0
                    label := interval_labels[i]
                else
                    label := "smallest $(@sprintf("%.2f", realintervals[i]*100))% interval(s)"
                end
                plothistogram(hists[i], swap)
            end
        end

        # black contour line for total histogram
        @series begin
            seriestype := :steppost
            linecolor --> :black
            linewidth --> 0.7
            label --> ""
            plothistogram(hist, swap)
        end


    # central intervals
    elseif seriestype == :central_intervals
        hists, realintervals = split_central(hist, intervals)
        colors = colors[sortperm(intervals, rev=true)]

        # colored histogram for each interval
        for i in 1:length(realintervals)
            @series begin
                seriestype := :steppost
                fillcolor --> colors[i]
                linewidth --> 0
                fillrange --> 0

                if length(interval_labels) > 0
                    label := interval_labels[i]
                else
                    label := "central $(@sprintf("%.2f", realintervals[i]*100))% interval(s)"
                end

                plothistogram(hists[i], swap)
            end
        end

        # black contour line for total histogram
        @series begin
            seriestype := :steppost
            linecolor --> :black
            linewidth --> 0.7
            label --> ""
            plothistogram(hist, swap)
        end

    else
        error("seriestype $seriestype not supported")
    end

end
