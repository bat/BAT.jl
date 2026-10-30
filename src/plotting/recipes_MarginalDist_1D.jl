# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function plothistogram(h::StatsBase.Histogram, swap::Bool; filled::Bool = false)
    edges = h.edges[1]
    n = length(h.weights)
    x = Vector{eltype(edges)}(undef, 2n)
    y = Vector{eltype(h.weights)}(undef, 2n)
    for i in eachindex(h.weights)
        x[2i - 1], x[2i] = edges[i], edges[i + 1]
        y[2i - 1] = y[2i] = h.weights[i]
    end
    if filled && n > 0
        # Close against the density-zero baseline before swapping coordinates.
        # Zero-height bins keep disconnected regions separated along that baseline.
        x = [first(edges); x; last(edges); first(edges)]
        y = [zero(eltype(y)); y; zero(eltype(y)); zero(eltype(y))]
    end
    return swap ? (y, x) : (x, y)
end

function _validate_histogram_region_metadata(intervals, colors, labels)
    length(colors) >= length(intervals) || throw(ArgumentError(
        "Histogram regions require one color per requested interval"))
    isempty(labels) || length(labels) >= length(intervals) || throw(ArgumentError(
        "Histogram regions require one custom label per requested interval"))
    return nothing
end

function _histogram_region_metadata(intervals, colors, labels;
    order = sortperm(intervals; rev=true, alg=Base.Sort.MergeSort))
    _validate_histogram_region_metadata(intervals, colors, labels)
    return (colors=[colors[i] for i in order],
        labels=isempty(labels) ? nothing : [labels[i] for i in order])
end

function _histogram_region_label(content; central::Bool = false)
    kind = central ? "central" : "highest-density"
    return "whole-bin $kind $(@sprintf("%.2f", 100 * content))% region"
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
    hist = convert(Histogram, marg.dist isa ReshapedDist ? marg.dist.dist : marg.dist)

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
            seriestype := :path
            label --> ""
            linecolor --> :dodgerblue
            plothistogram(hist, swap)
        end

    # filled histogram
    elseif seriestype == :histogram
        @series begin
            seriestype := :shape
            label --> ""
            fillrange := nothing
            fillcolor --> :dodgerblue
            linewidth --> 0
            plothistogram(hist, swap; filled=true)
        end


    # smallest intervals aka highest density region (HDR)
    elseif seriestype == :smallest_intervals || seriestype == :HDR
        metadata = _histogram_region_metadata(intervals, colors, interval_labels)
        hists, realintervals = get_smallest_intervals(hist, intervals)

        # colored histogram for each interval
        for i in 1:length(realintervals)
            @series begin
                seriestype := :shape
                fillcolor --> metadata.colors[i]
                linewidth --> 0
                fillrange := nothing
                label := metadata.labels === nothing ?
                    _histogram_region_label(realintervals[i]) : metadata.labels[i]
                plothistogram(hists[i], swap; filled=true)
            end
        end

        # black contour line for total histogram
        @series begin
            seriestype := :path
            linecolor --> :black
            linewidth --> 0.7
            label --> ""
            plothistogram(hist, swap)
        end


    # central intervals
    elseif seriestype == :central_intervals
        metadata = _histogram_region_metadata(intervals, colors, interval_labels)
        hists, realintervals = split_central(hist, intervals)

        # colored histogram for each interval
        for i in 1:length(realintervals)
            @series begin
                seriestype := :shape
                fillcolor --> metadata.colors[i]
                linewidth --> 0
                fillrange := nothing
                label := metadata.labels === nothing ?
                    _histogram_region_label(realintervals[i]; central=true) : metadata.labels[i]
                plothistogram(hists[i], swap; filled=true)
            end
        end

        # black contour line for total histogram
        @series begin
            seriestype := :path
            linecolor --> :black
            linewidth --> 0.7
            label --> ""
            plothistogram(hist, swap)
        end

    else
        error("seriestype $seriestype not supported")
    end

end
