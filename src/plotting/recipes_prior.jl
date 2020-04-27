# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# 1D
@recipe function f(
    prior::NamedTupleDist,
    parsel::Union{Integer, Symbol};
    intervals = standard_confidence_vals,
    bins = 200,
    nsamples = 10^6,
    normalize = true,
    colors = standard_colors,
    interval_labels = [],
    closed = :left
)

    orientation = get(plotattributes, :orientation, :vertical)
    (orientation != :vertical) ? swap=true : swap = false
    plotattributes[:orientation] = :vertical # without: auto-scaling of axes not correct

    indx = asindex(prior, parsel)
    bathist = BATHistogram(prior, indx, nbins = bins, closed = closed)
    normalize ? bathist.h = StatsBase.normalize(bathist.h) : nothing

    xlabel, ylabel  = if isa(parsel, Symbol)
        "$parsel", "p($parsel)"
    else
        String(keys(prior)[indx]), "p("*String(keys(prior)[indx])*")"
    end

    if swap
        xlabel, ylabel = ylabel, xlabel
    end

    xguide := get(plotattributes, :xguide, xlabel)
    yguide := get(plotattributes, :yguide, ylabel)


   @series begin
        seriestype --> :stephist
        linecolor --> :dimgray
        label --> "prior"

        intervals --> intervals
        bins --> bins
        normalize --> normalize
        colors --> colors
        interval_labels --> interval_labels

        bathist, 1
    end

end



# 2D plots
@recipe function f(
    prior::NamedTupleDist,
    parsel::Union{NTuple{2,Integer}, NTuple{2,Symbol}};
    nsamples=10^6,
    intervals = standard_confidence_vals,
    bins = 200,
    nsamples = 10^6,
    normalize = true,
    colors = standard_colors,
    interval_labels = [],
    closed = :left,
    diagonal = Dict(),
    upper = Dict(),
    right = Dict()
)

    xindx = asindex(prior, parsel[1])
    yindx = asindex(prior, parsel[2])

    bathist = BATHistogram(
        prior,
        (xindx, yindx),
        nbins=bins,
        closed=closed,
        nsamples=nsamples
    )

    xlabel, ylabel  = if isa(parsel, Symbol)
        "$(parsel[1])", "$(parsel[2])"
    else
        String(keys(prior)[xindx]), String(keys(prior)[yindx])
    end

    xguide := get(plotattributes, :xguide, xlabel)
    yguide := get(plotattributes, :yguide, ylabel)

    @series begin
        seriestype --> :smallest_intervals_contour
        label --> "prior"
        intervals --> intervals
        colors --> colors
        diagonal -->  diagonal
        upper --> upper
        right --> right

        bathist, (1, 2)
    end
end
