# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# 1D
@recipe function f(
    prior::NamedTupleDist,
    parsel::Union{Integer, Symbol, Expr};
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

    idx = asindex(prior, parsel)
    if length(idx) > 1
        throw(ArgumentError("Symbol :$parsel refers to a multivariate parameter. Use :($parsel[i]) instead."))
    end

    marg = bat_marginalize(
        prior,
        idx,
        nbins = bins,
        nsamples = nsamples,
        closed = closed,
        normalize = normalize
    ).result

    xlabel = if isa(parsel, Symbol) || isa(parsel, Expr)
        "$parsel"
    else
        getstring(prior, idx)
    end
    ylabel = "p("*xlabel*")"

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

        marg, idx
    end

end



# 2D plots
@recipe function f(
    prior::NamedTupleDist,
    parsel::Union{NTuple{2,Integer}, NTuple{2,Union{Symbol, Expr, Integer}}};
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

    xidx = asindex(prior, parsel[1])
    yidx = asindex(prior, parsel[2])

    if length(xidx) > 1
        throw(ArgumentError("Symbol :$(parsel[1]) refers to a multivariate parameter. Use :($(parsel[1])[i]) instead."))
    elseif length(yidx) > 1
        throw(ArgumentError("Symbol :$(parsel[2]) refers to a multivariate parameter. Use :($(parsel[2])[i]) instead."))
    end


    marg = bat_marginalize(
        prior,
        (xidx, yidx),
        nbins = bins,
        closed = closed,
        normalize = normalize
    ).result


    xlabel, ylabel = if isa(parsel, Symbol) || isa(parsel, Expr)
        "$(parsel[1])", "$(parsel[2])"
    else
        getstring(prior, xidx), getstring(prior, yidx)
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

        marg, (xidx, yidx)
    end
end
