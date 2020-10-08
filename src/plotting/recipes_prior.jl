# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# 1D
@recipe function f(
    prior::NamedTupleDist,
    vsel::Union{Integer, Symbol, Expr};
    intervals = default_credibilities,
    bins = 200,
    nsamples = 10^6,
    colors = default_colors,
    interval_labels = [],
    closed = :left
)
    orientation = get(plotattributes, :orientation, :vertical)
    (orientation != :vertical) ? swap=true : swap = false
    plotattributes[:orientation] = :vertical # without: auto-scaling of axes not correct

    idx = asindex(prior, vsel)
    if length(idx) > 1
        throw(ArgumentError("Symbol :$vsel refers to a multivariate parameter. Use :($vsel[i]) instead."))
    end

    marg = get_marginal_dist(
        prior,
        idx,
        bins = bins,
        nsamples = nsamples,
        closed = closed
    ).result

    xlabel = if isa(vsel, Symbol) || isa(vsel, Expr)
        "$vsel"
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
        normalize --> true
        colors --> colors
        interval_labels --> interval_labels

        marg, idx
    end

end



# 2D plots
@recipe function f(
    prior::NamedTupleDist,
    vsel::Union{NTuple{2,Integer}, NTuple{2,Union{Symbol, Expr, Integer}}};
    nsamples=10^6,
    intervals = default_credibilities,
    bins = 200,
    nsamples = 10^6,
    colors = default_colors,
    interval_labels = [],
    closed = :left,
    diagonal = Dict(),
    upper = Dict(),
    right = Dict()
)

    xidx = asindex(prior, vsel[1])
    yidx = asindex(prior, vsel[2])

    if length(xidx) > 1
        throw(ArgumentError("Symbol :$(vsel[1]) refers to a multivariate parameter. Use :($(vsel[1])[i]) instead."))
    elseif length(yidx) > 1
        throw(ArgumentError("Symbol :$(vsel[2]) refers to a multivariate parameter. Use :($(vsel[2])[i]) instead."))
    end


    marg = get_marginal_dist(
        prior,
        (xidx, yidx),
        bins = bins,
        closed = closed
    ).result


    xlabel, ylabel = if isa(vsel, Symbol) || isa(vsel, Expr)
        "$(vsel[1])", "$(vsel[2])"
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
