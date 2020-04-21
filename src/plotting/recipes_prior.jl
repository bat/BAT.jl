# This file is a part of BAT.jl, licensed under the MIT License (MIT).

@recipe function f(
    prior::NamedTupleDist,
    param::Union{Integer, Symbol};
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

    param_ind = get_param_index(prior, param)
    bathist = BATHistogram(prior, param_ind, nbins = bins, closed = closed)

    normalize ? bathist.h = StatsBase.normalize(bathist.h) : nothing
    #TODO: names
    if swap
        yguide --> "v$(param)"
        xguide --> "p(v$(param))"
    else
        yguide --> "p(v$(param))"
        xguide --> "v$(param)"
    end

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
    params::Union{NTuple{2,Integer}, NTuple{2,Symbol}};
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

    param_x = get_param_index(prior, params[1])
    param_y = get_param_index(prior, params[2])

    bathist = BATHistogram(
        prior,
        (params[1], params[2]),
        nbins=bins,
        closed=closed,
        nsamples=nsamples
    )

    @series begin
        seriestype --> :smallest_intervals_contour
        label --> "prior"
        #TODO: names
        xguide --> "v$(params[1])"
        yguide --> "v$(params[2])"

        intervals --> intervals
        colors --> colors
        diagonal -->  diagonal
        upper --> upper
        right --> right

        bathist, (1, 2)
    end
end
