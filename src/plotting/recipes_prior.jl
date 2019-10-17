# This file is a part of BAT.jl, licensed under the MIT License (MIT).

@recipe function f(prior::NamedPrior, 
    param::Symbol; 
    intervals = standard_confidence_vals, 
    bins=200,
    nsamples=10^6,
    normalize = true, 
    colors = standard_colors,
    intervallabels = [])

    i = findfirst(x -> x == param, keys(prior))

    println(bins)

    @series begin 
        intervals --> intervals
        bins --> bins
        normalize --> normalize
        colors --> colors
        intervallabels --> intervallabels
        nsamples --> nsamples

        prior, i
    end
end


@recipe function f(prior::NamedPrior, 
                    param::Integer; 
                    intervals = standard_confidence_vals, 
                    bins=200,
                    nsamples=10^6,
                    normalize = true, 
                    colors = standard_colors,
                    intervallabels = [])


    r = rand(prior, nsamples)

    orientation = get(plotattributes, :orientation, :vertical)
    (orientation != :vertical) ? swap=true : swap = false
    plotattributes[:orientation] = :vertical # without: auto-scaling of axes not correct

    hist = fit(Histogram, r[param, :], nbins = bins, closed = :left)
    normalize ? hist=StatsBase.normalize(hist) : nothing
    weights = hist.weights

    if swap
        yguide --> "\$\\theta_$(param)\$"
        xguide --> "\$p(\\theta_$(param))\$"
    else 
        yguide --> "\$p(\\theta_$(param))\$"
        xguide --> "\$\\theta_$(param)\$"
    end
    
   @series begin   
    
        seriestype --> :stephist
        linecolor --> :dimgray
        label --> "prior"

        intervals --> intervals
        bins --> bins
        normalize --> normalize
        colors --> colors
        intervallabels --> intervallabels

        hist, param
    end

end


# 2D plots

@recipe function f(prior::NamedPrior, 
    params::NTuple{2,Symbol}; 
    nsamples=10^6,
    intervals = standard_confidence_vals, 
    colors = standard_colors,
    diagonal = Dict(),
    upper = Dict(),
    right = Dict())

    i = findfirst(x -> x == params[1], keys(prior))
    j = findfirst(x -> x == params[2], keys(prior))

    @series begin 
        intervals --> intervals
        colors --> colors
        nsamples --> nsamples
        diagonal -->  diagonal
        upper --> upper
        right --> right

        prior, (i, j)
    end

end


@recipe function f(prior::NamedPrior, 
                params::NTuple{2,Integer}; 
                nsamples=10^6,
                intervals = standard_confidence_vals, 
                colors = standard_colors,
                diagonal = Dict(),
                upper = Dict(),
                right = Dict())


    r = rand(prior, nsamples)

    bins = get(plotattributes, :bins, "default")

    if bins=="default"
        hist = fit(Histogram, (r[params[1], :], r[params[2], :]), closed = :left)
    else
        hist = fit(Histogram, (r[params[1], :], r[params[2], :]), closed = :left, nbins=bins)
    end

    
   @series begin   
        seriestype --> :smallest_intervals_contour
        label --> "prior"

        xguide --> "\$\\theta_$(params[1])\$"
        yguide --> "\$\\theta_$(params[2])\$"

        intervals --> intervals
        nbins --> bins
        colors --> colors
        diagonal -->  diagonal
        upper --> upper
        right --> right

        hist, params
    end

end