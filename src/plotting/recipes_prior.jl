# This file is a part of BAT.jl, licensed under the MIT License (MIT).

@recipe function f(prior::NamedPrior, 
                    param::Integer; 
                    intervals = standard_confidence_vals, 
                    bins=200,
                    normalize = true, 
                    colors = standard_colors)


    r = rand(prior, 10^6)

    orientation = get(plotattributes, :orientation, :vertical)
    (orientation != :vertical) ? swap=true : swap = false
    plotattributes[:orientation] = :vertical # without: auto-scaling of axes not correct

    hist = fit(Histogram, r[param, :], nbins = bins, closed = :left)
    normalize ? hist=StatsBase.normalize(hist) : nothing
    weights = hist.weights

    if swap
        yguide --> "\$\\pi_$(param)\$"
        xguide --> "\$p(\\pi_$(param))\$"
    else 
        yguide --> "\$p(\\pi_$(param))\$"
        xguide --> "\$\\pi_$(param)\$"
    end
    
   @series begin   
    
        seriestype --> :stephist
        linecolor --> :dimgray
        label --> "prior"

        intervals --> intervals
        bins --> bins
        normalize --> normalize
        colors --> colors

        hist, param
    end

end



@recipe function f(prior::NamedPrior, 
                params::NTuple{2,Integer}; 
                intervals = standard_confidence_vals, 
                colors = standard_colors,
                diagonal = Dict(),
                upper = Dict(),
                right = Dict())


    r = rand(prior, 10^6)

    bins = get(plotattributes, :bins, "default")

    if bins=="default"
        hist = fit(Histogram, (r[params[1], :], r[params[2], :]), closed = :left)
    else
        hist = fit(Histogram, (r[params[1], :], r[params[2], :]), closed = :left, nbins=bins)
    end

    
   @series begin   
        seriestype --> :histogram2d
        label --> "prior"

        xguide --> "\$\\pi_$(params[1])\$"
        yguide --> "\$\\pi_$(params[2])\$"

        intervals --> intervals
        nbins --> bins
        colors --> colors

        hist, params
    end

end