# This file is a part of BAT.jl, licensed under the MIT License (MIT).
@recipe function f(
    parshapes::NamedTupleShape,
    samples::DensitySampleVector, 
    param::Symbol; 
    intervals = standard_confidence_vals, 
    bins=200,
    normalize = true, 
    colors = standard_colors,
    intervallabels = [],
    mean = false,
    std_dev = false,
    globalmode = false,
    localmode = true)

    i = findfirst(x -> x == param, keys(parshapes))

    @series begin
        intervals --> intervals
        bins --> bins
        normalize --> normalize 
        colors --> colors
        intervallabels --> intervallabels
        mean --> mean
        std_dev -->  std_dev
        globalmode --> globalmode
        localmode --> localmode

        samples, i
    end

end


@recipe function f(
    posterior::PosteriorDensity,
    samples::DensitySampleVector, 
    param::Symbol; 
    intervals = standard_confidence_vals, 
    bins=200,
    normalize = true, 
    colors = standard_colors,
    intervallabels = [],
    mean = false,
    std_dev = false,
    globalmode = false,
    localmode = true)

    i = findfirst(x -> x == param, keys(params_shape(posterior)))

    @series begin
        intervals --> intervals
        bins --> bins
        normalize --> normalize 
        colors --> colors
        intervallabels --> intervallabels
        mean --> mean
        std_dev -->  std_dev
        globalmode --> globalmode
        localmode --> localmode

        samples, i
    end

end


@recipe function f(samples::DensitySampleVector, 
                    param::Integer; 
                    intervals = standard_confidence_vals, 
                    bins=200,
                    normalize = true, 
                    colors = standard_colors,
                    intervallabels = [],
                    mean = false,
                    std_dev = false,
                    globalmode = false,
                    localmode = true)
    
    #seriestype = get(plotattributes, :seriestype, :smallest_intervals)
    
    hist = fit(Histogram, flatview(samples.params)[param, :], FrequencyWeights(samples.weight), nbins = bins, closed = :left)
    normalize ? hist=StatsBase.normalize(hist) : nothing
    line_height = maximum(hist.weights)*1.03

    orientation = get(plotattributes, :orientation, :vertical)
    (orientation != :vertical) ? swap=true : swap = false

    stats = MCMCBasicStats(samples) 

    if swap
        yguide --> "\$\\theta_$(param)\$"
        xguide --> "\$p(\\theta_$(param))\$"
    else 
        yguide --> "\$p(\\theta_$(param))\$"
        xguide --> "\$\\theta_$(param)\$"
    end


    @series begin   
        seriestype --> :smallest_intervals

        intervals --> intervals
        bins --> bins
        normalize --> normalize
        colors --> colors
        intervallabels --> intervallabels

        hist, param
    end


    mean_options = convert_to_options(mean)
    globalmode_options = convert_to_options(globalmode)
    localmode_options = convert_to_options(localmode)
    stddev_options = convert_to_options(std_dev)


    if stddev_options != ()  
        Σ_all = stats.param_stats.cov
        Σ = Σ_all[param, param]
        dev = sqrt(Σ)
        meanvalue = stats.param_stats.mean[param]
        @series begin
            seriestype := :shape
            label := get(stddev_options, "label", "std. dev.") 
            linewidth := 0
            fillcolor := get(stddev_options, "fillcolor", :grey)
            fillalpha := get(stddev_options, "fillalpha", 0.5)
            uncertaintyband(meanvalue, dev, line_height, swap=swap)
        end
    end

    if mean_options != ()  
        meanvalue = stats.param_stats.mean[param]
        @series begin
            seriestype := :line
            label := get(mean_options, "label", "mean") # : $(@sprintf("%.2f", meanvalue))
            linestyle := get(mean_options, "linestyle", :solid)
            linecolor := get(mean_options, "linecolor", :dimgrey)
            linewidth := get(mean_options, "linewidth", 1)
            alpha := get(mean_options, "alpha", 1)

            line(meanvalue, line_height, swap=swap)
        end
    end

   if globalmode_options != ()  
        globalmode_value = stats.mode[param]
         @series begin
            seriestype := :line
            label := get(globalmode_options, "label", "global mode") #: $(@sprintf("%.2f", globalmode_value))
            linestyle := get(globalmode_options, "linestyle", :dash)
            linecolor := get(globalmode_options, "linecolor", :black)
            linewidth := get(globalmode_options, "linewidth", 1)
            alpha := get(globalmode_options, "alpha", 1)

            line(globalmode_value, line_height, swap=swap)
        end
    end


    if localmode_options != ()  
        localmode_values = calculate_localmode(hist)
        for (i, l) in enumerate(localmode_values)
         @series begin
            seriestype := :line
            if i==1 && length(localmode_values)==1
                label := get(localmode_options, "label", "local mode") #: $(@sprintf("%.2f", l))
            elseif i ==1
                label := get(localmode_options, "label", "local modes")
            else 
                label :=""
            end
            linestyle := get(localmode_options, "linestyle", :dot)
            linecolor := get(localmode_options, "linecolor", :black)
            linewidth := get(localmode_options, "linewidth", 1)
            alpha := get(localmode_options, "alpha", 1)
            line(l, line_height, swap=swap)
            end
        end
    end

end



function line(pos, height; swap=false)
    if swap
        return  [(0, pos), (height, pos)]
    else
        return  [(pos, 0), (pos, height)]
    end
end


function convert_to_options(arg::Dict)
    return arg
end


function convert_to_options(arg::Bool)
    arg ? Dict() : ()
end


function uncertaintyband(m, u, h; swap=false)
    if _plots_module() != nothing
        if swap
            return _plots_module().Shape([0,0,h,h], [m-u,m+u,m+u,m-u])
        else
            return _plots_module().Shape([m-u,m+u,m+u,m-u], [0,0,h,h])
        end
    else
        ()
    end
end



