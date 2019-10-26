# This file is a part of BAT.jl, licensed under the MIT License (MIT).
@recipe function f(
    parshapes::NamedTupleShape,
    samples::DensitySampleVector, 
    parsel::NTuple{2,Symbol}; 
    intervals = standard_confidence_vals, 
    colors = standard_colors,
    mean = false,
    std_dev = false,
    globalmode = false,
    localmode = true,
    diagonal = Dict(),
    upper = Dict(),
    right = Dict(),
    filter = true
)
    i = findfirst(x -> x == parsel[1], keys(parshapes))
    j = findfirst(x -> x == parsel[2], keys(parshapes))

    @series begin
        intervals --> intervals
        colors --> colors
        mean --> mean
        std_dev --> std_dev
        globalmode --> globalmode
        localmode --> localmode
        diagonal --> diagonal
        upper --> upper
        right --> right
        filter --> filter

        samples, (i,j)
    end
end


@recipe function f(
    posterior::PosteriorDensity,
    samples::DensitySampleVector, 
    parsel::NTuple{2,Symbol}; 
    intervals = standard_confidence_vals, 
    colors = standard_colors,
    mean = false,
    std_dev = false,
    globalmode = false,
    localmode = true,
    diagonal = Dict(),
    upper = Dict(),
    right = Dict(),
    filter = true
)

    i = findfirst(x -> x == parsel[1], keys(params_shape(posterior)))
    j = findfirst(x -> x == parsel[2], keys(params_shape(posterior)))

    @series begin
        intervals --> intervals
        colors --> colors
        mean --> mean
        std_dev --> std_dev
        globalmode --> globalmode
        localmode --> localmode
        diagonal --> diagonal
        upper --> upper
        right --> right
        filter --> filter

        samples, (i,j)
    end
end


@recipe function f(
    samples::DensitySampleVector,
    parsel::NTuple{2,Integer};
    intervals = standard_confidence_vals,
    colors = standard_colors,
    mean = false,
    std_dev = false,
    globalmode = false,
    localmode = true,
    diagonal = Dict(),
    upper = Dict(),
    right = Dict(),
    filter = true
)
    if filter
        samples = BAT.drop_low_weight_samples(samples)
    end

    pi_x, pi_y = parsel
    bins = get(plotattributes, :bins, 200)
    seriestype = get(plotattributes, :seriestype, :smallest_intervals)
    
    if seriestype == :histogram2d || seriestype == :histogram || seriestype == :hist
        colorbar = true
    else
        colorbar = false
    end

    xguide --> "\$\\theta_$(pi_x)\$"
    yguide --> "\$\\theta_$(pi_y)\$"


    h = fit(Histogram, (flatview(samples.params)[pi_x, :], flatview(samples.params)[pi_y, :]), FrequencyWeights(samples.weight), closed=:left, nbins=bins)


    if seriestype == :scatter
        base_markersize = get(plotattributes, :markersize, 1.5)

        acc = findall(x -> x > 0, samples.weight)
        rej = findall(x -> x <= 0, samples.weight)
    
        color = parse(RGBA{Float64}, get(plotattributes, :seriescolor, :green))
        label = get(plotattributes, :label, isempty(rej) ? "samples" : "accepted")

        @series begin
            seriestype := :scatter
            label := label
            markersize := [w < 1 ? base_markersize : base_markersize * sqrt(w) for w in samples.weight[acc]]
            markerstrokewidth := 0
            color := [w >= 1 ? color : RGBA(convert(RGB, color), color.alpha * w) for w in samples.weight[acc]]
            (flatview(samples.params)[pi_x, acc], flatview(samples.params)[pi_y, acc])
        end

        if !isempty(rej)
            @series begin
                seriestype := :scatter
                label := "rejected"
                markersize := base_markersize
                markerstrokewidth := 0
                color := :red
                (flatview(samples.params)[pi_x, rej], flatview(samples.params)[pi_y, rej])
            end
        end

    else

        @series begin

            seriestype --> seriestype
            intervals --> intervals
            colors --> colors
            diagonal --> diagonal
            upper --> upper
            right --> right

            h, (pi_x, pi_y)

        end

    end



    #------ stats -----------------------------
    stats = MCMCBasicStats(samples) 

    mean_options = convert_to_options(mean)
    globalmode_options = convert_to_options(globalmode)
    localmode_options = convert_to_options(localmode)
    stddev_options = convert_to_options(std_dev)


    if mean_options != ()  
        mx= stats.param_stats.mean[pi_x]
        my = stats.param_stats.mean[pi_y]

        Σ_all = stats.param_stats.cov
        Σx = Σ_all[pi_x, pi_x]
        Σy = Σ_all[pi_y, pi_y]

        @series begin
            seriestype := :scatter
            label := get(mean_options, "label", "mean") #: ($(@sprintf("%.2f", mx)), $(@sprintf("%.2f", my)))
            seriestype==:marginal ? subplot := 3 : 
            markeralpha := get(mean_options, "markeralpha", 1)
            markercolor := get(mean_options, "markercolor", :black)
            markersize := get(mean_options, "markersize", 4)
            markershape := get(mean_options, "markershape", :circle)
            markerstrokealpha := get(mean_options, "markerstrokealpha", 1)
            markerstrokecolor := get(mean_options, "markerstrokecolor", :black)
            markerstrokestyle := get(mean_options, "markerstrokestyle", :solid)
            markerstrokewidth := get(mean_options, "markerstrokewidth", 1)
            colorbar := colorbar

            if(std_dev)
                xerror := sqrt(Σx)
                yerror := sqrt(Σy)
            end
           ([mx], [my])
        end
    end


   if globalmode_options != ()  
        globalmode_x = stats.mode[pi_x]
        globalmode_y = stats.mode[pi_y]

        @series begin
            seriestype := :scatter
            label := get(globalmode_options, "label", "global mode") #: ($(@sprintf("%.2f", globalmode_x)), $(@sprintf("%.2f", globalmode_y)))
            seriestype==:marginal ? subplot := 3 : 
            markeralpha := get(globalmode_options, "markeralpha", 1)
            markercolor := get(globalmode_options, "markercolor", :black)
            markersize := get(globalmode_options, "markersize", 4)
            markershape := get(globalmode_options, "markershape", :rect)
            markerstrokealpha := get(globalmode_options, "markerstrokealpha", 1)
            markerstrokecolor := get(globalmode_options, "markerstrokecolor", :black)
            markerstrokestyle := get(globalmode_options, "markerstrokestyle", :solid)
            markerstrokewidth := get(globalmode_options, "markerstrokewidth", 1)
            colorbar := colorbar

           ([globalmode_x], [globalmode_y])
        end
    end


    if localmode_options != ()  
        localmode_values = calculate_localmode_2d(h)
        for (i, l) in enumerate(localmode_values)
         @series begin
            seriestype := :scatter
            if i==1 && length(localmode_values)==1
                label := get(localmode_options, "label", "local mode") #: ($(@sprintf("%.2f", l[1])), $(@sprintf("%.2f", l[2])))
            elseif i ==1
                label := get(localmode_options, "label", "local modes")
            else 
                label :=""
            end

            seriestype == :marginal ? subplot := 3 : 
            markeralpha := get(localmode_options, "markeralpha", 0)
            markercolor := get(localmode_options, "markercolor", :black)
            markersize := get(localmode_options, "markersize", 4)
            markershape := get(localmode_options, "markershape", :rect)
            markerstrokealpha := get(localmode_options, "markerstrokealpha", 1)
            markerstrokecolor := get(localmode_options, "markerstrokecolor", :black)
            markerstrokestyle := get(localmode_options, "markerstrokestyle", :solid)
            markerstrokewidth := get(localmode_options, "markerstrokewidth", 1)
            colorbar := colorbar
            
            ([localmode_values[i][1]], [localmode_values[i][2]])
            end
        end
    end
end
