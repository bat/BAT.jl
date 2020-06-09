# This file is a part of BAT.jl, licensed under the MIT License (MIT).
@recipe function f(
    maybe_shaped_samples::DensitySampleVector,
    parsel::NTuple{2,Union{Symbol, Expr, Integer}};
    intervals = standard_confidence_vals,
    interval_labels = [],
    colors = standard_colors,
    mean = false,
    std = false,
    globalmode = false,
    localmode = true,
    diagonal = Dict(),
    upper = Dict(),
    right = Dict(),
    filter = false,
    closed = :left
)
    xindx = asindex(maybe_shaped_samples, parsel[1])
    yindx = asindex(maybe_shaped_samples, parsel[2])

    if length(xindx) > 1
        throw(ArgumentError("Symbol :$(parsel[1]) refers to a multivariate parameter. Use :($(parsel[1])[i]) instead."))
    elseif length(yindx) > 1
        throw(ArgumentError("Symbol :$(parsel[2]) refers to a multivariate parameter. Use :($(parsel[2])[i]) instead."))
    end

    samples = unshaped.(maybe_shaped_samples)
    filter ? samples = BAT.drop_low_weight_samples(samples) : nothing

    bins = get(plotattributes, :bins, 200)
    seriestype = get(plotattributes, :seriestype, :smallest_intervals)

    colorbar = false
    if seriestype == :histogram2d || seriestype == :histogram || seriestype == :hist
        colorbar = true
    end

    xlabel, ylabel  = if !isshaped(maybe_shaped_samples)
        "v$xindx", "v$yindx"
    else
        getstring(maybe_shaped_samples, xindx), getstring(maybe_shaped_samples, yindx)
    end

    xguide := get(plotattributes, :xguide, xlabel)
    yguide := get(plotattributes, :yguide, ylabel)

    marg = bat_marginalize(
        samples,
        (xindx, yindx),
        nbins = bins,
        closed = closed,
        filter = filter
    ).result

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
            markercolor := [w >= 1 ? color : RGBA(convert(RGB, color), color.alpha * w) for w in samples.weight[acc]]
            (flatview(samples.v)[xindx, acc], flatview(samples.v)[yindx, acc])
        end

        if !isempty(rej)
            @series begin
                seriestype := :scatter
                label := "rejected"
                markersize := base_markersize
                markerstrokewidth := 0
                markercolor := :red
                (flatview(samples.v)[xindx, rej], flatview(samples.v)[yindx, rej])
            end
        end


    else
        @series begin
            seriestype --> seriestype
            intervals --> intervals
            interval_labels --> interval_labels
            colors --> colors
            diagonal --> diagonal
            upper --> upper
            right --> right

            marg, (xindx, yindx)
        end
    end


    #------ stats ----------------------------
    stats = MCMCBasicStats(samples)

    mean_options = convert_to_options(mean)
    globalmode_options = convert_to_options(globalmode)
    localmode_options = convert_to_options(localmode)
    std_options = convert_to_options(std)


    if mean_options != ()
        mx= stats.param_stats.mean[xindx]
        my = stats.param_stats.mean[yindx]

        Σ_all = stats.param_stats.cov
        Σx = Σ_all[xindx, xindx]
        Σy = Σ_all[yindx, yindx]

        @series begin
            seriestype := :scatter
            label := get(mean_options, "label", "mean")
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

            if(std)
                xerror := sqrt(Σx)
                yerror := sqrt(Σy)
            end
           ([mx], [my])
        end
    end


   if globalmode_options != ()
        globalmode_x = stats.mode[xindx]
        globalmode_y = stats.mode[yindx]

        @series begin
            seriestype := :scatter
            label := get(globalmode_options, "label", "global mode")
            seriestype==:marginal ? subplot := 3 :
            markeralpha := get(globalmode_options, "markeralpha", 1)
            markercolor := get(globalmode_options, "markercolor", :black)
            markersize := get(globalmode_options, "markersize", 4)
            markershape := get(globalmode_options, "markershape", :rect)
            linealpha := 0
            markerstrokealpha := get(globalmode_options, "markerstrokealpha", 1)
            markerstrokecolor := get(globalmode_options, "markerstrokecolor", :black)
            markerstrokestyle := get(globalmode_options, "markerstrokestyle", :solid)
            markerstrokewidth := get(globalmode_options, "markerstrokewidth", 1)
            colorbar := colorbar

           ([globalmode_x], [globalmode_y])
        end
    end


    if localmode_options != ()
        localmode_values = find_localmodes(marg)
        for (i, l) in enumerate(localmode_values)
         @series begin
            seriestype := :scatter
            if i==1 && length(localmode_values)==1
                label := get(localmode_options, "label", "local mode")
            elseif i ==1
                label := get(localmode_options, "label", "local modes")
            else
                label :=""
            end

            seriestype == :marginal ? subplot := 3 :
            markercolor := get(localmode_options, "markercolor", :dimgrey)
            markersize := get(localmode_options, "markersize", 4)
            markershape := get(localmode_options, "markershape", :rect)
            markeralpha := get(localmode_options, "markeralpha", 1)
            linealpha := 0
            markerstrokealpha := get(localmode_options, "markerstrokealpha", 1)
            markerstrokecolor := get(localmode_options, "markerstrokecolor", :dimgrey)
            markerstrokestyle := get(localmode_options, "markerstrokestyle", :solid)
            markerstrokewidth := get(localmode_options, "markerstrokewidth", 1)
            colorbar := colorbar

            ([localmode_values[i][1]], [localmode_values[i][2]])
            end
        end
    end
end
