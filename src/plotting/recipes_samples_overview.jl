# This file is a part of BAT.jl, licensed under the MIT License (MIT).

@recipe function f(
    samples::DensitySampleVector;
    vsel=collect(1:5),
    mean=false,
    std=false,
    globalmode=false,
    localmode=false,
    diagonal = Dict(),
    upper = Dict(),
    lower = Dict(),
    vsel_label = []
)
    vsel = vsel[vsel .<= length(reduce(vcat, samples.v[1]))]

    xlabel = ["v$i" for i in vsel]
    ylabel = ["p(v$i)" for i in vsel]

    if isshaped(samples)
        xlabel = getstring.(Ref(samples), vsel)
        ylabel = ["p($l)" for l in xlabel]
    end

    if length(vsel_label) > 0
        xlabel = [vsel_label[i] for i in 1:length(vsel_label)]
        ylabel = ["p("*vsel_label[i]*")" for i in 1:length(vsel_label)]
    end


    nparams = length(vsel)
    layout --> nparams^2
    size --> (1000, 600)


    for i in 1:nparams
        # diagonal
        @series begin
            subplot := i + (i-1)*nparams

            seriestype --> get(diagonal, "seriestype", :smallest_intervals)
            bins --> get(diagonal, "bins", 200)
            colors --> get(diagonal, "colors", standard_colors)
            intervals --> get(diagonal, "intervals", standard_confidence_vals)
            interval_labels --> get(diagonal, "interval_labels", [])
            legend --> get(diagonal, "legend", false)
            mean --> get(diagonal, "mean", mean)
            std --> get(diagonal, "std", std)
            globalmode --> get(diagonal, "globalmode", globalmode)
            localmode --> get(diagonal, "localmode", localmode)
            xlims --> get(diagonal, "xlims", :auto)
            ylims --> get(diagonal, "ylims", :auto)
            xguide --> xlabel[i]
            yguide --> ylabel[i]

            samples, (vsel[i])
        end


        # upper right plots
        for j in i+1:nparams
            @series begin
                subplot := j + (i-1)*nparams

                seriestype --> get(upper, "seriestype", :histogram)
                bins --> get(upper, "bins", 200)
                colors --> get(upper, "colors", standard_colors)
                intervals --> get(upper, "intervals", standard_confidence_vals)
                interval_labels --> get(upper, "interval_labels", [])
                legend --> get(upper, "legend", false)
                colorbar --> get(upper, "colorbar", false)
                mean --> get(upper, "mean", mean)
                std --> get(upper, "std", std)
                globalmode --> get(upper, "globalmode", globalmode)
                localmode --> get(upper, "localmode", localmode)
                xlims --> get(upper, "xlims", :auto)
                ylims --> get(upper, "ylims", :auto)
                xguide --> xlabel[i]
                yguide --> xlabel[j]

                samples, (vsel[i], vsel[j])
            end

            # lower left plots
             @series begin
                 subplot := i + (j-1)*nparams

                 seriestype --> get(lower, "seriestype", :smallest_intervals)
                 bins --> get(lower, "bins", 200)
                 colors --> get(lower, "colors", standard_colors)
                 colorbar --> get(lower, "colorbar", false)
                 intervals --> get(lower, "intervals", standard_confidence_vals)
                 interval_labels --> get(lower, "interval_labels", [])
                 legend --> get(lower, "legend", false)
                 mean --> get(lower, "mean", mean)
                 std --> get(lower, "std", std)
                 globalmode --> get(lower, "globalmode", globalmode)
                 localmode --> get(lower, "localmode", localmode)
                 xlims --> get(lower, "xlims", :auto)
                 ylims --> get(lower, "ylims", :auto)
                 xguide --> xlabel[i]
                 yguide --> xlabel[j]

                 samples, (vsel[i], vsel[j])
             end

        end
    end

end

@recipe function f(model::Function,
        x::Union{StepRangeLen, Vector},
        samples::DensitySampleVector;
        conf_intervals = standard_confidence_vals,
        colors = standard_colors,
        mean = true,
        globalmode = true,
        localmode = true,
        xlabel = "x",
        ylabel = "f(x)",
        title="",
        legend=:topleft,
        size = (600, 400)
    )

    y_ribbons = zeros(length(x), 6)
    y_50_interval = zeros(length(x))

    intervals = 0.5*(1 .- conf_intervals)

    for x_ind in Base.OneTo(length(x))
        y_samples = model.(samples.v, x[x_ind])

        y_50_interval[x_ind] = quantile(y_samples, 0.5)
        y_ribbons[x_ind,:] .= [quantile(y_samples, intervals[1]), quantile(y_samples, 1. - intervals[1]),
                                 quantile(y_samples, intervals[2]), quantile(y_samples, 1. - intervals[2]),
                                 quantile(y_samples, intervals[3]), quantile(y_samples, 1. - intervals[3])]

        y_ribbons[x_ind,:] .= abs.(y_ribbons[x_ind,:] .- y_50_interval[x_ind])
    end

    xguide --> xlabel
    yguide --> ylabel
    title --> title
    legend --> legend
    size --> size

    @series begin
        ribbon --> (y_ribbons[:,5],y_ribbons[:,6])
        fillcolor --> colors[1]
        linecolor --> colors[1]
        seriesalpha --> 1
        linealpha --> 0.5
        label --> "$(standard_confidence_vals[3])"
        x, y_50_interval
    end

    @series begin
        ribbon --> (y_ribbons[:,3],y_ribbons[:,4])
        fillcolor --> colors[2]
        linecolor --> colors[2]
        seriesalpha --> 1
        linealpha --> 0.5
        label --> "$(standard_confidence_vals[2])"
        x, y_50_interval
    end

    @series begin
        ribbon --> (y_ribbons[:,1],y_ribbons[:,2])
        fillcolor --> colors[3]
        linecolor --> colors[3]
        seriesalpha --> 1
        linealpha --> 0.5
        label --> "$(standard_confidence_vals[1])"
        x, y_50_interval
    end

    if localmode
        local_mode_params = bat_findlocalmode(samples).result[1]

        @series begin
            linecolor --> :black
            linestyle --> :dash
            linewidth --> 1.5
            label --> "Local Mode"
            x, broadcast(x -> model(local_mode_params, x), x)
        end
    end

    if globalmode

        global_mode_params = mode(samples)[1]

        @series begin
            linecolor --> :black
            linestyle --> :dot
            linewidth --> 1.5
            label --> "Global Mode"
            x, broadcast(x -> model(global_mode_params, x), x)
        end
    end

    if mean

        global_mode_params = mode(samples)[1]

        @series begin
            linecolor --> :black
            linestyle --> :solid
            linewidth --> 1.5
            label --> "0.5 Quantile"
            x, y_50_interval
        end
    end

end
