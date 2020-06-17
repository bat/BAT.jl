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

@recipe function f(x::Union{StepRangeLen, Vector},
        model::Function,
        sample_from::Union{DensitySampleVector, AbstractDensity};
        n_samples = 10^4,
        conf_intervals = standard_confidence_vals,
        colors = standard_colors,
        global_mode = true,
        marginal_mode = false)

    if typeof(sample_from) <: DensitySampleVector
        samples = bat_sample(sample_from, n_samples).result
    else
        samples = bat_sample(sample_from.prior.dist, n_samples).result
    end

    y_ribbons = zeros(Float64, length(x), 2*length(conf_intervals))
    y_median = zeros(Float64, length(x))
    quantile_values = zeros(Float64, 2*length(conf_intervals))

    quantile_values[1:2:end] .= 0.5*(1 .- conf_intervals)
    quantile_values[2:2:end] .= 1 .- 0.5*(1 .- conf_intervals)

    for x_ind in Base.OneTo(length(x))
        y_samples = model.(samples.v, x[x_ind])
        y_median[x_ind] = quantile(y_samples, weights(samples.weight), 0.5)
        y_ribbons[x_ind,:] .= [quantile(y_samples, weights(samples.weight), quantile_tmp) for quantile_tmp in quantile_values]
        y_ribbons[x_ind,:] .= abs.(y_ribbons[x_ind,:] .- y_median[x_ind])
    end

    xguide --> "x"
    yguide --> "f(x)"
    title --> ""
    legend --> :topleft
    size --> (600, 400)

    for interval_ind in length(conf_intervals):-1:1
        @series begin
            ribbon --> (y_ribbons[:,interval_ind*2 - 1],y_ribbons[:,interval_ind*2])
            fillcolor --> colors[interval_ind]
            linecolor --> colors[interval_ind]
            seriesalpha --> 1
            linealpha --> 1
            fillalpha --> 1
            label --> "$(conf_intervals[interval_ind])"
            x, y_median
        end
    end

    @series begin
        linecolor --> :black
        linestyle --> :solid
        linewidth --> 1.5
        label --> "Median"
        x, y_median
    end

    if global_mode
        global_mode_params = mode(samples)[1]
        @series begin
            linecolor --> :black
            linestyle --> :dot
            linewidth --> 1.5
            label --> "Global Mode"
            x, broadcast(x -> model(global_mode_params, x), x)
        end
    end

    if marginal_mode
        marginal_mode_params = bat_marginalmode(samples).result[1]
        @series begin
            linecolor --> :black
            linestyle --> :dash
            linewidth --> 1.5
            label --> "Marginal Mode"
            x, broadcast(x -> model(marginal_mode_params, x), x)
        end
    end
end
