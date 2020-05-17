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
