# This file is a part of BAT.jl, licensed under the MIT License (MIT).

@recipe function f(
    prior::NamedTupleDist;
    vsel=collect(1:5),
    bins = 200,
    diagonal = Dict(),
    upper = Dict(),
    lower = Dict(),
    vsel_label = []
)
    vsel = collect(vsel)
    if isa(vsel, Vector{<:Integer}) == false
        vsel = asindex.(Ref(BAT.varshape(prior)), vsel)
        vsel = reduce(vcat, vsel)
    end
    vsel = vsel[vsel .<=  totalndof(BAT.varshape(prior))]

    xlabel = [getstring(prior, i) for i in vsel]
    ylabel = ["p($l)" for l in xlabel]

    if length(vsel_label) > 0
        xlabel = [vsel_label[i] for i in 1:length(vsel_label)]
        ylabel = ["p("*vsel_label[i]*")" for i in 1:length(vsel_label)]
    end

    nparams = length(vsel)
    layout --> nparams^2
    size --> (1000, 600)


    bins = if isa(bins, Integer)
        fill(bins, length(vsel))
    elseif isa(bins, Tuple{Vararg{Union{Integer, StepRangeLen}}})
        bins
    elseif isa(bins, NamedTuple)
        tmp_bins::Vector{Any} = fill(200, length(vsel))
        for k in keys(bins)
            idx = findfirst(isequal(string(k)), getstring.(Ref(prior), vsel))
            tmp_bins[idx] = bins[idx]
        end
        tmp_bins
    end


    for i in 1:nparams
        # diagonal
        @series begin
            subplot := i + (i-1)*nparams

            seriestype --> get(diagonal, "seriestype", :stephist)
            bins := bins[i]
            colors --> get(diagonal, "colors", default_colors)
            intervals --> get(diagonal, "intervals", default_credibilities)
            legend --> get(diagonal, "legend", false)
            xlims --> get(diagonal, "xlims", :auto)
            ylims --> get(diagonal, "ylims", :auto)
            linecolor --> get(diagonal, "linecolor", :black)
            xguide --> xlabel[i]
            yguide --> ylabel[i]

            prior, (vsel[i])
        end

        # upper right plots
        for j in i+1:nparams

            @series begin
                subplot := j + (i-1)*nparams

                seriestype --> get(upper, "seriestype", :smallest_intervals_contour)
                bins := (bins[i], bins[j])
                more_colors = []
                colors --> get(upper, "colors", colormap("Blues", 10))
                more_intervals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
                intervals --> get(upper, "intervals", more_intervals)
                interval_labels --> get(upper, "interval_labels", [])
                legend --> get(upper, "legend", false)
                xlims --> get(upper, "xlims", :auto)
                ylims --> get(upper, "ylims", :auto)
                xguide --> xlabel[i]
                yguide --> xlabel[j]

                prior, (vsel[i], vsel[j])
            end

            # lower left plots
            @series begin
                subplot := i + (j-1)*nparams

                seriestype --> get(lower, "seriestype", :smallest_intervals_contour)
                bins := (bins[i], bins[j])
                colors --> get(lower, "colors", default_colors)
                intervals --> get(lower, "intervals", default_credibilities)
                interval_labels --> get(lower, "interval_labels", [])
                legend --> get(lower, "legend", false)
                xlims --> get(lower, "xlims", :auto)
                ylims --> get(lower, "ylims", :auto)
                xguide --> xlabel[i]
                yguide --> xlabel[j]

                prior, (vsel[i], vsel[j])
            end

        end
    end

end
