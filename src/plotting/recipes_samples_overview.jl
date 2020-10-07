# This file is a part of BAT.jl, licensed under the MIT License (MIT).

@recipe function f(
    samples::DensitySampleVector;
    vsel=collect(1:5),
    mean=false,
    std=false,
    bins = 200,
    globalmode=false,
    marginalmode=false,
    diagonal = Dict(),
    upper = Dict(),
    lower = Dict(),
    vsel_label = []
)
    vsel = collect(vsel)
    if isa(vsel, Vector{<:Integer}) == false
        vsel = asindex.(Ref(BAT.varshape(samples)), vsel)
        vsel = reduce(vcat, vsel)
    end
    vsel = vsel[vsel .<=  totalndof(BAT.varshape(samples))]

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

    partition_tree = get(upper, "partition_tree", false)
    min_samples = minimum(flatview(unshaped.(samples.v)), dims=2)[:,1]
    max_samples = maximum(flatview(unshaped.(samples.v)), dims=2)[:,1]

    # get bin edges
    bins = if isa(bins, Integer)
        fill(bins, length(vsel))
    elseif isa(bins, Tuple{Vararg{Union{Integer, StepRangeLen}}})
        bins
    elseif isa(bins, NamedTuple)
        tmp_bins::Vector{Any} = fill(200, length(vsel))
        for k in keys(bins)
            idx = findfirst(isequal(string(k)), getstring.(Ref(samples), vsel))
            tmp_bins[idx] = bins[idx]
        end
        tmp_bins
    end


    for i in 1:nparams
        # diagonal
        @series begin
            subplot := i + (i-1)*nparams

            seriestype --> get(diagonal, "seriestype", :smallest_intervals)
            bins := bins[i]
            colors --> get(diagonal, "colors", default_colors)
            intervals --> get(diagonal, "intervals", default_credibilities)
            interval_labels --> get(diagonal, "interval_labels", [])
            legend --> get(diagonal, "legend", false)
            mean --> get(diagonal, "mean", mean)
            std --> get(diagonal, "std", std)
            globalmode --> get(diagonal, "globalmode", globalmode)
            marginalmode --> get(diagonal, "marginalmode", marginalmode)
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
                bins := (bins[i], bins[j])
                colors --> get(upper, "colors", default_colors)
                intervals --> get(upper, "intervals", default_credibilities)
                interval_labels --> get(upper, "interval_labels", [])
                legend --> get(upper, "legend", false)
                colorbar --> get(upper, "colorbar", false)
                mean --> get(upper, "mean", mean)
                std --> get(upper, "std", std)
                globalmode --> get(upper, "globalmode", globalmode)
                marginalmode --> get(upper, "marginalmode", marginalmode)
                xlims --> get(upper, "xlims", :auto)
                ylims --> get(upper, "ylims", :auto)

                if partition_tree != false
                    xguide --> xlabel[j]
                    yguide --> xlabel[i]

                    samples, (vsel[j], vsel[i])
                else
                    xguide --> xlabel[i]
                    yguide --> xlabel[j]

                    samples, (vsel[i], vsel[j])
                end
            end

            if partition_tree != false
                subspaces_rect_bounds = get_tree_par_bounds(partition_tree)
                axes = [vsel[j], vsel[i]]
                for rect in subspaces_rect_bounds
                    for ind in Base.OneTo(size(rect)[1])
                        rect[ind,:] = replace(rect[ind,:], Pair(Inf, max_samples[ind]), Pair(-Inf, min_samples[ind]))
                    end
                    box = _rect_box(rect[axes[1],1], rect[axes[1],2], rect[axes[2],1], rect[axes[2],2])

                    @series begin
                        subplot := j + (i-1)*nparams
                        seriestype --> :path
                        linewidth --> get(upper, "linewidth", 1)
                        linealpha --> get(upper, "linealpha", 1)
                        linecolor -->  get(upper, "linecolor", :darkgray)
                        legend --> false

                        xlims --> (min_samples[vsel[j]], max_samples[vsel[j]])
                        ylims --> (min_samples[vsel[i]], max_samples[vsel[i]])

                        box[1], box[2]
                    end
                end
            end

            # lower left plots
             @series begin
                 subplot := i + (j-1)*nparams

                 seriestype --> get(lower, "seriestype", :smallest_intervals)
                 bins := (bins[i], bins[j])
                 colors --> get(lower, "colors", default_colors)
                 colorbar --> get(lower, "colorbar", false)
                 intervals --> get(lower, "intervals", default_credibilities)
                 interval_labels --> get(lower, "interval_labels", [])
                 legend --> get(lower, "legend", false)
                 mean --> get(lower, "mean", mean)
                 std --> get(lower, "std", std)
                 globalmode --> get(lower, "globalmode", globalmode)
                 marginalmode --> get(lower, "marginalmode", marginalmode)
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
        conf_intervals = default_credibilities,
        colors = default_colors,
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


function _rect_box(x_s::F, x_f::F, y_s::F, y_f::F) where {F<:AbstractFloat}
    x = [x_s, x_f, x_f, x_s, x_s]
    y = [y_s, y_s, y_f, y_f, y_s]
    return (x, y)
end
