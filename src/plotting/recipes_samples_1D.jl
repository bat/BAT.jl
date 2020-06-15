# This file is a part of BAT.jl, licensed under the MIT License (MIT).

@recipe function f(
    maybe_shaped_samples::DensitySampleVector,
    parsel::Union{Integer, Symbol, Expr};
    intervals = standard_confidence_vals,
    bins = 200,
    normalize = true,
    colors = standard_colors,
    interval_labels = [],
    mean = false,
    std = false,
    globalmode = false,
    localmode = true,
    filter = false,
    closed = :left
)
    idx = asindex(maybe_shaped_samples, parsel)

    if length(idx) > 1
        throw(ArgumentError("Symbol :$parsel refers to a multivariate parameter. Use :($parsel[i]) instead."))
    end

    marg = bat_marginalize(
        maybe_shaped_samples,
        parsel,
        nbins = bins,
        closed = closed,
        filter = filter,
        normalize = normalize
    ).result

    orientation = get(plotattributes, :orientation, :vertical)
    (orientation != :vertical) ? swap=true : swap = false

    xlabel = if !isshaped(maybe_shaped_samples)
        "v$idx"
    else
        getstring(maybe_shaped_samples, idx)
    end

    ylabel = "p("*xlabel*")"

    xlabel = get(plotattributes, :xguide, xlabel)
    ylabel = get(plotattributes, :yguide, ylabel)

    if swap
        xguide := ylabel
        yguide := xlabel
    else
        xguide := xlabel
        yguide := ylabel
    end


    @series begin
        seriestype --> :smallest_intervals
        intervals --> intervals
        normalize --> normalize
        colors --> colors
        interval_labels --> interval_labels

        marg, idx
    end

    #------ stats ----------------------------
    stats = MCMCBasicStats(maybe_shaped_samples)

    line_height = maximum(marg.dist.h.weights)*1.03

    mean_options = convert_to_options(mean)
    globalmode_options = convert_to_options(globalmode)
    localmode_options = convert_to_options(localmode)
    std_options = convert_to_options(std)

    # standard deviation
    if std_options != ()
        Σ_all = stats.param_stats.cov
        Σ = Σ_all[idx, idx]
        dev = sqrt(Σ)
        meanvalue = stats.param_stats.mean[idx]
        @series begin
            seriestype := :shape
            label := get(std_options, "label", "std. dev.")
            linewidth := 0
            fillcolor := get(std_options, "fillcolor", :grey)
            fillalpha := get(std_options, "fillalpha", 0.5)

            uncertaintyband(meanvalue, dev, line_height, swap=swap)
        end
    end

    # mean
    if mean_options != ()
        meanvalue = stats.param_stats.mean[idx]
        @series begin
            seriestype := :line
            label := get(mean_options, "label", "mean")
            linestyle := get(mean_options, "linestyle", :solid)
            linecolor := get(mean_options, "linecolor", :dimgrey)
            linewidth := get(mean_options, "linewidth", 1)
            linealpha := get(mean_options, "alpha", 1)

            line(meanvalue, line_height, swap=swap)
        end
    end

    # global mode
    if globalmode_options != ()
        globalmode_value = stats.mode[idx]
         @series begin
            seriestype := :line
            label := get(globalmode_options, "label", "global mode")
            linestyle := get(globalmode_options, "linestyle", :dash)
            linecolor := get(globalmode_options, "linecolor", :black)
            linewidth := get(globalmode_options, "linewidth", 1)
            linealpha := get(globalmode_options, "alpha", 1)

            line(globalmode_value, line_height, swap=swap)
        end
    end

    # local mode(s)
    if localmode_options != ()
        localmode_values = find_localmodes(marg)

        for (i, l) in enumerate(localmode_values)
         @series begin
            seriestype := :line
            if length(localmode_values)==1
                label := get(localmode_options, "label", "local mode")
            elseif i ==1
                label := get(localmode_options, "label", "local modes")
            else
                label :=""
            end

            linestyle := get(localmode_options, "linestyle", :dot)
            linecolor := get(localmode_options, "linecolor", :black)
            linewidth := get(localmode_options, "linewidth", 1)
            linealpha := get(localmode_options, "alpha", 1)

            line(l[1], line_height, swap=swap)
            end
        end
    end
end



function convert_to_options(arg::Dict)
    return arg
end

function convert_to_options(arg::Bool)
    arg ? Dict() : ()
end

function line(pos::Real, height::Real; swap::Bool=false)
    if swap
        return  [(0, pos), (height, pos)]
    else
        return  [(pos, 0), (pos, height)]
    end
end


function uncertaintyband(m, u, h; swap=false)
    if _plots_module() != nothing
        x = [m-u,m+u,m+u,m-u]
        y = [0,0,h,h]

        if swap
            x, y = y, x
        end

        return _plots_module().Shape(x, y)
    else
        ()
    end
end
