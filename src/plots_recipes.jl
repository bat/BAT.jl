# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# ToDo: Modularize

@recipe function f(samples::DensitySampleVector, parsel::NTuple{2,Integer})
    pi_x, pi_y = parsel

    acc = find(x -> x > 0, samples.weight)
    rej = find(x -> x <= 0, samples.weight)

    base_markersize = get(d, :markersize, 1.5)
    seriestype = get(d, :seriestype, :scatter)

    plot_bounds = get(d, :bounds, true)
    delete!(d, :bounds)

    if seriestype == :scatter
        color = get(d, :seriescolor, :green)
        label = get(d, :label, isempty(rej) ? "samples" : "accepted")

        @series begin
            seriestype := :scatter
            label := label
            markersize := base_markersize * sqrt.(samples.weight[acc])
            markerstrokewidth := 0
            color := color
            xlabel --> "\$\\theta_$(pi_x)\$"
            ylabel --> "\$\\theta_$(pi_y)\$"
            (samples.params[pi_x, acc], samples.params[pi_y, acc])
        end

        if !isempty(rej)
            @series begin
                seriestype := :scatter
                label := "rejected"
                markersize := base_markersize
                markerstrokewidth := 0
                color := :red
                (samples.params[pi_x, rej], samples.params[pi_y, rej])
            end
        end
    elseif seriestype == :histogram2d
        @series begin
            seriestype := :histogram2d
            label --> "samples"
            xlabel --> "\$\\theta_$(pi_x)\$"
            ylabel --> "\$\\theta_$(pi_y)\$"
            weights := samples.weight
            (samples.params[pi_x, acc], samples.params[pi_y, acc])
        end
    else
        error("seriestype $seriestype not supported")
    end

    nothing
end


@recipe function f(bounds::HyperRectBounds, parsel::NTuple{2,Integer})
    pi_x, pi_y = parsel

    vhi = bounds.vol.hi; vlo = bounds.vol.lo
    bext = 0.1 * (vhi - vlo)
    xlims = (vlo[pi_x] - bext[pi_x], vhi[pi_x] + bext[pi_x])
    ylims = (vlo[pi_y] - bext[pi_y], vhi[pi_y] + bext[pi_y])
    bounds_rect_X = [vlo[pi_x],vhi[pi_x],vhi[pi_x],vlo[pi_x],vlo[pi_x]]
    bounds_rect_Y = [vlo[pi_y],vlo[pi_y],vhi[pi_y],vhi[pi_y],vlo[pi_y]]

    @series begin
        seriestype := :path
        label --> "bounds"
        linewidth --> 2
        linecolor --> :violet
        xlims --> xlims
        ylims --> ylims
        (bounds_rect_X, bounds_rect_Y)
    end

    nothing
end
