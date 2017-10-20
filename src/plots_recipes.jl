# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# ToDo: Modularize

@recipe function f(samples::DensitySampleVector, parsel::NTuple{2,Integer} = (1,2))
    pi_x, pi_y = parsel

    acc = find(x -> x > 0, samples.weight)
    rej = find(x -> x <= 0, samples.weight)

    base_markersize = get(d, :markersize, 1.5)

    plot_bounds = get(d, :bounds, true)
    delete!(d, :bounds)

    color = get(d, :seriescolor, :green)
    label = get(d, :label, isempty(rej) ? "samples" : "accepted")

    @series begin
        seriestype := :scatter
        label := label
        markersize := base_markersize * sqrt.(samples.weight[acc])
        markerstrokewidth := 0
        color := color
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

    nothing
end


@recipe function f(bounds::HyperRectBounds)
    vhi = bounds.vol.hi; vlo = bounds.vol.lo
    bext = 0.1 * (vhi - vlo)
    xlims = (vlo[1] - bext[1], vhi[1] + bext[1])
    ylims = (vlo[2] - bext[2], vhi[2] + bext[2])
    bounds_rect_X = [vlo[1],vhi[1],vhi[1],vlo[1],vlo[1]]
    bounds_rect_Y = [vlo[2],vlo[2],vhi[2],vhi[2],vlo[2]]

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
