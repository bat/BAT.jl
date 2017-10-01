# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using RecipesBase


@recipe function f(target::AbstractTargetSubject, samples::DensitySampleVector)
    acc = find(x -> x > 0, samples.weight)
    rej = find(x -> x <= 0, samples.weight)

    bounds = target.bounds
    vhi = bounds.vol.hi; vlo = bounds.vol.lo
    bext = 0.15 * (vhi - vlo)
    xlims = (vlo[1] - bext[1], vhi[1] + bext[1])
    ylims = (vlo[2] - bext[2], vhi[2] + bext[2])

    base_markersize = 2

    @series begin
        seriestype := :scatter
        label := isempty(rej) ? "samples" : "accepted"
        markersize := base_markersize * sqrt.(samples.weight[acc])
        markerstrokewidth := 0
        xlims := xlims
        ylims := ylims
        color := :green
        (samples.params[1, acc], samples.params[2, acc])
    end

    if !isempty(rej)
        @series begin
            seriestype := :scatter
            label := "rejected"
            markersize := base_markersize
            markerstrokewidth := 0
            xlims := xlims
            ylims := ylims
            color := :red
            (samples.params[1, rej], samples.params[2, rej])
        end
    end

    bounds_rect_X = [vlo[1],vhi[1],vhi[1],vlo[1],vlo[1]]
    bounds_rect_Y = [vlo[2],vlo[2],vhi[2],vhi[2],vlo[2]]

    @series begin
        seriestype := :path
        label := "bounds"
        linewidth := 3
        linecolor := :violet
        xlims := xlims
        ylims := ylims
        (bounds_rect_X, bounds_rect_Y)
    end

    ()
end
