# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function _histogram_heatmap_surface(hist::StatsBase.Histogram)
    # Plots stores z in (y, x) order. Passing edges directly avoids the
    # center reconstruction performed by its bins2d recipe on unequal cells.
    values = map(w -> iszero(w) ? NaN : w, hist.weights)
    return _Plots_Surface(permutedims(values))
end

function _histogram_contour_centers(hist::Histogram)
    # Match admitted region arithmetic: subtract after promotion, since native
    # integer endpoint differences can wrap even when the final center fits.
    T = promote_type(Float64, float(eltype(hist.weights)), map(e -> float(eltype(e)), hist.edges)...)
    centers = map(hist.edges) do edges
        converted = [_plot_float64(T(edges[i]) + (T(edges[i+1]) - T(edges[i])) / 2,
            "histogram contour center") for i in 1:length(edges)-1]
        all(i -> edges[i] < converted[i] < edges[i+1], eachindex(converted)) ||
            _plot_weight_error("histogram contour centers must remain inside their cells in Float64; use cell regions for this grid")
        all(i -> converted[i] < converted[i + 1], 1:length(converted)-1) ||
            _plot_weight_error("histogram contour centers must remain strictly increasing in Float64; use cell regions for this grid")
        converted
    end
    return centers
end

@recipe function f(
    marg::MarginalDist,
    vsel::NTuple{2,Union{Symbol, Expr, Integer}};
    intervals = default_credibilities,
    colors = default_colors,
    smoothing = 0,
    diagonal = Dict(),
    upper = Dict(),
    right = Dict(),
    interval_labels = [],
    normalize = true
)
    hist = convert(Histogram, marg.dist isa ReshapedDist ? marg.dist.dist : marg.dist)
    seriestype = get(plotattributes, :seriestype, :histogram2d)

    xlabel = get(plotattributes, :xguide, "x$(vsel[1])")
    ylabel = get(plotattributes, :yguide, "x$(vsel[2])")

    # histogram / heatmap
    if seriestype == :histogram2d || seriestype == :histogram || seriestype == :hist
        @series begin
            seriestype := :heatmap
            xguide --> xlabel
            yguide --> ylabel
            colorbar --> true

            collect(hist.edges[1]), collect(hist.edges[2]), _histogram_heatmap_surface(hist)
        end


    # smallest interval contours
    elseif seriestype == :smallest_intervals_contour || seriestype == :smallest_intervals_contourf

        _validate_histogram_region_metadata(intervals, colors, interval_labels)
        get(plotattributes, :contour_labels, false) && throw(ArgumentError(
            "Histogram contour labels use relative density, not original PDF units; disable contour_labels (clabels)"))
        get(plotattributes, :clabels, false) && throw(ArgumentError(
            "Histogram contour labels use relative density, not original PDF units; disable contour_labels (clabels)"))
        data = _histogram_contour_data(hist, intervals; smoothing=smoothing)
        metadata = _histogram_region_metadata(intervals, colors, interval_labels; order=data.order)
        filled = seriestype == :smallest_intervals_contourf
        x, y = any(c -> c !== nothing, data.cutoffs) ? _histogram_contour_centers(hist) : (Float64[], Float64[])
        surface = permutedims(data.density)

        colorbar --> false
        xguide --> xlabel
        yguide --> ylabel

        # Paint broad to tight. A repeated cutoff has one owner: its last
        # request, whose color would otherwise overpaint all earlier copies.
        seen_cutoffs = Set{Float64}()
        contour_indices = Int[]
        for i in reverse(eachindex(data.cutoffs))
            cutoff = data.cutoffs[i]
            cutoff === nothing && continue
            cutoff in seen_cutoffs && continue
            push!(seen_cutoffs, cutoff)
            push!(contour_indices, i)
        end
        for i in reverse(contour_indices)
            cutoff = data.cutoffs[i]
            contour_color = metadata.colors[i]
            @series begin
                seriestype := filled ? :contourf : :contour
                # The normalized field's maximum is one. Two is only the
                # renderer's upper fill sentinel, never a probability cutoff.
                levels := filled ? [cutoff, 2.0] : [cutoff]
                linewidth --> (filled ? 0 : 2)
                linecolor := contour_color
                seriescolor := filled ? _Plots_cgrad([contour_color, contour_color]) : contour_color
                label := metadata.labels === nothing ?
                    _histogram_region_label(data.contents[i]) : metadata.labels[i]
                x, y, surface
            end
        end


    # smallest intervals heatmap
    elseif seriestype == :smallest_intervals || seriestype == :HDR
        metadata = _histogram_region_metadata(intervals, colors, interval_labels)

        hists, realintervals = get_smallest_intervals(hist, intervals)

        for (i, int) in enumerate(realintervals)
            @series begin
                seriestype := :heatmap
                seriescolor --> _Plots_cgrad([metadata.colors[i], metadata.colors[i]])
                xguide --> xlabel
                yguide --> ylabel
                colorbar --> false
                label := ""

                collect(hists[i].edges[1]), collect(hists[i].edges[2]), _histogram_heatmap_surface(hists[i])
            end

            # fake a legend
            interval_label = metadata.labels === nothing ?
                _histogram_region_label(realintervals[i]) : metadata.labels[i]

            @series begin
                seriestype := :shape
                fillcolor --> metadata.colors[i]
                linewidth --> 0
                label := interval_label
                colorbar --> false
                [hists[i].edges[1][1], hists[i].edges[1][1]], [hists[i].edges[2][1], hists[i].edges[2][1]]
            end
        end


    # marginal histograms
    elseif seriestype == :marginal

        layout --> _Plots_grid(2,2, widths=(0.8, 0.2), heights=(0.2, 0.8))
        link --> :both

        if get(diagonal, "seriestype", :histogram) != :histogram
            colorbar --> false
        end

        @series begin
            subplot := 1
            xguide := xlabel
            yguide := "p("*xlabel*")"
            seriestype := get(upper, "seriestype", :histogram)
            bins --> get(upper, "nbins", 200)
            normalize --> get(upper, "normalize", true)
            colors --> get(upper, "colors", default_colors)
            intervals --> get(upper, "intervals", default_credibilities)
            legend --> get(upper, "legend", true)

            marg, vsel[1]
        end

        # empty plot (needed since @layout macro not available)
        @series begin
            seriestype := :scatter
            subplot := 2
            grid := false
            xaxis := false
            yaxis := false
            markersize := 0.001
            markerstrokewidth := 0
            markeralpha := 1
            markerstrokealpha := 1
            legend := false
            label := ""
            xguide := ""
            yguide := ""
            [(0,0)]
        end

        @series begin
            subplot := 3
            seriestype := get(diagonal, "seriestype", :histogram)
            xguide --> xlabel
            yguide --> ylabel
            normalize --> get(diagonal, "normalize", true)
            bins --> get(diagonal, "nbins", 200)
            colors --> get(diagonal, "colors", default_colors)
            intervals --> get(diagonal, "intervals", default_credibilities)
            legend --> get(diagonal, "legend", false)

            marg, (vsel[1], vsel[2])
        end

        @series begin
            subplot := 4
            seriestype := get(right, "seriestype", :histogram)
            orientation := :horizontal
            xguide := ylabel
            yguide := "p("*ylabel*")"
            normalize --> get(right, "normalize", true)
            bins --> get(right, "nbins", 200)
            colors --> get(right, "colors", default_colors)
            intervals --> get(right, "intervals", default_credibilities)
            legend --> get(right, "legend", true)

            marg, vsel[2]
        end

    else
        error("seriestype $seriestype not supported")
    end

end


# Shouldn't be used anymore:
#=

# rectangle bounds
@recipe function f(bounds::HyperRectBounds, vsel::NTuple{2,Integer})
    pi_x, pi_y = vsel

    vol = spatialvolume(bounds)
    vhi = vol.hi[[pi_x, pi_y]]; vlo = vol.lo[[pi_x, pi_y]]
    rect_xy = rectangle_path(vlo, vhi)
    bext = 0.1 * (vhi - vlo)
    xlims = (vlo[1] - bext[1], vhi[1] + bext[1])
    ylims = (vlo[2] - bext[2], vhi[2] + bext[2])

    @series begin
        seriestype := :path
        label --> "bounds"
        seriescolor --> :darkred
        seriesalpha --> 0.75
        linewidth --> 2
        xlims --> xlims
        ylims --> ylims
        (rect_xy[:,1], rect_xy[:,2])
    end

    nothing
end

=#
