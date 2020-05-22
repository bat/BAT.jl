# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function create_rectangle(rect::HyperRectVolume, dim1::Integer, dim2::Integer)
    xpoints = zeros(Float64, 6)
    ypoints = zeros(Float64, 6)

    xpoints[1] = rect.lo[dim1]
    xpoints[2] = rect.lo[dim1]
    xpoints[3] = rect.hi[dim1]
    xpoints[4] = rect.hi[dim1]
    xpoints[5] = rect.lo[dim1]
    xpoints[6] = NaN    #necessary to group rectangles under same label
    ypoints[1] = rect.lo[dim2]
    ypoints[2] = rect.hi[dim2]
    ypoints[3] = rect.hi[dim2]
    ypoints[4] = rect.lo[dim2]
    ypoints[5] = rect.lo[dim2]
    ypoints[6] = NaN

    xpoints, ypoints
end



@recipe function f(data::HMIData, dim1::Integer = 1, dim2::Integer = 2; rscale = 0.5,
        plot_seedsamples = true,
        plot_seedcubes = false,
        plot_samples = true,
        plot_acceptedrects = true,
        plot_rejectedrects = false,
        plot_datasets = 0,
        font_scale = 1.0)

    @assert data.dataset1.P == data.dataset2.P
    @assert data.dataset1.P > 1

    #plot both data sets
    if plot_datasets == 0
        layout := _plots_module().GridLayout(1, 2)
        plotattributes[:size] = (1600, 800)
    else
        plotattributes[:size] = (800, 800)
    end

    for current_ds in [1, 2]
        if plot_datasets != 0 && current_ds != plot_datasets
            continue
        end

        dataset, rejectedids, volumes, cubes =
            if current_ds == 1
                data.dataset1, data.rejectedrects2, data.volumelist2, data.cubelist2
            else
                data.dataset2, data.rejectedrects1, data.volumelist1, data.cubelist1
            end


        @assert dataset.P > 1
        @assert dim1 != dim2

        xlimits=(minimum(dataset.data[dim1, :]), maximum(dataset.data[dim1, :]))
        ylimits=(minimum(dataset.data[dim2, :]), maximum(dataset.data[dim2, :]))

        pts = dataset.data[:, dataset.startingIDs]

        rejected = length(rejectedids)
        accepted = length(volumes) - rejected
        rectangles_x = Array{Float64, 1}(undef, accepted * 6)
        rectangles_y = Array{Float64, 1}(undef, accepted * 6)
        rectanglesrej_x = Array{Float64, 1}(undef, rejected * 6)
        rectanglesrej_y = Array{Float64, 1}(undef, rejected * 6)
        cntr = 1
        cntrrej = 1
        for i in eachindex(volumes)
            if i in rejectedids
                rectanglesrej_x[cntrrej:cntrrej+5], rectanglesrej_y[cntrrej:cntrrej+5] = create_rectangle(volumes[i].spatialvolume, dim1, dim2)
                cntrrej += 6
            else
                rectangles_x[cntr:cntr+5], rectangles_y[cntr:cntr+5] = create_rectangle(volumes[i].spatialvolume, dim1, dim2)
                cntr += 6
            end
        end

        cubes_x = zeros(Float64, length(cubes) * 6)
        cubes_y = zeros(Float64, length(cubes) * 6)
        cntr = 1
        for i in eachindex(cubes)
            cubes_x[cntr:cntr+5], cubes_y[cntr:cntr+5] = create_rectangle(cubes[i], dim1, dim2)
            cntr += 6
        end

        if plot_datasets == 0
            title  := "Data Set $current_ds"
        end
        xlabel := latexstring("\$\\lambda_$(dim1)\$")
        ylabel := latexstring("\$\\lambda_$(dim2)\$")

        titlefont := _plots_module().font("sans-serif", 21 * font_scale)
        guidefont := _plots_module().font("sans-serif", 18 * font_scale)
        tickfont  := _plots_module().font("sans-serif", 15 * font_scale)
        legendfont:= _plots_module().font("sans-serif", 12 * font_scale)
        legend := :topright



        #plot samples
        if plot_samples
            @series begin
                seriestype := :scatter
                if plot_datasets == 0
                    subplot := current_ds
                end
                label := "Samples"
                markercolor := :red
                marker := :circle
                markersize := rscale .* sqrt.(dataset.weights)
                markerstrokewidth := 0
                size := (1000, 1000)
                xlim := xlimits
                ylim := ylimits

                dataset.data[dim1, :], dataset.data[dim2, :]
            end
        end

        #plot seeds
        if plot_seedsamples
            @series begin
                seriestype := :scatter
                if plot_datasets == 0
                    subplot := current_ds
                end
                label := "Seed Samples"
                markercolor := :black
                markersize := 5.0
                marker := :rect
                linewidth := 0.0

                pts[dim1, :], pts[dim2, :]
            end
        end

        #plot hyper-rectangles, that survive the trimming
        if plot_acceptedrects
            @series begin
                seriestype := :path
                if plot_datasets == 0
                    subplot := current_ds
                end
                linecolor := :blue
                linewidth := 1.5

                if plot_rejectedrects && plot_acceptedrects
                    label := "Accepted Rectangles"
                else
                    label := "Rectangles"
                end

                rectangles_x[1:end-1], rectangles_y[1:end-1]

            end
        end

        #plot hyper-rectangles, that did not survive the trimming
        if plot_rejectedrects
            @series begin
                seriestype := :path
                if plot_datasets == 0
                    subplot := current_ds
                end
                linecolor := :green
                linewidth := 1.5
                label := "Rejected Rectangles"

                rectanglesrej_x[1:end-1], rectanglesrej_y[1:end-1]
            end
        end

        #plot seed hyper-cubes
        if plot_seedcubes
            @series begin
                seriestype := :path
                if plot_datasets == 0
                    subplot := current_ds
                end
                linecolor := :black
                label := "Hyper-Rectangle Seeds"
                linewidth := 1.5

                cubes_x[1:end-1], cubes_y[1:end-1]
            end
        end

    end
end
