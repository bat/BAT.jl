# This file is a part of BAT.jl, licensed under the MIT License (MIT).


const standard_confidence_vals = [0.68, 0.95, 0.997]


function rectangle_path(lo::Vector{<:Real}, hi::Vector{<:Real})
    [
        lo[1] lo[2];
        hi[1] lo[2];
        hi[1] hi[2];
        lo[1] hi[2];
        lo[1] lo[2];
    ]
end


function err_ellipsis_path(μ::Vector{<:Real}, Σ::Matrix{<:Real}, confidence::Real = 0.68, npts = 256)
    σ_sqr, A = eigen(Hermitian(Σ))
    σ = sqrt.(σ_sqr)
    ϕ = range(0, stop = 2π, length = 100)
    σ_scaled = σ .* sqrt(invlogcdf(Chisq(2), log(confidence)))
    xy = hcat(σ_scaled[1] * cos.(ϕ), σ_scaled[2] * sin.(ϕ)) * [A[1,1] A[1,2]; A[2,1] A[2,2]]
    xy .+ μ'
end


@recipe function f(samples::DensitySampleVector, parsel::NTuple{2,Integer})
    pi_x, pi_y = parsel

    flat_params = flatview(samples.params)

    acc = findall(x -> x > 0, samples.weight)
    rej = findall(x -> x <= 0, samples.weight)

    base_markersize = get(plotattributes, :markersize, 1.5)
    seriestype = get(plotattributes, :seriestype, :scatter)

    plot_bounds = get(plotattributes, :bounds, true)
    delete!(plotattributes, :bounds)

    if seriestype == :scatter
        color = parse(RGBA{Float64}, get(plotattributes, :seriescolor, :green))
        label = get(plotattributes, :label, isempty(rej) ? "samples" : "accepted")

        @series begin
            seriestype := :scatter
            label := label
            markersize := [w < 1 ? base_markersize : base_markersize * sqrt(w) for w in samples.weight[acc]]
            markerstrokewidth := 0
            color := [w >= 1 ? color : RGBA(convert(RGB, color), color.alpha * w) for w in samples.weight[acc]]
            xlabel --> "\$\\theta_$(pi_x)\$"
            ylabel --> "\$\\theta_$(pi_y)\$"
            (flat_params[pi_x, acc], flat_params[pi_y, acc])
        end

        if !isempty(rej)
            @series begin
                seriestype := :scatter
                label := "rejected"
                markersize := base_markersize
                markerstrokewidth := 0
                color := :red
                (flat_params[pi_x, rej], flat_params[pi_y, rej])
            end
        end
    elseif seriestype == :histogram2d
        @series begin
            seriestype := :histogram2d
            label --> "samples"
            xlabel --> "\$\\theta_$(pi_x)\$"
            ylabel --> "\$\\theta_$(pi_y)\$"
            weights := samples.weight[:]
            (flat_params[pi_x, :], flat_params[pi_y, :])
        end
    else
        error("seriestype $seriestype not supported")
    end

    nothing
end


@recipe function f(bounds::HyperRectBounds, parsel::NTuple{2,Integer})
    pi_x, pi_y = parsel

    vol = spatialvolume(bounds)
    vhi = vol.hi[[pi_x, pi_y]]; vlo = vol.lo[[pi_x, pi_y]]
    rect_xy = rectangle_path(vlo, vhi)
    bext = 0.1 * (vhi - vlo)
    xlims = (vlo[1] - bext[1], vhi[1] + bext[1])
    ylims = (vlo[2] - bext[2], vhi[2] + bext[2])

    @series begin
        seriestype := :path
        label --> "bounds"
        color --> :darkred
        alpha --> 0.75
        linewidth --> 2
        xlims --> xlims
        ylims --> ylims
        (rect_xy[:,1], rect_xy[:,2])
    end

    nothing
end


@recipe function f(stats::MCMCBasicStats, parsel::NTuple{2,Integer})
    pi_x, pi_y = parsel

    Σ_all = stats.param_stats.cov
    Σ = [Σ_all[pi_x, pi_x] Σ_all[pi_x, pi_y]; Σ_all[pi_y, pi_x] Σ_all[pi_y, pi_y]]

    μ = stats.param_stats.mean[[pi_x, pi_y]]
    mode_xy = stats.mode[[pi_x, pi_y]]

    conf = standard_confidence_vals

    color --> :darkviolet
    alpha --> 0.75
    linewidth --> 2
    
    for i in eachindex(conf)
        xy = err_ellipsis_path(μ, Σ, conf[i])
        @series begin
            seriestype := :path
            label --> "$(100 *conf[i])%"
            (xy[:,1], xy[:,2])
        end
    end

    markercolor --> :white
    markeralpha --> 0
    markersize --> 7
    markerstrokecolor --> :black
    markerstrokealpha --> 1
    markerstrokewidth --> 2

    @series begin
        seriestype := :scatter
        label := "mean"
        markershape := :circle
        ([μ[1]], [μ[2]])
    end

    @series begin
        seriestype := :scatter
        label := "mode"
        markershape := :rect
        ([mode_xy[1]], [mode_xy[2]])
    end

    vlo = stats.param_stats.minimum[[pi_x, pi_y]]
    vhi = stats.param_stats.maximum[[pi_x, pi_y]]
    rect_xy = rectangle_path(vlo, vhi)

    @series begin
        seriestype := :path
        label --> "bbox"
        (rect_xy[:,1], rect_xy[:,2])
    end

    nothing
end
