# This file is a part of BAT.jl, licensed under the MIT License (MIT).

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


@recipe function f(stats::MCMCBasicStats, vsel::NTuple{2,Integer})
    pi_x, pi_y = vsel

    Σ_all = stats.param_stats.cov
    Σ = [Σ_all[pi_x, pi_x] Σ_all[pi_x, pi_y]; Σ_all[pi_y, pi_x] Σ_all[pi_y, pi_y]]

    μ = stats.param_stats.mean[[pi_x, pi_y]]
    mode_xy = stats.mode[[pi_x, pi_y]]

    conf = default_credibilities

    seriescolor --> :darkviolet
    seriesalpha --> 0.75
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

    # @series begin
    #     seriestype := :scatter
    #     label := "mean"
    #     markershape := :circle
    #     ([μ[1]], [μ[2]])
    # end

    # @series begin
    #     seriestype := :scatter
    #     label := "mode"
    #     markershape := :rect
    #     ([mode_xy[1]], [mode_xy[2]])
    # end

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
