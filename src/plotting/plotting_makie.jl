using ValueShapes
using ArraysOfArrays
using StatsBase
using Distributions
using DataStructures
using ColorSchemes

using FillArrays

using StatsBase
using LinearAlgebra

using Makie
using Makie.Colors
import Makie: plot, plot!

# ------------ Type Defs: ----------

export plot_cfg_1d, quantile_hist1d, kde1d, mean1d
export plot_cfg_2d, quantile_hist2d, scatter2d, cov2d, mean2d, hexbin2d

abstract type plot_cfg_1d end
abstract type plot_cfg_2d end

@kwdef struct quantile_hist1d <: plot_cfg_1d
    levels=cdf(Chi(1), 0:3)
    nbins=30
    cmap=:Blues
    rev=true
end

@kwdef struct kde1d <: plot_cfg_1d
    bandwidth=Makie.automatic
    color=Makie.wong_colors()[1]
end

@kwdef struct mean1d <: plot_cfg_1d
    color=:black
    linestyle=:dot
end

@kwdef struct quantile_hist2d <: plot_cfg_2d
    levels=cdf(Chi(2), 0:3)
    nbins=(30, 30)
    cmap=:Blues
    rev=true
end

@kwdef struct scatter2d <: plot_cfg_2d
    color=Makie.wong_colors()[1]
end

@kwdef struct cov2d <: plot_cfg_2d
    color=:red
end

@kwdef struct mean2d <: plot_cfg_2d
    color=:black
    linestyle=:dot
end

@kwdef struct hexbin2d <: plot_cfg_2d
    nbins=(30,30)
    cmap=:Blues
    rev=true
end

# ----------------------------


# --------- Implementation of single plots --------

function plot2d!(ax::Axis, cfg::scatter2d, x::AbstractArray, y::AbstractArray, w::Union{Real, AbstractArray}=1)
    scatter!(ax, x, y, markersize=sqrt.(w./mean(w)), color=cfg.color)
end

function plot2d!(ax::Axis, cfg::hexbin2d, x::AbstractArray, y::AbstractArray)
    hexbin!(ax, x, y,colormap=cfg.cmap, bins=cfg.nbins)
end

function plot2d!(ax::Axis, cfg::hexbin2d, x::AbstractArray, y::AbstractArray, w::AbstractArray)
    Makie.hexbin!(ax, x, y, weights=w, colormap=cfg.cmap, bins=cfg.nbins, threshold=minimum(w[w .> 0]))
end

function plot2d!(ax::Axis, cfg::quantile_hist2d, x::AbstractArray, y::AbstractArray, w::Union{Real, AbstractArray}=1)
    # Compute 2D histogram
    edges_x = LinRange(minimum(x), maximum(x), cfg.nbins[1]+1)
    edges_y = LinRange(minimum(y), maximum(y), cfg.nbins[2]+1)
    h = fit(Histogram, (x, y), weights(w), (edges_x, edges_y))
    p = h.weights ./ sum(h.weights)  # Normalize to probability

    # Flatten and sort by descending probability
    flat_p = vec(p)
    idx = reverse(sortperm(flat_p))
    c = cumsum(flat_p[idx])

    # Map cumulative probabilities to colors
    pal = cgrad(cfg.cmap, length(cfg.levels), categorical=true, rev=cfg.rev)
    inds = clamp.(searchsortedlast.(Ref(cfg.levels), c), 1, length(pal))
    flat_colors = pal[inds]
    
    # Reassign colors to original grid order
    color_grid = reshape(pal[inds][sortperm(idx)], cfg.nbins)
    #show color_grid
    color_grid[h.weights .== 0.] .= RGBA(0,0,0,0)
    # Plot as heatmap
    heatmap!(ax, h.edges[1], h.edges[2], color_grid)
end

function plot2d!(ax::Axis, cfg::cov2d, x::AbstractArray, y::AbstractArray, w::Union{Real, AbstractArray}=1)
    # 2D data matrix: each row is a sample, each column is a variable
    data = hcat(x, y)  # 100 × 2
    
    # Mean and covariance
    μ = mean(data, ProbabilityWeights(w), dims=1) |> vec  # Make it a vector of length 2
    Σ = cov(data, ProbabilityWeights(w))
    
    # Eigen-decomposition
    eigvals, eigvecs = eigen(Σ)
    stds = sqrt.(clamp.(eigvals, 0, Inf))  # protect against negative values
    
    # Unit circle points
    θ = range(0, 2π, length=200)
    circle = [cos.(θ)'; sin.(θ)']  # 2 × 200 matrix
    
    # Transform unit circle into ellipse
    ellipse = eigvecs * Diagonal(stds) * circle .+ μ  # 2 × 200 matrix + 2-vector
    
    lines!(ax, ellipse[1, :], ellipse[2, :], color=cfg.color, linewidth=2, )

    
    for i in 1:2
        direction = eigvecs[:, i] * stds[i] * 3  # 3σ scaling for visibility
        p1 = μ .- direction
        p2 = μ .+ direction
        lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]],
               color=cfg.color, linewidth=2)
    end
end

function plot2d!(ax::Axis, cfg::mean2d, x::AbstractArray, y::AbstractArray, w::Union{Real, AbstractArray}=1)
    data = hcat(x, y)
    μ = mean(data, ProbabilityWeights(w), dims=1) |> vec  # Make it a vector of length 2
    hlines!(ax, [μ[2]], color=cfg.color, linestyle=cfg.linestyle)
    vlines!(ax, [μ[1]], color=cfg.color, linestyle=cfg.linestyle)
end

function plot1d!(ax::Axis, cfg::mean1d, x::AbstractArray, w::Union{Real, AbstractArray}=1)
    μ = mean(x, ProbabilityWeights(w))
    vlines!(ax, μ, color=cfg.color, linestyle=cfg.linestyle)
end

function plot1d!(ax::Axis, cfg::quantile_hist1d, x::AbstractArray, w::Union{Real, AbstractArray}=1)
    edges = LinRange(minimum(x), maximum(x), cfg.nbins)
    h = fit(Histogram, x, weights(w), edges)
    p = h.weights ./ sum(h.weights)
    idx = reverse(sortperm(p))
    c = cumsum(p[idx])

    # Get color palette
    pal = cgrad(cfg.cmap, length(cfg.levels), categorical=true, rev=cfg.rev)

    # Assign color to each bin based on cumulative probability
    inds = clamp.(searchsortedlast.(Ref(cfg.levels), c), 1, length(pal))
    colors = pal[inds][sortperm(idx)]
    # Plot histogram bars with assigned colors
    barplot!(ax, h.edges[1][1:end-1], p; width=diff(h.edges[1]), color=colors, gap=0,
             strokecolor=:black, strokewidth=0.)
end

function plot1d!(ax::Axis, cfg::kde1d, x::AbstractArray, w::Union{Real, AbstractArray}=1)
    density!(ax, x, weights=w, color=cfg.color, bandwidth=cfg.bandwidth)
end

function plot1d(cfg::plot_cfg_1d, x::AbstractArray, w::Union{Real, AbstractArray})
    fig = Figure()
    ax = Axis(fig[1,1])
    p = plot1d!(ax, cfg, x, w)
    Makie.FigureAxisPlot(fig, ax, p)
end

function plot2d(cfg::plot_cfg_2d, x::AbstractArray, y::AbstractArray, w::Union{Real, AbstractArray})
    fig = Figure()
    ax = Axis(fig[1,1])
    p = plot2d!(ax, cfg, x, y, w)
    Makie.FigureAxisPlot(fig, ax, p)
end

# ----------------------------------------- 


# ----------- Grid plots ----------

function plot_diagonal!(ax_grid, cfg, x, w)
    variables = collect(keys(x))
    N = length(variables)
    for i in 1:N
        var = variables[i]
        ax = ax_grid[i, i]
        plot1d!(ax, cfg, x[var], w)
    end
    ax_grid
end

function plot_lower_triangle!(ax_grid, cfg, x, w)
    variables = collect(keys(x))
    N = length(variables)
    for i in 1:N
        var1 = variables[i]
        for j in i+1:N
            var2 = variables[j]
            ax = ax_grid[j, i]
            plot2d!(ax, cfg, x[var1], x[var2], w)
        end
    end
    ax_grid
end  


function plot_upper_triangle!(ax_grid, cfg, x, w)
    variables = collect(keys(x))
    N = length(variables)
    for i in 1:N
        var1 = variables[i]
        for j in i+1:N
            var2 = variables[j]
            ax = ax_grid[i, j]
            plot2d!(ax, cfg, x[var2], x[var1], w)
        end
    end
    ax_grid
end 

"""Helper funtion to flatten sample vectors"""
function flatten(samples::DensitySampleVector)
    variables = keys(varshape(samples))
    ls = [length(varshape(samples)[var]) for var in variables]
    
    x = OrderedDict{String, AbstractArray}()
    
    for i in 1:length(variables)
        var = variables[i]
        col = flatview(getproperty(samples.v, var))
        if col isa Fill
            continue
        else
            for k in 1:ls[i]
                if ls[i] > 1
                    xlabel = String(var) * "[$k]"
                    x[xlabel] = col[k, :]
                else
                    xlabel=String(var)
                    x[xlabel] = col
                end
            end
        end
    end
    return x
end

function make_flat_samples(samples::DensitySampleVector, variables=nothing)
    x = flatten(samples)

    if variables isa AbstractDict
        x = OrderedDict(variables[var]=>x[var] for var in keys(variables))
        variables = nothing
    end
    
    if isnothing(variables)
        variables = collect(keys(x))
    end

    return x, variables
end


function make_visible!(ax, visible=true)
    ax.bottomspinevisible = visible
    ax.leftspinevisible = visible
    ax.rightspinevisible = visible
    ax.topspinevisible = visible

    ax.subtitlevisible = visible
    ax.titlevisible = visible
    
    ax.xgridvisible = visible
    ax.xticksvisible= visible
    ax.xticklabelsvisible = visible
    ax.xlabelvisible = visible

    ax.ygridvisible = visible
    ax.yticksvisible= visible
    ax.yticklabelsvisible = visible
    ax.ylabelvisible = visible

    for p in ax.scene.plots
        p.visible = visible
    end

    ax
end


function upper_triangle_visible!(fig, visible=true)
    N = fig.layout.size[1] - 2
    ax_grid = [only(contents(fig[i, j])) for i in 2:N+1, j in 2:N+1]
    for i in 1:N
        for j in i+1:N
            ax = ax_grid[i, j]
            make_visible!(ax, visible)

            if j < N
                ax.yticklabelsvisible=false
                ax.ylabelvisible=false
            end
            if i > 1
                ax.xticklabelsvisible=false
                ax.xlabelvisible=false
            end
        end
  
        only(contents(fig[1, i+1])).visible = visible
        if i < N
            only(contents(fig[i+1, N+2])).visible = visible
        end
    end

    if visible
        ax_grid[1,1].xaxisposition=:top
    else
        ax_grid[1,1].xaxisposition=:bottom        
    end
    ax_grid[1,1].xticklabelsvisible=visible
    ax_grid[1,1].xlabelvisible=visible
    
    fig
end

function lower_triangle_visible!(fig, visible=true)
    N = fig.layout.size[1] - 2
    ax_grid = [only(contents(fig[i, j])) for i in 2:N+1, j in 2:N+1]
    for i in 1:N
        for j in i+1:N
            ax = ax_grid[j, i]
            make_visible!(ax, visible)

            if j < N
                ax.xticklabelsvisible=false
                ax.xlabelvisible=false
            end
            if i > 1
                ax.yticklabelsvisible=false
                ax.ylabelvisible=false
            end
        end
  
        only(contents(fig[N+2, i+1])).visible = visible
        if i > 1
            only(contents(fig[i+1, 1])).visible = visible
        end
    end


    ax_grid[N,N].xticklabelsvisible=visible
    ax_grid[N,N].xlabelvisible=visible
    
    fig
end

function diagonal_visible!(fig, visible=true)
    N = fig.layout.size[1] - 2
    ax_grid = [only(contents(fig[i, j])) for i in 2:N+1, j in 2:N+1]
    for i in 1:N
        ax = ax_grid[i, i]
        make_visible!(ax, visible)
        ax.yticklabelsvisible=false
        ax.yticksvisible=false
        ylims!(ax, 0, nothing)
        ax.ygridvisible = false
        hidespines!(ax, :t, :l, :r)
        if 1 < i < N
            ax.xticklabelsvisible=false
            ax.xlabelvisible=false
        end
    end   
    fig
end


function plot!(fig, samples::DensitySampleVector; diagonal=nothing, lower_triangle=nothing, upper_triangle=nothing, variables=nothing)

    x, variables = make_flat_samples(samples, variables)
  
    N = length(variables)

    ax_grid = [only(contents(fig[i, j])) for i in 2:N+1, j in 2:N+1]
    
    if !(diagonal isa Nothing)
        plot_diagonal!(ax_grid, diagonal, x, samples.weight)
        diagonal_visible!(fig, true)
    end
    if !(lower_triangle isa Nothing)
        plot_lower_triangle!(ax_grid, lower_triangle, x, samples.weight)
        lower_triangle_visible!(fig, true)
    end
    if !(upper_triangle isa Nothing)
        plot_upper_triangle!(ax_grid, upper_triangle, x, samples.weight)
        upper_triangle_visible!(fig, true)
    end
    
    fig
end


function plot(samples::DensitySampleVector; diagonal=quantile_hist1d(), lower_triangle=quantile_hist2d(), upper_triangle=nothing, variables=nothing)

    x, variables = make_flat_samples(samples, variables)

    ranges = OrderedDict()
    for var in variables
        m = mean(x[var], weights(samples.weight))
        s = std(x[var], weights(samples.weight))
        ranges[var] = (max(m-4s, minimum(x[var])), min(m+4s, maximum(x[var])))
    end
    
    N = length(variables)

    fig = Figure(size=(N*200,N*200), fontsize=12)
        
    ax_grid = [Axis(fig[i, j], aspect=1) for i in 2:N+1, j in 2:N+1]
    
    if !(diagonal isa Nothing)
        plot_diagonal!(ax_grid, diagonal, x, samples.weight)
    end
    if !(lower_triangle isa Nothing)
        plot_lower_triangle!(ax_grid, lower_triangle, x, samples.weight)
    end
    if !(upper_triangle isa Nothing)
        plot_upper_triangle!(ax_grid, upper_triangle, x, samples.weight)
    end


    for i in 1:N
        ax = ax_grid[i,i]       
        var = variables[i]
        xlims!(ax, ranges[var])
        ax.xticklabelrotation = pi/2

        
        for j in i+1:N
            var2 = variables[j]
            ax2 = ax_grid[j, i]
            ax2.xticklabelrotation = pi/2
            ax3 = ax_grid[i, j]
            ax3.xaxisposition=:top
            ax3.yaxisposition=:right

            xlims!(ax2, ranges[var])
            xlims!(ax3, ranges[var2])
            ylims!(ax2, ranges[var2])
            ylims!(ax3, ranges[var])
        end

        # Add Labels
        #top
        Label(fig[1, i+1], var, fontsize=12, tellwidth=false)
        #left
        if i > 1
            Label(fig[i+1, 1], var, fontsize=12, rotation=π/2, tellheight=false)
        end
        #right
        if i < N
            Label(fig[i+1, N+2], var, fontsize=12, rotation=π/2, tellheight=false)
        end
        #bottom
        Label(fig[N+2, i+1], var, fontsize=12, tellwidth=false)

    end

    diagonal_visible!(fig, !(diagonal isa Nothing))
    upper_triangle_visible!(fig, !(upper_triangle isa Nothing))
    lower_triangle_visible!(fig, !(lower_triangle isa Nothing))

    fig
end

