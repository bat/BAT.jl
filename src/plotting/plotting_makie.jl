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

export plot_cfg_1d, hist1d, quantile_hist1d, kde1d, quantile_kde1d, mean1d, std1d, errorbars1d, pdf1d
export plot_cfg_2d, hist2d, quantile_hist2d, kde2d, quantile_kde2d, scatter2d, mean2d, std2d, cov2d, errorbars2d, hexbin2d

abstract type plot_cfg_1d end
abstract type plot_cfg_2d end


@kwdef struct kde1d <: plot_cfg_1d
    color = Makie.wong_colors()[1]
    alpha::Real = 1
    filled::Bool = true
    edge::Bool = false
    edgecolor = Makie.wong_colors()[1]
    edgewidth::Real = 1
end

@kwdef struct kde2d <: plot_cfg_2d
    cmap=:Blues
    rev::Bool = false
    alpha::Real = 1
end

@kwdef struct quantile_kde1d <: plot_cfg_1d
    levels::AbstractArray{<:Real} = cdf(Chi(1), 0:3)
    cmap=:Blues
    rev::Bool = false
    alpha::Real = 1
    edge::Bool = false
    edgecolor = Makie.wong_colors()[1]
    edgewidth::Real = 1
end

@kwdef struct quantile_kde2d <: plot_cfg_2d
    levels::AbstractArray{<:Real} = cdf(Chi(2), 0:3)
    nbins::Tuple{Int, Int} = (30, 30)
    cmap = :Blues
    rev::Bool = false
    alpha::Real = 1
end

@kwdef struct hist1d <: plot_cfg_1d
    nbins::Int = 30
    color = Makie.wong_colors()[1]
    alpha::Real = 1
    filled::Bool = true
    edge::Bool = false
    edgecolor = Makie.wong_colors()[1]
    edgewidth::Real = 1
end

@kwdef struct hist2d <: plot_cfg_2d
    cmap=:Blues
    nbins::Tuple{Int, Int} = (30, 30)
    rev::Bool = false
    alpha::Real = 1
end

@kwdef struct quantile_hist1d <: plot_cfg_1d
    levels::AbstractArray{<:Real} = cdf(Chi(1), 0:3)
    nbins::Int = 30
    cmap = :Blues
    rev::Bool = false
    alpha::Real = 1
    edge::Bool = false
    edgecolor = Makie.wong_colors()[1]
    edgewidth::Real = 1
end

@kwdef struct quantile_hist2d <: plot_cfg_2d
    levels::AbstractArray{<:Real} = cdf(Chi(2), 0:3)
    nbins::Tuple{Int, Int} = (30, 30)
    cmap = :Blues
    rev::Bool = false
    alpha::Real = 1
end

@kwdef struct scatter2d <: plot_cfg_2d
    color = Makie.wong_colors()[1]
    alpha::Real = 1
    size::Real = 1
end

@kwdef struct cov2d <: plot_cfg_2d
    color = :red
    linestyle = :solid
    linewidth::Real = 2
    nsigma::Real = 1.
end

@kwdef struct std1d <: plot_cfg_1d
    color = :red
    linestyle = :solid
    linewidth::Real = 2
    nsigma::Real = 1.
end

@kwdef struct std2d <: plot_cfg_2d
    color = :red
    linestyle = :solid
    linewidth::Real = 2
    nsigma::Real = 1.
end

@kwdef struct mean1d <: plot_cfg_1d
    color = :black
    linestyle = :dot
    linewidth::Real = 2
end

@kwdef struct mean2d <: plot_cfg_2d
    color = :black
    linestyle = :dot
    linewidth::Real = 2
end

@kwdef struct errorbars2d <: plot_cfg_2d
    color = :red
    linewidth::Real = 2
    nsigma::Real = 1.
    whiskerwidth::Real = 10
end

@kwdef struct errorbars1d <: plot_cfg_1d
    color = :red
    linewidth::Real = 2
    nsigma::Real = 1.
    whiskerwidth::Real = 10
    rel_y_pos::Real = 0.5
end

@kwdef struct hexbin2d <: plot_cfg_2d
    nbins::Tuple{Int, Int} = (30, 30)
    cmap = :Blues
    rev::Bool = true
    alpha::Real = 1
end


@kwdef struct pdf1d <: plot_cfg_1d
    color = Makie.wong_colors()[1]
    alpha::Real = 1
    filled::Bool = true
    edge::Bool = false
    edgecolor = Makie.wong_colors()[1]
    edgewidth::Real = 1
end

# ----------------------------


# --------- Implementation of single plots --------

function plot2d!(ax::Axis, cfg::scatter2d, x::AbstractArray, y::AbstractArray, w::Union{Real, AbstractArray}=1)
    scatter!(ax, x, y, markersize=sqrt.(w./mean(w))*cfg.size, color=cfg.color, alpha=cfg.alpha)
end

function plot2d!(ax::Axis, cfg::hexbin2d, x::AbstractArray, y::AbstractArray)
    hexbin!(ax, x, y,colormap=cfg.cmap, bins=cfg.nbins, alpha=cfg.alpha)
end

function plot2d!(ax::Axis, cfg::hexbin2d, x::AbstractArray, y::AbstractArray, w::AbstractArray)
    hexbin!(ax, x, y, weights=w, colormap=cfg.cmap, bins=cfg.nbins, threshold=minimum(w[w .> 0]), alpha=cfg.alpha)
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
    pal = cgrad(cfg.cmap, length(cfg.levels), categorical=true, rev=!cfg.rev, alpha=cfg.alpha)
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
    ellipse = eigvecs * cfg.nsigma * Diagonal(stds) * circle .+ μ  # 2 × 200 matrix + 2-vector
    
    lines!(ax, ellipse[1, :], ellipse[2, :], color=cfg.color, linestyle=cfg.linestyle, linewidth=cfg.linewidth)

    
    for i in 1:2
        direction = eigvecs[:, i] * stds[i] * 2 * cfg.nsigma  # 3σ scaling for visibility
        p1 = μ .- direction
        p2 = μ .+ direction
        lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]],
               color=cfg.color, linestyle=cfg.linestyle, linewidth=cfg.linewidth)
    end
end

function plot2d!(ax::Axis, cfg::mean2d, x::AbstractArray, y::AbstractArray, w::Union{Real, AbstractArray}=1)
    μ_x = mean(x, ProbabilityWeights(w))
    μ_y = mean(y, ProbabilityWeights(w))
    hlines!(ax, μ_y, color=cfg.color, linestyle=cfg.linestyle, linewidth=cfg.linewidth)
    vlines!(ax, μ_x, color=cfg.color, linestyle=cfg.linestyle, linewidth=cfg.linewidth)
end

function plot1d!(ax::Axis, cfg::mean1d, x::AbstractArray, w::Union{Real, AbstractArray}=1)
    μ = mean(x, ProbabilityWeights(w))
    vlines!(ax, μ, color=cfg.color, linestyle=cfg.linestyle, linewidth=cfg.linewidth)
end

function plot1d!(ax::Axis, cfg::std1d, x::AbstractArray, w::Union{Real, AbstractArray}=1)
    μ = mean(x, ProbabilityWeights(w))
    σ = std(x, ProbabilityWeights(w))
    vlines!(ax, [μ-cfg.nsigma*σ,μ+cfg.nsigma*σ] , color=cfg.color, linestyle=cfg.linestyle, linewidth=cfg.linewidth)
end

function plot2d!(ax::Axis, cfg::std2d, x::AbstractArray, y::AbstractArray, w::Union{Real, AbstractArray}=1)
    μ_x = mean(x, ProbabilityWeights(w))
    μ_y = mean(y, ProbabilityWeights(w))
    σ_x = std(x, ProbabilityWeights(w))
    σ_y = std(y, ProbabilityWeights(w))
    hlines!(ax, [μ_y-cfg.nsigma*σ_y, μ_y+cfg.nsigma*σ_y], color=cfg.color, linestyle=cfg.linestyle, linewidth=cfg.linewidth)
    vlines!(ax, [μ_x-cfg.nsigma*σ_x, μ_x+cfg.nsigma*σ_x], color=cfg.color, linestyle=cfg.linestyle, linewidth=cfg.linewidth)
end

function plot1d!(ax::Axis, cfg::errorbars1d, x::AbstractArray, w::Union{Real, AbstractArray}=1)
    μ = mean(x, ProbabilityWeights(w))
    σ = std(x, ProbabilityWeights(w))
    y_pos = ax.finallimits[].widths[2] * cfg.rel_y_pos
    errorbars!(ax, [μ,], [y_pos,], [σ*cfg.nsigma,], direction = :x, color=cfg.color, linewidth=cfg.linewidth, whiskerwidth=cfg.whiskerwidth)
end

function plot2d!(ax::Axis, cfg::errorbars2d, x::AbstractArray, y::AbstractArray, w::Union{Real, AbstractArray}=1)
    μ_x = mean(x, ProbabilityWeights(w))
    μ_y = mean(y, ProbabilityWeights(w))
    σ_x = std(x, ProbabilityWeights(w))
    σ_y = std(y, ProbabilityWeights(w))
    errorbars!(ax, [μ_x,], [μ_y,], [σ_y*cfg.nsigma,], color=cfg.color, linewidth=cfg.linewidth, whiskerwidth=cfg.whiskerwidth)
    errorbars!(ax, [μ_x,], [μ_y,], [σ_x*cfg.nsigma,], direction = :x, color=cfg.color, linewidth=cfg.linewidth, whiskerwidth=cfg.whiskerwidth)
end

function plot1d!(ax::Axis, cfg::quantile_hist1d, x::AbstractArray, w::Union{Real, AbstractArray}=1)
    edges = LinRange(minimum(x), maximum(x), cfg.nbins)
    h = fit(Histogram, x, weights(w), edges)
    p = h.weights ./ sum(h.weights)
    idx = reverse(sortperm(p))
    c = cumsum(p[idx])

    # Get color palette
    pal = cgrad(cfg.cmap, length(cfg.levels), categorical=true, rev=!cfg.rev, alpha=cfg.alpha)

    # Assign color to each bin based on cumulative probability
    inds = clamp.(searchsortedlast.(Ref(cfg.levels), c), 1, length(pal))
    colors = pal[inds][sortperm(idx)]
    # Plot histogram bars with assigned colors
    barplot!(ax, midpoints(edges), p ./diff(edges) ; width=diff(edges), color=colors, gap=0,
             strokecolor=:black, strokewidth=0.)
end
function plot1d!(ax::Axis, cfg::hist1d, x::AbstractArray, w::Union{Real, AbstractArray}=1)
    edges = LinRange(minimum(x), maximum(x), cfg.nbins)
    if cfg.filled
        hist!(ax, x, bins=edges, weights=w, normalization = :pdf, color=(cfg.color, cfg.alpha))
    end
    if cfg.edge
        stephist!(ax, x, bins=edges, weights=w, normalization = :pdf, color=cfg.edgecolor, linewidth=cfg.edgewidth)
    end
end

function plot2d!(ax::Axis, cfg::hist2d, x::AbstractArray, y::AbstractArray, w::Union{Real, AbstractArray}=1)
    # Compute 2D histogram
    edges_x = LinRange(minimum(x), maximum(x), cfg.nbins[1]+1)
    edges_y = LinRange(minimum(y), maximum(y), cfg.nbins[2]+1)
    h = fit(Histogram, (x, y), weights(w), (edges_x, edges_y))
    heatmap!(ax, edges_x, edges_y, h.weights, colormap=cfg.cmap, alpha=cfg.alpha)
end

function plot1d!(ax::Axis, cfg::kde1d, x::AbstractArray, w::Union{Real, AbstractArray}=1)
    k = kde(x, weights=w)
    if cfg.edge
        lines!(ax, k.x, k.density, color=cfg.edgecolor, linewidth=cfg.edgewidth)
    end
    if cfg.filled
        poly!(ax, vcat(k.x, reverse(k.x)), vcat(zeros(length(k.x)), reverse(k.density)), color=cfg.color, alpha=cfg.alpha)
    end
end

function plot2d!(ax::Axis, cfg::kde2d, x::AbstractArray, y::AbstractArray, w::Union{Real, AbstractArray}=1)
    k = kde((x, y), weights=w)
    heatmap!(ax, k.x, k.y, k.density, colormap=cfg.cmap, alpha=cfg.alpha)
end

function plot2d!(ax::Axis, cfg::quantile_kde2d, x::AbstractArray, y::AbstractArray, w::Union{Real, AbstractArray}=1)
    k = kde((x, y), weights=w)
    xgrid = k.x
    ygrid = k.y
    Z = k.density
    Z_flat = vec(Z)
    sorted_Z = sort(Z_flat, rev=true)
    cum = cumsum(sorted_Z)
    cum ./= cum[end]  # Normalize to 1
    
    # Target probability mass
    thresholds = [sorted_Z[searchsortedfirst(cum, level)] for level in cfg.levels]
    push!(thresholds, 0.)
    cmap = cgrad(cfg.cmap, rev=cfg.rev, alpha=cfg.alpha)

    contourf!(ax, xgrid, ygrid, Z, levels=thresholds[end:-1:1], colormap=cmap)
end


function plot1d!(ax::Axis, cfg::quantile_kde1d, x::AbstractArray, w::Union{Real, AbstractArray}=1)
    k = kde(x, weights=w)
    #lines!(ax, k.x, k.density)

    p = k.density * step(k.x)
    # Flatten and sort density values
    sorted_y = sort(p, rev=true)
    cum = cumsum(sorted_y)
    cum ./= cum[end]
    
    levels=copy(cfg.levels)
    push!(levels, 1.)
    pal = cgrad(cfg.cmap, length(levels), categorical=true, rev=!cfg.rev, alpha=cfg.alpha)
    thresholds = [sorted_y[searchsortedfirst(cum, level)] for level in levels]
    for i in length(thresholds):-1:1
        mask = p .>= thresholds[i]
        x_fill = k.x[mask]
        y_fill = k.density[mask]
        poly!(ax, vcat(x_fill, reverse(x_fill)), vcat(zeros(length(x_fill)), reverse(y_fill)), color=pal[i])
    end
end


function plot1d!(ax::Axis, cfg::pdf1d, d::Distribution)
    x = LinRange(ax.limits.val[1]..., 300)
    y = pdf(d, x)
    if cfg.edge
        lines!(ax, x, y, color=cfg.edgecolor, linewidth=cfg.edgewidth)
    end
    if cfg.filled
        poly!(ax, vcat(x, reverse(x)), vcat(zeros(length(x)), reverse(y)), color=cfg.color, alpha=cfg.alpha)
    end
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

function plot_diagonal!(ax_grid, cfg::pdf1d, d)
    variables = collect(keys(d))
    N = length(variables)
    for i in 1:N
        var = variables[i]
        ax = ax_grid[i, i]
        plot1d!(ax, cfg, d[var])
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

function flatten(dists::NamedTupleDist)
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

function flatten(prior::NamedTupleDist)
    variables = keys(prior)
    ls = [length(prior[var]) for var in variables]
    
    x = OrderedDict{String, Distribution}()
    
    for i in 1:length(variables)
        var = variables[i]
        col = prior[var]
        if col isa ValueShapes.ConstValueDist
            continue
        else
            if ls[i] > 1
                error("unimplemented")
            else
                x[String(var)] = prior[var]
            end
        end
    end
    return x
end

function make_flat_dict(collection::Union{DensitySampleVector, NamedTupleDist}, variables=nothing)
    x = flatten(collection)

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

    x, variables = make_flat_dict(samples, variables)
  
    N = length(variables)

    ax_grid = [only(contents(fig[i, j])) for i in 2:N+1, j in 2:N+1]
    
    if !isnothing(diagonal)
        plot_diagonal!(ax_grid, diagonal, x, samples.weight)
        diagonal_visible!(fig, true)
    end
    if !isnothing(lower_triangle)
        plot_lower_triangle!(ax_grid, lower_triangle, x, samples.weight)
        lower_triangle_visible!(fig, true)
    end
    if !isnothing(upper_triangle)
        plot_upper_triangle!(ax_grid, upper_triangle, x, samples.weight)
        upper_triangle_visible!(fig, true)
    end
    
    fig
end

function plot!(fig, prior::NamedTupleDist; diagonal=pdf1d(), variables=nothing)

    x, variables = make_flat_dict(prior, variables)
  
    N = length(variables)

    ax_grid = [only(contents(fig[i, j])) for i in 2:N+1, j in 2:N+1]
    
    plot_diagonal!(ax_grid, diagonal, x)
    diagonal_visible!(fig, true)

    fig
end

function plot(samples::DensitySampleVector; diagonal=quantile_hist1d(), lower_triangle=quantile_hist2d(), upper_triangle=nothing, variables=nothing, size=nothing, fontsize=12)

    x, variables = make_flat_dict(samples, variables)

    ranges = OrderedDict()
    for var in variables
        m = mean(x[var], weights(samples.weight))
        s = std(x[var], weights(samples.weight))
        ranges[var] = (max(m-4s, minimum(x[var])), min(m+4s, maximum(x[var])))
    end
    
    N = length(variables)

    fig = Figure(size = isnothing(size) ? (N*200,N*200) : size, fontsize=fontsize)
        
    ax_grid = [Axis(fig[i, j], aspect=1) for i in 2:N+1, j in 2:N+1]
    
    if !isnothing(diagonal)
        plot_diagonal!(ax_grid, diagonal, x, samples.weight)
    end
    if !isnothing(lower_triangle)
        plot_lower_triangle!(ax_grid, lower_triangle, x, samples.weight)
    end
    if !isnothing(upper_triangle)
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

    diagonal_visible!(fig, !isnothing(diagonal))
    upper_triangle_visible!(fig, !isnothing(upper_triangle))
    lower_triangle_visible!(fig, !isnothing(lower_triangle))

    fig
end

