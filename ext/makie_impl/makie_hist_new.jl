
@with_kw struct Hist1D <: BATMakieRecipe
    weights::Union{Vector{Integer},Nothing} = nothing
    nbins::Union{Tuple{Vararg{Integer}},Integer} = 30
    closed::Symbol = :left
    normalization::Symbol = :pdf
    filter::Bool = false
    color::Any = Makie.wong_colors()[1]
    alpha::Float64 = 1.0
    filled::Bool = true
    edge::Bool = false
    strokecolor::Any = Makie.wong_colors()[1]
    strokewidth::Float64 = 1.0
end

function compute_plotting_primitives(
    ::Nothing,
    ::Nothing,
    ::Hist1D
)
    return (centers=Vector{Float64}(), weights=Vector{Float64}(), widths=Vector{Float64}())
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Hist1D
)
    (; normalization, nbins, closed, filter) = recipe

    global gs_cp = (marg_coords, weights, filter, nbins, closed, normalization)
    BREAK

    hist = _marginal_view_dist(marg_coords, weights, filter, nbins, closed, normalization)

    centers = _get_bin_centers(hist)

    return (centers=centers, weights=hist.weights, widths=hist.edges[1])
end


function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Hist1D
)
    (; centers, weights, widths) = primitives
    (; color, alpha, filled, strokecolor, strokewidth, edge) = recipe

    bars = barplotspec((centers, widths);
        color=color,
        alpha=alpha,
        gap=0.0,
        width=widths,
        visible=filled
    )

    stairs = stairsspec((widths, vcat(weights, weights[end]));
        step=:post,
        color=strokecolor,
        linewidth=strokewidth,
        visible=edge
    )
    return [bars, stairs]
end


@with_kw struct Hist2D <: BATMakieRecipe
    weights::Union{Vector{Integer},Nothing} = nothing
    nbins::Union{Tuple{Vararg{Integer}},Integer} = (30, 30)
    closed::Symbol = :left
    normalization::Symbol = :pdf
    filter::Bool = false
    colormap::Symbol = :Blues
    alpha::Float64 = 1.0
    rev::Bool = false
end

function compute_plotting_primitives(
    ::Nothing,
    ::Nothing,
    ::Hist2D
)
    return (centers=Matrix{Float64}(), weights=Vector{Float64}())
end

function compute_plotting_primitves(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Hist2D
)
    (; normalization, nbins, closed, filter) = recipe

    hist = _marginal_view_dist(marg_coords, weights, filter, nbins, closed, normalization)

    centers = _get_bin_centers(hist)

    return (centers=centers, weights=h_norm.weights)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Hist2D
)

    final_cmap = rev ? Reverse(colormap) : colormap

    heat = heatmapspec(plot_data;
        colormap=final_cmap,
        alpha=p.alpha
    )
    return [heat]
end







@with_kw struct QuantileHist1D <: BATMakieRecipe
    weights = nothing
    nbins = 30
    closed = :left
    normalization = :pdf
    filter = false
    levels = cdf.(Chi(1), 0:3)
    colormap = :Blues
    rev = false
    alpha = 1.0
    edge = false
    strokecolor = Makie.wong_colors()[1]
    strokewidth = 1.0
end




@with_kw struct QuantileHist2D <: BATMakieRecipe
    weights = nothing
    nbins = (30, 30)
    closed = :left
    normalization = :pdf
    filter = false
    levels = cdf.(Chi(2), 0:3)
    colormap = :Blues
    rev = false
    alpha = 1.0
end












@with_kw struct Hexbin2D <: BATMakieRecipe
    weights = nothing
    nbins = (30, 30)
    filter = false
    colormap = :Blues
    rev = true
    alpha = 1.0
    threshold = nothing
end

function Makie.plot!(p::Hexbin2D)
    data = lift(p.samples, p.vsel, p.filter) do smpls, vsel, f
        marg = bat_marginalize(smpls, vsel)
        marg_res = marg.result

        if f
            marg_res = BAT.drop_low_weight_samples(marg_res)
        end

        w = marg_res.weight

        flat_vals = flatview(unshaped.(marg_res).v)
        x = flat_vals[1, :]
        y = flat_vals[2, :]

        return (x, y, w)
    end

    x_vals = lift(d -> d[1], data)
    y_vals = lift(d -> d[2], data)
    w_vals = lift(d -> d[3], data)

    final_thresh = lift(w_vals, p.threshold) do w, user_thresh
        thresh = if isnothing(user_thresh)
            pos_w = w[w.>0]
            isempty(pos_w) ? 0.0 : minimum(pos_w)
        else
            user_thresh
        end
        return thresh
    end

    final_cmap = lift(p.colormap, p.rev) do cm, r
        r ? Reverse(cm) : cm
    end

    final_bins = lift(p.nbins) do b
        b isa Integer ? (b, b) : b
    end

    hexbin!(p, x_vals, y_vals;
        weights=w_vals,
        bins=final_bins,
        colormap=final_cmap,
        alpha=p.alpha,
        threshold=final_thresh
    )

    return p
end

