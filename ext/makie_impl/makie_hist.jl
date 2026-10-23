# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Keep numeric geometry in Float64 until Makie applies its coordinate transform.
# Early Float32 conversion can erase unit-scale variation around large offsets.
# Fresh sentinels per call, not shared consts -- see _empty_scatter2d_primitives (makie_scatter.jl).
_empty_hist1d_primitives() = (centers=Vector{Float64}(), weights=Vector{Float64}(), edges=Vector{Float64}())
_empty_hist2d_primitives() = (xedges=Vector{Float64}(), yedges=Vector{Float64}(), weights=Matrix{Float64}(undef, 0, 0))
_empty_quantilehist1d_primitives() = (polys=Vector{Point{2,Float64}}[], fill_colors=Vector{RGBA{Float32}}(), stairs_data=Vector{Point{2,Float64}}())
_empty_quantilehist2d_primitives() = (xedges=Vector{Float64}(), yedges=Vector{Float64}(), color_grid=Matrix{RGBA{Float32}}(undef, 0, 0))
_empty_hexbin2d_primitives() = (x=Float64[], y=Float64[], weights=Float64[], thresh=0.0)

# Bin edges use the supplied per-dimension domain when available,
# and sample coordinates otherwise. The graph preserves the domain across
# window changes and may replace it during live publication.
function _hist_edges(::Nothing, cols, bins::Integer, closed::Symbol)
    return _get_edges(cols, (bins,), closed)
end
function _hist_edges(::Nothing, cols, bins::Tuple, closed::Symbol)
    return Tuple(_get_edges(cols[i], bins[i], closed) for i in 1:length(bins))
end
function _hist_edges(domain::Tuple{Float64,Float64}, cols, bins::Integer, closed::Symbol)
    return _get_edges(([domain[1], domain[2]],), (bins,), closed)
end
function _hist_edges(domain::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}}, cols, bins::Tuple, closed::Symbol)
    (dlo1, dhi1), (dlo2, dhi2) = domain
    return _get_edges(([dlo1, dhi1], [dlo2, dhi2]), bins, closed)
end

function _marginal_view_dist(
    locations::SubArray,
    weights::SubArray,
    filter::Bool,
    bins::Union{Tuple{Vararg{Integer}},Integer},
    closed::Symbol,
    normalization::Symbol,
    domain = nothing,
)
    BAT._plot_histogram_mode(normalization)
    BAT._plot_validate_weights(weights)
    BAT._plot_validate_coordinates(Tuple(eachrow(locations)), weights)
    if filter
        mask = _low_weight_mask(weights)
        locations = view(locations, :, mask)
        weights = view(weights, mask)
    end
    cols = Tuple(eachrow(locations))
    edges = _hist_edges(domain, cols, bins, closed)
    return BAT._plot_fit_histogram(cols, weights, edges, normalization; closed=closed)
end

_low_weight_mask(weights::AbstractVector, fraction::Real=1e-5, threshold::Real=1e-2) =
    BAT._plot_low_weight_mask(weights, fraction, threshold)

function _get_bin_centers(hist::Histogram)
    edges = hist.edges
    dims = ndims(hist.weights)

    # Explicitly typed comprehensions: an untyped comprehension over <= 1 edges
    # infers Vector{Any}, violating the graph's TypedEdge live/dead type contract.
    centers = Vector{Float64}[
        Float64[BAT._plot_float64(edges[d][i] + 0.5 * (edges[d][i+1] - edges[d][i]), "histogram center") for i in 1:length(edges[d])-1]
        for d in 1:dims
    ]

    return centers
end

# Preserve zero-mass histograms as empty displays without dividing by zero.
# The shared histogram normalizer owns validation and normalization.
_safe_normalize(hist::Histogram, normalization::Symbol) =
    BAT._plot_normalize_histogram(hist, normalization)

# Heatmap cell boundaries and polygon corners must come from the original
# edges. Midpoint reconstruction changes unequal bins, and even a one-bin
# midpoint can be unrepresentable when both original endpoints are Float64.
function _histogram_plot_edges(hist::Histogram)
    return map(hist.edges) do original
        edges = map(e -> BAT._plot_float64(e, "histogram edge"), collect(original))
        all(w -> isfinite(w) && w > 0, diff(edges)) ||
            BAT._plot_weight_error("histogram edges are not distinct with finite positive widths as Float64")
        edges
    end
end

# Zero is the default inactive sentinel; one requests the full positive-mass
# region. Validate before empty/dead-cell returns as well as before selection.
# Duplicates remain distinct requests and count toward the fixed palette size.
function _histogram_region_levels(levels)
    levels isa Union{AbstractVector,Tuple} || throw(ArgumentError(
        "Histogram credible levels must be a vector or tuple of probabilities"))
    all(p -> p isa Real && isfinite(p) && 0 <= p <= 1, levels) ||
        throw(ArgumentError("Histogram credible levels must be finite probabilities in [0, 1]"))
    active = Float64[BAT._plot_float64(p, "histogram credible level") for p in levels if p > 0]
    length(active) <= length(_QUANTILE_LEVEL_COLORS) || throw(ArgumentError(
        "Histogram credible regions support at most $(length(_QUANTILE_LEVEL_COLORS)) active levels"))
    return sort!(active)
end

function _hist1d_output(hist::Histogram, normalization::Symbol)
    h_norm = _safe_normalize(hist, normalization)
    centers = _get_bin_centers(h_norm)
    # Float64.(...): with :none normalization the weights keep the samples' own
    # eltype, mismatching the empty sentinel's Vector{Float64} (TypedEdge).
    return (centers=centers[1], weights=map(w -> BAT._plot_float64(w, "histogram height"), h_norm.weights), edges=map(e -> BAT._plot_float64(e, "histogram edge"), collect(h_norm.edges[1])))
end

function _hist2d_output(hist::Histogram, normalization::Symbol)
    h_norm = _safe_normalize(hist, normalization)
    xedges, yedges = _histogram_plot_edges(h_norm)
    # Zero-height bins become NaN for transparency. Positive heights are checked
    # and converted to Float64 to match the graph's output representation.
    weights = map(w -> w > 0 ? BAT._plot_float64(w, "histogram height") : NaN, h_norm.weights)
    return (xedges=xedges, yedges=yedges, weights=weights)
end

function _quantilehist1d_output(hist::Histogram, config::NamedTuple)
    (; normalization, levels) = config
    valid_intervals = _histogram_region_levels(levels)
    h_norm = _safe_normalize(hist, normalization)
    sub_hists, _ = BAT.get_smallest_intervals(h_norm, valid_intervals)

    bin_colors = fill(RGBA{Float32}(0, 0, 0, 0), length(h_norm.weights))

    for (i, sub_hist) in enumerate(sub_hists)
        c = _quantile_level_color(i)
        mask = sub_hist.weights .> 0
        bin_colors[mask] .= c
    end

    heights = map(w -> BAT._plot_float64(w, "histogram height"), h_norm.weights)
    edges = only(_histogram_plot_edges(h_norm))
    isempty(heights) && return _empty_quantilehist1d_primitives()
    polys = Vector{Point{2,Float64}}[]
    fill_colors = RGBA{Float32}[]
    for i in eachindex(heights)
        # Do not send degenerate zero-height polygons or invisible cells to
        # triangulation. The complete stairs still include their bin edges.
        if heights[i] > 0 && bin_colors[i].alpha > 0
            lo, hi, height = edges[i], edges[i+1], heights[i]
            push!(polys, Point{2,Float64}[(lo, 0), (hi, 0), (hi, height), (lo, height)])
            push!(fill_colors, bin_colors[i])
        end
    end
    stairs_y = vcat(heights, heights[end])
    stairs_data = Point{2,Float64}.(edges, stairs_y)

    return (polys=polys, fill_colors=fill_colors, stairs_data=stairs_data)
end

function _quantilehist2d_output(hist::Histogram, config::NamedTuple)
    (; normalization, levels) = config
    valid_intervals = _histogram_region_levels(levels)
    h_norm = _safe_normalize(hist, normalization)
    sub_hists, _ = BAT.get_smallest_intervals(h_norm, valid_intervals)

    dims = size(h_norm.weights)
    color_grid = fill(RGBA{Float32}(0, 0, 0, 0), dims)

    for (i, sub_hist) in enumerate(sub_hists)
        c = _quantile_level_color(i)
        mask = sub_hist.weights .> 0
        color_grid[mask] .= c
    end

    xedges, yedges = _histogram_plot_edges(h_norm)
    return (xedges=xedges, yedges=yedges, color_grid=color_grid)
end

function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::Hist1D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _empty_hist1d_primitives()
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Hist1D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    # A live cell can still have zero samples -- degrade like a dead cell.
    BAT._plot_histogram_mode(config.normalization)
    isempty(weights) && return _empty_hist1d_primitives()
    (; normalization, nbins, closed, filter) = config
    # nbins + 1, unlike every other hist recipe's plain nbins -- deliberate asymmetry.
    hist = _marginal_view_dist(marg_coords, weights, filter, nbins + 1, closed, config.normalization, get(config, :domain, nothing))
    return _hist1d_output(hist, :none)
end


# Bound endpoint displacement in units of the styled bin width, not the
# absolute coordinate. This implies at most twice this relative width error.
const _HIST1D_BAR_GEOMETRY_RTOL = 1 // (BigInt(1) << 18) # 32eps(Float32)
_hist1d_bar_geometry_error() = throw(ArgumentError(
    "Hist1D bin geometry is not representable accurately by native bars; use QuantileHist1D or recenter coordinates"))

function _checked_hist1d_bar_layout(
    edges::AbstractVector{Float64}, gap, dodge, n_dodge, dodge_gap, requested_width
)
    requested_width === Makie.automatic || throw(ArgumentError("Hist1D width is determined by bin edges"))
    gap isa Real && dodge_gap isa Real && isfinite(gap) && isfinite(dodge_gap) || _hist1d_bar_geometry_error()
    if dodge !== Makie.automatic
        (dodge isa Integer || dodge isa AbstractVector{<:Integer}) ||
            throw(ArgumentError("Hist1D dodge indices must be integers"))
        dodge = dodge isa Integer ? Int(dodge) : Int.(dodge)
        all(>(0), dodge) || throw(ArgumentError("Hist1D dodge indices must be positive"))
    end
    if n_dodge !== Makie.automatic
        n_dodge isa Integer || throw(ArgumentError("Hist1D n_dodge must be an integer"))
        n_dodge = Int(n_dodge)
        n_dodge > 0 || throw(ArgumentError("Hist1D n_dodge must be positive"))
    end

    n = length(edges) - 1
    n >= 1 && all(isfinite, edges) || _hist1d_bar_geometry_error()
    widths = diff(edges)
    all(w -> isfinite(w) && w > 0, widths) || _hist1d_bar_geometry_error()
    centers = edges[1:end-1] .+ 0.5 .* widths
    actual_x, actual_width = Makie.compute_x_and_width(centers, widths, gap, dodge, n_dodge, dodge_gap)

    # Explicit widths alone cannot repair rounded midpoints at large offsets.
    # Native layout arithmetic on exact edges supplies the reference without
    # changing global BigFloat precision. Preserve native scalar style types
    # above: coercing Float32 gaps to Float64 changes native subtraction.
    exact_edges = Rational{BigInt}.(edges)
    exact_widths = diff(exact_edges)
    exact_centers = (exact_edges[1:end-1] .+ exact_edges[2:end]) ./ 2
    exact_x, exact_width = Makie.compute_x_and_width(exact_centers, exact_widths,
        Rational{BigInt}(gap), dodge, n_dodge, Rational{BigInt}(dodge_gap))
    # Fail closed if a future native helper starts rounding the reference.
    eltype(exact_x) <: Rational{BigInt} && eltype(exact_width) <: Rational{BigInt} ||
        error("Hist1D native bar layout must preserve exact reference arithmetic")
    for i in 1:n
        rectangle = Makie.bar_rectangle(actual_x[i], 0.0, actual_width[i], 0.0, true)
        lo = rectangle.origin[1]
        hi = lo + rectangle.widths[1]
        isfinite(lo) && isfinite(hi) || _hist1d_bar_geometry_error()
        halfwidth = abs(exact_width[i]) / 2
        exact_lo, exact_hi = exact_x[i] - halfwidth, exact_x[i] + halfwidth
        # A gap may intentionally hide the bar by making its width zero.
        scale = iszero(halfwidth) ? exact_widths[i] : 2halfwidth
        displacement = max(abs(Rational{BigInt}(lo) - exact_lo), abs(Rational{BigInt}(hi) - exact_hi))
        displacement <= _HIST1D_BAR_GEOMETRY_RTOL * scale || _hist1d_bar_geometry_error()
    end
    return centers, widths, gap, dodge, n_dodge, dodge_gap
end

# Keep native BarPlot's rendering, labels and theme namespace. Only its bin
# geometry is BAT-owned, following the checked Hexbin adapter below.
Makie.@recipe BATCheckedHistBars (bin_edges, bin_heights) begin
    Makie.documented_attributes(Makie.BarPlot)...
end
Makie.add_theme!(::Type{<:BATCheckedHistBars}, kw, graph::Makie.ComputeGraph, scene::Makie.Scene) =
    Makie.add_theme!(Makie.BarPlot, kw, graph, scene)

function Makie.plot!(p::BATCheckedHistBars)
    # Publish every geometry control through the same check. Heights alone
    # neither invalidate the layout cache nor repeat exact arithmetic.
    Makie.map!(p, [:bin_edges, :gap, :dodge, :n_dodge, :dodge_gap, :width],
        [:checked_centers, :checked_widths, :checked_gap, :checked_dodge, :checked_n_dodge, :checked_dodge_gap]
    ) do edges, gap, dodge, n_dodge, dodge_gap, width
        _checked_hist1d_bar_layout(edges, gap, dodge, n_dodge, dodge_gap, width)
    end
    Makie.barplot!(p, Makie.Attributes(p), p.checked_centers, p.bin_heights;
        width=p.checked_widths, gap=p.checked_gap, dodge=p.checked_dodge,
        n_dodge=p.checked_n_dodge, dodge_gap=p.checked_dodge_gap)
    return p
end
function _validate_plot_display(p::BATCheckedHistBars)
    p.checked_centers[]
    return nothing
end
Makie.extract_colormap(p::BATCheckedHistBars) = Makie.extract_colormap(only(p.plots))
Makie.data_limits(p::BATCheckedHistBars) = Makie.data_limits(only(p.plots))
Makie.boundingbox(p::BATCheckedHistBars, space::Symbol=:data) = Makie.boundingbox(only(p.plots), space)


function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Hist1D,
    config::NamedTuple
)
    (; weights, edges) = primitives

    if isempty(weights)
        return PlotSpec[]
    end

    # Bin edges own width, including when a caller themes BarPlot.width.
    bars = Makie.PlotSpec(BATCheckedHistBars, edges, weights; width=Makie.automatic)
    stairs = S.Stairs(edges, vcat(weights, weights[end]); step=:post)

    return [bars, stairs]
end

# Peak height for a diagonal cell's own y limit. The zero fallback requests
# automatic limits for recipes without a positive finite extent.
_diag_y_extent(::NamedTuple, ::BATMakieRecipe) = 0.0
_diag_y_extent(primitives::NamedTuple, ::Hist1D) = isempty(primitives.weights) ? 0.0 : maximum(primitives.weights)

function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::Hist2D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _empty_hist2d_primitives()
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Hist2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    BAT._plot_histogram_mode(config.normalization)
    isempty(weights) && return _empty_hist2d_primitives()
    (; normalization, nbins, closed, filter) = config
    hist = _marginal_view_dist(marg_coords, weights, filter, nbins, closed, config.normalization, get(config, :domain, nothing))
    return _hist2d_output(hist, :none)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Hist2D,
    config::NamedTuple;
    transposed::Bool = false
)
    (; xedges, yedges, weights) = primitives

    if isempty(weights)
        return PlotSpec[]
    end

    # Forwarded as KDE2D and Hexbin2D already do. QuantileHist2D/QuantileKDE2D
    # deliberately do not: their heatmap cells are explicit RGBA colors, not
    # values a colormap could map.
    (; colormap, rev) = config
    cmap_final = rev ? Reverse(colormap) : colormap

    # Lower-triangle cells swap axes at compose time (see _init_gridlayout's
    # invariant comment). permutedims, not lazy transpose: the sibling methods
    # reuse this reasoning on RGBA grids, which recursive transpose has no
    # method for.
    all(w -> isnan(w) || 0 < Float32(w) < Inf, weights) || throw(ArgumentError(
        "Hist2D visible height is not representable as positive finite Float32 color values; change units/normalization or use QuantileHist2D"))
    heat = transposed ?
        S.Heatmap(yedges, xedges, permutedims(weights); colormap=cmap_final) :
        S.Heatmap(xedges, yedges, weights; colormap=cmap_final)
    return [heat]
end


function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::QuantileHist1D,
    recipe_status::RS,
    ::CS,
    config::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    recipe_status isa LiveRecipe && _histogram_region_levels(config.levels)
    return _empty_quantilehist1d_primitives()
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::QuantileHist1D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    BAT._plot_histogram_mode(config.normalization)
    _histogram_region_levels(config.levels)
    isempty(weights) && return _empty_quantilehist1d_primitives()
    (; nbins, closed, filter) = config
    hist = _marginal_view_dist(marg_coords, weights, filter, nbins, closed, config.normalization, get(config, :domain, nothing))
    return _quantilehist1d_output(hist, merge(config, (normalization=:none,)))
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::QuantileHist1D,
    config::NamedTuple
)
    (; polys, fill_colors, stairs_data) = primitives

    if isempty(stairs_data)
        return PlotSpec[]
    end

    plotspecs = PlotSpec[]
    isempty(polys) || push!(plotspecs, S.Poly(polys; color=fill_colors, strokewidth=0))
    stairs = S.Stairs(stairs_data; step=:post)
    push!(plotspecs, stairs)
    return plotspecs
end

_diag_y_extent(primitives::NamedTuple, ::QuantileHist1D) = isempty(primitives.stairs_data) ? 0.0 : maximum(p -> p[2], primitives.stairs_data)


function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::QuantileHist2D,
    recipe_status::RS,
    ::CS,
    config::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    recipe_status isa LiveRecipe && _histogram_region_levels(config.levels)
    return _empty_quantilehist2d_primitives()
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::QuantileHist2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    BAT._plot_histogram_mode(config.normalization)
    _histogram_region_levels(config.levels)
    isempty(weights) && return _empty_quantilehist2d_primitives()
    (; nbins, closed, filter) = config
    hist = _marginal_view_dist(marg_coords, weights, filter, nbins, closed, config.normalization, get(config, :domain, nothing))
    return _quantilehist2d_output(hist, merge(config, (normalization=:none,)))
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::QuantileHist2D,
    config::NamedTuple;
    transposed::Bool = false
)
    (; xedges, yedges, color_grid) = primitives
    if isempty(color_grid)
        return PlotSpec[]
    end
    # permutedims (not transpose) -- see Hist2D's matching comment above.
    heat = transposed ?
        S.Heatmap(yedges, xedges, permutedims(color_grid)) :
        S.Heatmap(xedges, yedges, color_grid)
    return [heat]
end


function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::Hexbin2D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _empty_hexbin2d_primitives()
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Hexbin2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    (; threshold) = config
    if !isnothing(threshold)
        threshold = BAT._plot_float64(threshold, "hexbin threshold")
        threshold >= 0 || BAT._plot_weight_error("hexbin threshold must be nonnegative")
    end
    # Zero-sample live cell: the placeholder view is 0x0, so row indexing would throw.
    isempty(weights) && return _empty_hexbin2d_primitives()
    BAT._plot_validate_weights(weights)
    BAT._plot_validate_coordinates(Tuple(eachrow(marg_coords)), weights)
    counts = map(w -> BAT._plot_float64(w, "hexbin count"), weights)
    BAT._plot_checked_total(counts)
    # Native hexbin uses every position to choose bounds and bin indices,
    # including zero-weight rows. Preserve that domain policy and reject
    # unusable positions before handing them to the backend.
    x = map(v -> BAT._plot_float64(v, "hexbin x coordinate"), marg_coords[1, :])
    y = map(v -> BAT._plot_float64(v, "hexbin y coordinate"), marg_coords[2, :])
    final_thresh = if isnothing(threshold)
        pos_w = counts[counts .> 0]
        isempty(pos_w) ? 0.0 : minimum(pos_w)
    else
        BAT._plot_float64(threshold, "hexbin threshold")
    end
    final_thresh >= 0 || BAT._plot_weight_error("hexbin threshold must be nonnegative")
    return (x=x, y=y, weights=counts, thresh=final_thresh)
end

function _check_hexbin_count_colors(counts::AbstractArray{<:Real})
    isempty(counts) && return counts
    fail() = throw(ArgumentError("Hexbin2D displayed counts exceed Float32 color range or resolution; rescaling count units and threshold together may help magnitude limits; otherwise use a normalized histogram or whole-bin region recipe"))
    for value in counts
        isfinite(value) && value >= 0 || fail()
        converted = Float32(value)
        isfinite(converted) && (iszero(value) || converted >= floatmin(Float32)) || fail()
    end
    lo, hi = extrema(counts)
    # Match native Hexbin's automatic range for uniform/zero counts.
    lo == hi && (lo = zero(lo); iszero(hi) && (hi = one(hi)))
    flo, fhi = Float32(lo), Float32(hi)
    span = Float64(hi) - Float64(lo)
    fspan = fhi - flo
    isfinite(span) && span > 0 && isfinite(fspan) && fspan >= floatmin(Float32) || fail()
    # Shaders may implement division by multiplying with a reciprocal. Keep
    # that reciprocal normal, with headroom at the flush-to-zero boundary.
    reciprocal_span = inv(fspan)
    isfinite(reciprocal_span) && reciprocal_span >= 2floatmin(Float32) || fail()
    # The scalar color coordinates must survive narrowing, including at a
    # large common offset. The allowance is in normalized colormap coordinates.
    tolerance = 4 * Float64(eps(Float32))
    for value in counts
        exact_position = (Float64(value) - Float64(lo)) / span
        numerator = Float32(value) - flo
        rendered_position = Float64(numerator / fspan)
        reciprocal_position = Float64(numerator * reciprocal_span)
        # GPU arithmetic may flush a subnormal subtraction to zero even when
        # the count and range endpoints themselves are normal.
        flushed_position = numerator < floatmin(Float32) ? 0.0 : rendered_position
        cpu_position = (Float64(Float32(value)) - Float64(flo)) / (Float64(fhi) - Float64(flo))
        max(abs(exact_position - rendered_position), abs(exact_position - cpu_position),
            abs(exact_position - flushed_position),
            abs(exact_position - reciprocal_position)) <= tolerance || fail()
    end
    return counts
end

# Internal adapter: native Hexbin owns aggregation, threshold and geometry.
# Its hidden renderer uses a harmless fixed range. Only checked counts reach
# the visible Scatter's Float32 color conversion.
Makie.@recipe BATCheckedHexbin begin
    Makie.documented_attributes(Makie.Hexbin)...
end
Makie.conversion_trait(::Type{<:BATCheckedHexbin}) = Makie.PointBased()
# Preserve the native Hexbin theme namespace despite the private adapter type.
# Explicit PlotSpec count range/scale and caller keywords retain precedence.
Makie.add_theme!(::Type{<:BATCheckedHexbin}, kw, graph::Makie.ComputeGraph, scene::Makie.Scene) =
    Makie.add_theme!(Makie.Hexbin, kw, graph, scene)

function Makie.plot!(p::BATCheckedHexbin)
    native = Makie.hexbin!(p, Makie.Attributes(p), p[1];
        visible=false, inspectable=false, colorscale=identity, colorrange=(0.0, 1.0))
    Makie.map!(p, [native.count_hex], [:checked_counts, :count_colorrange]) do counts
        _check_hexbin_count_colors(counts)
        # Native automatic Hexbin range, including empty/uniform cases.
        range = if isempty(counts)
            (0.0, 1.0)
        else
            lo, hi = extrema(counts)
            lo == hi ? (0.0, iszero(hi) ? 1.0 : hi) : (lo, hi)
        end
        return counts, range
    end
    Makie.scatter!(p, Makie.Attributes(p), native.points;
        color=p.checked_counts, colorrange=p.count_colorrange, colorscale=identity,
        marker=only(native.plots).marker, markersize=native.markersize,
        markerspace=:data, transformation=:inherit_model)
    return p
end
# BAT-owned updates demand the check before handing control back to a renderer.
# A normal native/public Makie consumer still resolves it on color demand.
_validate_plot_display(::Makie.AbstractPlot) = nothing
function _validate_plot_display(p::BATCheckedHexbin)
    p.checked_counts[]
    return nothing
end
Makie.extract_colormap(p::BATCheckedHexbin) = Makie.extract_colormap(p.plots[2])
Makie.data_limits(p::BATCheckedHexbin) = Makie.data_limits(p.plots[1])
Makie.boundingbox(p::BATCheckedHexbin, space::Symbol=:data) = Makie.boundingbox(p.plots[1], space)

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Hexbin2D,
    config::NamedTuple;
    transposed::Bool = false
)
    (; x, y, weights, thresh) = primitives
    (; colormap, rev, nbins) = config
    if isempty(weights)
        return PlotSpec[]
    end
    final_cmap = rev ? Reverse(colormap) : colormap

    # See Scatter2D's matching comment (makie_scatter.jl); per-axis bin counts
    # swap along with the axes.
    if transposed
        x, y = y, x
        nbins isa Tuple && (nbins = reverse(nbins))
    end

    hex = Makie.PlotSpec(BATCheckedHexbin, x, y;
        weights=weights,
        bins=nbins,
        colormap=final_cmap,
        threshold=thresh,
        colorrange=Makie.automatic,
        colorscale=identity
    )

    return [hex]
end

