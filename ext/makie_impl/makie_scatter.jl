# This file is a part of BAT.jl, licensed under the MIT License (MIT).


_empty_scatter2d_primitives() = (x = Float64[], y = Float64[], weights = Float64[])

# Marker areas depend on relative mass. Normalize before narrowing so finite
# BigFloat/logarithmic weights cannot become uniform zeros or infinities.
function _scatter_relative_weights(weights)
    relative = BAT._plot_relative_weights(weights)
    return [BAT._plot_float64(w, "relative scatter weight") for w in relative]
end


# Normalize relative marker areas without overflowing the weight sum.
# Uniform and non-positive-total cases retain their existing marker convention.
function _scatter_markersizes(weights::AbstractVector, markersize)
    if isempty(weights) || all(w -> w ≈ weights[1], weights)
        return markersize
    end
    scale = maximum(abs, weights)
    relative = weights ./ scale
    relative_mean = mean(relative)
    relative_mean <= 0 && return markersize
    return sqrt.(relative ./ relative_mean) .* markersize
end


function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::Scatter2D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _empty_scatter2d_primitives()
end


function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Scatter2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    # Zero-sample live cell: the placeholder view is 0x0, so row indexing would throw.
    isempty(weights) && return _empty_scatter2d_primitives()
    x = marg_coords[1, :]
    y = marg_coords[2, :]

    # Keep the graph's weight representation stable across empty/live states.
    return (x = x, y = y, weights = _scatter_relative_weights(weights))
end


function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Scatter2D,
    config::NamedTuple;
    transposed::Bool = false
)
    (; x, y, weights) = primitives
    (; markersize) = config
    isempty(x) && return PlotSpec[]
    transposed && ((x, y) = (y, x))

    real_markersize = _scatter_markersizes(weights, markersize)

    scatter = S.Scatter(x, y;
        markersize = real_markersize
    )

    return [scatter]
end


# Whether `samples` carries chain identity (true for MCMC SampleID types);
# gates whether ChainScatter2D is offered in the recipe dropdown.
_samples_have_chain_ids(samples) =
    isconcretetype(eltype(samples.info)) && hasfield(eltype(samples.info), :chainid)

_empty_chainscatter2d_primitives() = (
    x = Float64[], y = Float64[], weights = Float64[], chainids = Int32[]
)

# Qualitative wrap-around palette for distinguishing chains -- deliberately a
# small fixed set rather than a continuous colormap.
const _CHAIN_COLOR_PALETTE = Makie.wong_colors()

function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::AbstractVector,
    ::ChainScatter2D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _empty_chainscatter2d_primitives()
end


function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    chainids::AbstractVector,
    recipe::ChainScatter2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    isempty(weights) && return _empty_chainscatter2d_primitives()
    length(chainids) == length(weights) ||
        throw(ArgumentError("ChainScatter2D requires one chain identity per sample"))
    x = marg_coords[1, :]
    y = marg_coords[2, :]

    return (x = x, y = y, weights = _scatter_relative_weights(weights), chainids = chainids)
end


function compose_plotspecs(
    primitives::NamedTuple,
    recipe::ChainScatter2D,
    config::NamedTuple;
    transposed::Bool = false
)
    (; x, y, weights, chainids) = primitives
    (; markersize) = config

    if isempty(x)
        return PlotSpec[]
    end
    transposed && ((x, y) = (y, x))

    real_markersize = _scatter_markersizes(weights, markersize)

    # Use sorted chain ranks for palette indices; raw IDs can be large or sparse.
    groups = Dict{eltype(chainids),Vector{Int}}()
    for (idx, id) in pairs(chainids)
        push!(get!(() -> Int[], groups, id), idx)
    end

    # The graph supplies the full dataset's palette so windowing cannot recolor
    # a surviving chain or disagree with its timing-based trace overlay.
    ids = sort!(collect(keys(groups)))
    palette_ids = hasproperty(primitives, :all_chainids) ? primitives.all_chainids : ids
    ranks = Dict(id => rank for (rank, id) in enumerate(palette_ids))
    specs = PlotSpec[]
    for id in ids
        idxs = groups[id]
        ms = real_markersize isa AbstractVector ? real_markersize[idxs] : real_markersize
        color = _CHAIN_COLOR_PALETTE[mod1(ranks[id], length(_CHAIN_COLOR_PALETTE))]
        push!(specs, S.Scatter(x[idxs], y[idxs]; markersize = ms, color = color))
    end

    return specs
end
