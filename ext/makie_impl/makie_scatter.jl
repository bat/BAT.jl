# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Empty results built FRESH per call, NOT shared consts: ComputePipeline's
# is_same treats a returned identical mutable object as CHANGED but a fresh
# isequal value as UNCHANGED, so fresh tuples stop dirty-propagation at the
# recipe's output node. Same convention for every _empty_*_primitives().
_empty_scatter2d_primitives() = (x=Float64[], y=Float64[], weights=Float64[])

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

    # Materialized to Vector{Float64}: ComputePipeline's TypedEdge locks this
    # node's output type at first resolution, and the dead-cell fallback
    # hardcodes Float64[] -- a live non-Float64 type would break a later
    # live->dead transition.
    return (x=x, y=y, weights=Float64.(weights))
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
    # Lower-triangle cells swap x/y at compose time so mirrored cells share
    # one computed primitive -- see the invariant comment in _init_gridlayout.
    transposed && ((x, y) = (y, x))

    real_markersize = if isempty(weights) || (all(w -> w ≈ weights[1], weights)) || (mean(weights) <= 0)
        markersize
    else
        sqrt.(weights ./ mean(weights)) .* markersize
    end

    scatter = S.Scatter(x, y;
        markersize=real_markersize
    )

    return [scatter]
end


# Whether `samples` carries chain identity (true for MCMC SampleID types);
# gates whether ChainScatter2D is offered in the recipe dropdown.
_samples_have_chain_ids(samples) = hasfield(eltype(samples.info), :chainid)

_empty_chainscatter2d_primitives() = (x=Float64[], y=Float64[], weights=Float64[], chainids=Int32[])

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
    # Zero-sample live cell -- see Scatter2D's matching comment above.
    isempty(weights) && return _empty_chainscatter2d_primitives()
    x = marg_coords[1, :]
    y = marg_coords[2, :]

    # weights materialized to Vector{Float64} -- see Scatter2D's live branch above.
    return (x=x, y=y, weights=Float64.(weights), chainids=chainids)
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
    # See Scatter2D's matching comment above.
    transposed && ((x, y) = (y, x))

    real_markersize = if isempty(weights) || (all(w -> w ≈ weights[1], weights)) || (mean(weights) <= 0)
        markersize
    else
        sqrt.(weights ./ mean(weights)) .* markersize
    end

    # Colors keyed by each chain's *rank* among the sorted unique ids present,
    # not the raw chainid: real ids are large/sparse and collide mod the
    # palette length; same rank -> same color across all cells of one render.
    groups = Dict{eltype(chainids),Vector{Int}}()
    for (idx, id) in pairs(chainids)
        push!(get!(() -> Int[], groups, id), idx)
    end

    specs = PlotSpec[]
    for (rank, id) in enumerate(sort!(collect(keys(groups))))
        idxs = groups[id]
        ms = real_markersize isa AbstractVector ? real_markersize[idxs] : real_markersize
        color = _CHAIN_COLOR_PALETTE[mod1(rank, length(_CHAIN_COLOR_PALETTE))]
        push!(specs, S.Scatter(x[idxs], y[idxs]; markersize=ms, color=color))
    end

    return specs
end

