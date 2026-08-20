# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Dead/empty primitive results are built FRESH on every call -- plain
# functions, deliberately NOT shared `const` sentinels (the previous design,
# reversed on purpose): ComputePipeline's is_same treats a callback returning
# the *identical mutable object* again as CHANGED (it cannot rule out
# in-place mutation), but a fresh, isequal-equal value as UNCHANGED -- which
# stops dirty-propagation right at the recipe's own output node, so
# downstream callbacks don't even run. Confirmed via a direct
# ComputePipeline-level experiment: a node returning a shared const re-ran
# its downstream on every update, the fresh-tuple variant zero times. Shared
# consts therefore made every dead/non-selected recipe re-dirty its
# downstream on every sample batch, defeating exactly the change-tracking a
# future per-cell grid rebuild would rely on. The cost of rebuilding these
# tiny empty NamedTuples per call is a few negligible allocations. The same
# convention applies to every _empty_*_primitives() in makie_stats.jl /
# makie_kde.jl / makie_hist.jl / makie_trace.jl.
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
        # A live cell can still have zero samples (right after vsel activates,
        # before the first batch flushes) -- and its placeholder view is 0x0
        # (not 2x0), so marg_coords[1, :] below would throw a BoundsError.
        # Degrade to the empty sentinel like the KDE recipes already do.
        isempty(weights) && return _empty_scatter2d_primitives()
        x = marg_coords[1, :]
        y = marg_coords[2, :]

        # Materialized to Vector{Float64} rather than passed through as the raw
        # SubArray{<:Any} from :flat_weights -- ComputePipeline's TypedEdge fixes
        # this node's output type from its first resolution, and the dead-cell
        # fallback (_empty_scatter2d_primitives()) hardcodes weights=Float64[]; a
        # live SubArray{Int64,...} (or any non-Float64 concrete type) here would
        # make a later live->dead transition (e.g. via vsel reduction) fail to
        # convert.
        return (x=x, y=y, weights=Float64.(weights))
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::Scatter2D,
        config::NamedTuple;
        transposed::Bool=false
)
        (; x, y, weights) = primitives
        (; markersize) = config
        # Lower-triangle cells swap x/y at compose time so both mirrored cells
        # can share one computed primitive while still following the standard
        # pair-plot orientation (cell (row, col): x = col's variable) -- see
        # the invariant comment in _init_gridlayout.
        transposed && ((x, y) = (y, x))

        real_markersize = if isempty(weights) || (all(x -> x ≈ weights[1], weights)) || (mean(weights) <= 0)
                markersize
        else
                sqrt.(weights ./ mean(weights)) .* markersize
        end

        scatter = S.Scatter(x, y;
                markersize=real_markersize
        )

        return [scatter]
end


# Whether `samples` carries chain identity at all -- true for MCMC samples
# (MCMCSampleID/AHMCSampleID both have a `chainid` field), false for
# anything else (importance sampling, MGVI, etc. default to `info = fill
# (nothing, ...)`, and BAT.jl's other SampleID subtype, MGVISampleInfo, has
# no chain concept either). Used to decide whether ChainScatter2D should even
# be offered in the recipe dropdown -- checking `hasfield` directly (rather
# than an explicit Union of known chain-bearing SampleID subtypes) means this
# stays correct if a future sampler adds another chainid-bearing SampleID
# subtype, with no changes needed here.
_samples_have_chain_ids(samples) = hasfield(eltype(samples.info), :chainid)

_empty_chainscatter2d_primitives() = (x=Float64[], y=Float64[], weights=Float64[], chainids=Int32[])

# Qualitative, wrap-around palette for distinguishing chains -- deliberately a
# small fixed set rather than a continuous colormap (per explicit request):
# BAT.jl commonly runs anywhere from a handful to 100+ concurrent chains, and
# a continuous gradient would leave *adjacent* chains just as visually
# indistinguishable as a wrapped categorical palette does once there are more
# chains than colors, while being harder to read at a glance for the common
# small-chain-count case.
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
        # See Scatter2D's matching comment above -- the zero-sample live
        # cell's placeholder view is 0x0, so row indexing would throw.
        isempty(weights) && return _empty_chainscatter2d_primitives()
        x = marg_coords[1, :]
        y = marg_coords[2, :]

        # weights materialized to Vector{Float64} for the same reason as
        # Scatter2D's live branch above -- see that comment.
        return (x=x, y=y, weights=Float64.(weights), chainids=chainids)
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::ChainScatter2D,
        config::NamedTuple;
        transposed::Bool=false
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

        # Colors are keyed by each chain's *rank* among the sorted unique ids
        # actually present (not the raw chainid value mod the palette length)
        # -- BAT's real chain ids are not small sequential 1:nchains integers
        # (confirmed empirically: a 4-chain run produced ids like
        # [7, 42, 56, 57]), so indexing the palette directly by chainid
        # collided most chains onto the same color (7, 42, and 56 are all
        # ≡0 mod 7, the palette length). Ranking first means the palette is
        # always used efficiently regardless of the actual id values. Stable
        # across cells within one render (every cell derives chainids from
        # the same underlying :flat_chainids/:flat_samples range, so the same
        # sorted-unique-id list -> same rank -> same color everywhere in the
        # grid), and stable over a live-sampling session too in the common
        # case (the chain *set* is fixed once real sampling starts; BAT only
        # ever replaces/renumbers chains during the earlier tuning phase).
        specs = PlotSpec[]
        for (rank, id) in enumerate(sort(unique(chainids)))
                mask = chainids .== id
                ms = real_markersize isa AbstractVector ? real_markersize[mask] : real_markersize
                color = _CHAIN_COLOR_PALETTE[mod1(rank, length(_CHAIN_COLOR_PALETTE))]
                push!(specs, S.Scatter(x[mask], y[mask]; markersize=ms, color=color))
        end

        return specs
end

