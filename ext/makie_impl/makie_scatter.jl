# This file is a part of BAT.jl, licensed under the MIT License (MIT).

const _EMPTY_SCATTER2D_PRIMITIVES = (x=Float64[], y=Float64[], weights=Float64[])

function compute_plotting_primitives(
        ::SubArray,
        ::SubArray,
        ::Scatter2D,
        ::RS,
        ::CS,
        ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
        return _EMPTY_SCATTER2D_PRIMITIVES
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Scatter2D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        x = marg_coords[1, :]
        y = marg_coords[2, :]

        # Materialized to Vector{Float64} rather than passed through as the raw
        # SubArray{<:Any} from :flat_weights -- ComputePipeline's TypedEdge fixes
        # this node's output type from its first resolution, and the dead-cell
        # fallback (_EMPTY_SCATTER2D_PRIMITIVES) hardcodes weights=Float64[]; a
        # live SubArray{Int64,...} (or any non-Float64 concrete type) here would
        # make a later live->dead transition (e.g. via vsel reduction) fail to
        # convert.
        return (x=x, y=y, weights=Float64.(weights))
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::Scatter2D,
        config::NamedTuple
)
        (; x, y, weights) = primitives
        (; markersize) = config

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

const _EMPTY_CHAINSCATTER2D_PRIMITIVES = (x=Float64[], y=Float64[], weights=Float64[], chainids=Int32[])

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
        return _EMPTY_CHAINSCATTER2D_PRIMITIVES
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
        x = marg_coords[1, :]
        y = marg_coords[2, :]

        # weights materialized to Vector{Float64} for the same reason as
        # Scatter2D's live branch above -- see that comment.
        return (x=x, y=y, weights=Float64.(weights), chainids=chainids)
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::ChainScatter2D,
        config::NamedTuple
)
        (; x, y, weights, chainids) = primitives
        (; markersize) = config

        if isempty(x)
                return PlotSpec[]
        end

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

