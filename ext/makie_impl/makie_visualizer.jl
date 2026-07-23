# This file is a part of BAT.jl, licensed under the MIT License (MIT).

const BAT_MAKIE_RECIPES_1D = [
    Hist1D(),
    QuantileHist1D(),
    KDE1D(),
    QuantileKDE1D(),
    Std1D(),
    Mean1D(),
    Errorbars1D(),
    PDF1D()
]

const BAT_MAKIE_RECIPES_2D = [
    Hist2D(),
    QuantileHist2D(),
    Hexbin2D(),
    KDE2D(),
    QuantileKDE2D(),
    Scatter2D(),
    Cov2D(),
    Std2D(),
    Mean2D(),
    Errorbars2D()
]

function recipe_symbol(recipe::R) where {R<:BATMakieRecipe}
    return Symbol(string(typeof(recipe)))
end

function primitive_symbol(recipe, vsel::Tuple{Int64,Int64})
    return Symbol(string(recipe) * "_prim" * "_" * "$(vsel[1])" * "$(vsel[2])")
end

function primitive_symbol(recipe::R, vsel::Tuple{Int64,Int64}) where {R<:BATMakieRecipe}
    return Symbol(string(typeof(recipe)) * "_prim" * "_" * "$(vsel[1])" * "$(vsel[2])")
end

function marg_symbol(vsel::Tuple{Int64,Int64})
    return Symbol("marg_$(vsel[1])$(vsel[2])")
end

# Dispatches each incremental recipe to the right kind of persistent per-cell
# state (Mean1D/Std1D/Mean2D/Cov2D/Std2D need running sufficient statistics;
# Hist1D/Hist2D/QuantileHist1D/QuantileHist2D need a running fixed-edge
# Histogram) -- see makie_stats.jl / makie_hist.jl for the state types
# themselves and is_incremental's definition.
_make_running_state_1d(::Union{Mean1D,Std1D}) = _IncrementalUvState()
_make_running_state_1d(::Union{Hist1D,QuantileHist1D}) = _IncrementalHist1DState()
_make_running_state_2d(::Union{Mean2D,Cov2D,Std2D}) = _IncrementalMvState()
_make_running_state_2d(::Union{Hist2D,QuantileHist2D}) = _IncrementalHist2DState()

# Estimates a fixed, reasonable initial axis-limit/histogram-bin-edge domain
# per real model dimension, from the PRIOR alone (before any real samples
# exist), by drawing a modest number of prior samples and taking robust tail
# quantiles. Exact when there's no separate likelihood (prior == posterior);
# a practical heuristic otherwise (a highly informative likelihood leaves the
# posterior occupying only a fraction of this range -- coarser resolution,
# not wrong; a prior/likelihood conflict could in principle exceed it). NOT a
# hard bound either way -- see the overflow-widening in init_visualizer!/
# flush_buffer! below, which callers must rely on rather than treating this
# as guaranteed.
function _estimate_prior_domain(mcmc_states::Vector{<:MCMCState}, n_dof::Integer; n_prior_samples::Integer=2000, tail_prob::Real=0.0015)
    target = mcmc_target(mcmc_states[1])
    initsrc = BAT.get_initsrc_from_target(target)
    shape = varshape(initsrc)
    draws = [ValueShapes.unshaped(rand(initsrc), shape) for _ in 1:n_prior_samples]
    M = reduce(hcat, draws)
    lo = [quantile(view(M, d, :), tail_prob) for d in 1:n_dof]
    hi = [quantile(view(M, d, :), 1 - tail_prob) for d in 1:n_dof]
    return lo, hi
end

# Same purpose as _estimate_prior_domain, but for the static bat_makie_plot
# path: all samples already exist there, so the true min/max is available
# directly and is strictly more accurate than a prior-based estimate --
# no need to guess. Not "the full domain plus margin" -- see the small
# proportional margin added when this feeds into axis_limits_i.
function _domain_from_samples(data::AbstractMatrix, n_dof::Integer)
    lo = [minimum(view(data, d, :)) for d in 1:n_dof]
    hi = [maximum(view(data, d, :)) for d in 1:n_dof]
    return lo, hi
end

# Widens (monotonically -- never shrinks) the graph's fixed domain if new
# real data exceeds it in any dimension, and propagates the update if so.
# Called with only the newly-arrived batch (not the full dataset) so this
# stays O(batch size), not O(total samples so far).
function _widen_domain!(graph::ComputeGraph, new_samples)
    isempty(new_samples) && return nothing
    domain_lo = graph[:domain_lo][]
    domain_hi = graph[:domain_hi][]
    isempty(domain_lo) && return nothing

    data = new_samples.v.data
    new_lo = copy(domain_lo)
    new_hi = copy(domain_hi)
    widened = false
    for d in eachindex(domain_lo)
        row = view(data, d, :)
        batch_lo, batch_hi = extrema(row)
        if batch_lo < new_lo[d]
            new_lo[d] = batch_lo
            widened = true
        end
        if batch_hi > new_hi[d]
            new_hi[d] = batch_hi
            widened = true
        end
    end
    widened && update!(graph, domain_lo=new_lo, domain_hi=new_hi)
    return nothing
end


function BATVisualizer(vis::BATMakieVisualization)
    (; recipes, N_max, triagonal_config, diagonal_config) = vis

    # TODO: Think about whether or not the more expressive vsel symbol approach is desired, or integers are sufficient
    # idxs = vsel isa AbstractVector{<:Int64} ? vsel : reduce(vcat, asindex.(Ref(vs), vsel))

    graph = _init_compute_graph(
        recipes,
        triagonal_config,
        diagonal_config,
        N_max,
    )

    buffer_lock = ReentrantLock()
    content = (
        graph=graph,
        buffer_lock=buffer_lock,
        # Tied to buffer_lock so sampling threads blocked on the high-watermark
        # (in update_visualizer_impl!) and the listener's flush-and-notify both
        # synchronize through the same lock, per Threads.Condition's contract.
        buffer_cond=Threads.Condition(buffer_lock),
        chain_ids=Vector{Integer}(),
        output_buffer=Vector{Vector{DensitySampleVector}}(),
        n_buffer_samples=Ref(0),
        # Current flush-trigger threshold. Starts at vis.n_batch; when
        # adaptive_batching is on, flush_buffer! grows it geometrically after
        # every flush. update_visualizer_impl!'s backpressure ceiling is always
        # derived from this (scaled by the configured max_buffered/n_batch
        # ratio) rather than a separately-growing value, so the two can never
        # drift out of the relationship where blocking would deadlock the
        # sampler against a flush trigger it's not allowed to reach.
        effective_batch_size=Ref(vis.n_batch),
        buffer_ratio=vis.max_buffered / vis.n_batch,
        is_live=Threads.Atomic{Bool}(true),
        listener_task=Ref{Union{Task,Nothing}}(nothing),
        # Set once in init_visualizer! (the model's dimensionality isn't known yet
        # here); read by _apply_vsel! to (re-)validate any requested vsel change.
        n_dof=Ref{Int}(0)
    )

    return BATVisualizer(vis, content)
end


function _init_compute_graph(
    recipes::NamedTuple,
    triagonal_config::NamedTuple,
    diagonal_config::NamedTuple,
    n::Integer,
)
    graph = ComputeGraph()

    smpls = Any[] #Vector{Vector{DensitySampleVector}}()

    add_input!(graph, :samples, smpls)

    curr_idxs = Vector{Vector{Integer}}()
    add_input!(graph, :current_idxs, curr_idxs)

    register_computation!(graph,
        [:samples, :current_idxs],
        [:flat_samples],
    ) do inputs, changed, cached
        samples = inputs.samples
        current_idxs = inputs.current_idxs

        walker_views = Any[] #Vector{DensitySampleVector}()
        for i in eachindex(samples)
            for j in eachindex(samples[i])
                push!(walker_views, view(samples[i][j], 1:current_idxs[i][j]))
            end
        end
        return (vcat(walker_views...),)
    end

    map!(smpls -> length(smpls),
        graph,
        :flat_samples,
        :current_idx
    )
    map!(
        # Empty view of the real weight type/eltype instead of a hardcoded Int64
        # placeholder, matching the empty marg_coords convention used elsewhere.
        (smpls, idx) -> view(smpls.weight, 1:idx),
        graph,
        [:flat_samples, :current_idx],
        :flat_weights
    )
    map!(
        # Per-sample chain id, for ChainScatter2D -- only meaningful when the
        # underlying samples actually carry chain identity (MCMCSampleID/
        # AHMCSampleID's `.chainid`; not importance sampling, MGVI, etc.,
        # whose `.info` has no such field). Degrades to an empty Int32[]
        # otherwise -- ChainScatter2D is never offered in the recipe menu in
        # that case (see _samples_have_chain_ids in makie_scatter.jl), but
        # this still needs to not error if the graph is ever resolved before
        # that's checked (e.g. during precompilation with a plain smoke-test
        # dataset).
        (smpls, idx) -> hasfield(eltype(smpls.info), :chainid) ? Int32[s.chainid for s in view(smpls.info, 1:idx)] : Int32[],
        graph,
        [:flat_samples, :current_idx],
        :flat_chainids
    )

    add_input!(graph, :idxs, Integer[])

    register_computation!(graph,
        [:idxs],
        [:vsel_map],
    ) do inputs, changed, cached
        idxs = inputs.idxs
        # idxs is meant to be user-changeable at runtime (via _apply_vsel!, which
        # enforces this), so this is a full stateless recompute on every change --
        # not a cache mutated in place -- rather than relying on any invariant
        # about how idxs evolves. Cheap regardless: n<=N_max is small and idxs
        # changes are rare (user-driven), not a per-sample-batch occurrence.
        @assert length(idxs) <= n "idxs has $(length(idxs)) entries, exceeding the grid size N_max=$n"

        n_active = length(idxs)
        vsel_map = Matrix{Tuple{Int,Int}}(undef, n, n)
        for i in 1:n, j in 1:n
            # (0, 0) is an unreachable sentinel for cells beyond the current
            # selection -- live_map already keeps these cells from being read,
            # but this way a hypothetical future bypass fails loudly (index 0)
            # instead of silently reusing a stale selection.
            vsel_map[i, j] = (i <= n_active && j <= n_active) ? (idxs[i], idxs[j]) : (0, 0)
        end

        return (vsel_map,)
    end

    register_computation!(graph,
        [:idxs],
        [:live_map],
    ) do inputs, changed, cached
        idxs = inputs.idxs
        @assert length(idxs) <= n "idxs has $(length(idxs)) entries, exceeding the grid size N_max=$n"

        live_map = fill(false, n, n)

        n_active = length(idxs)
        for i in 1:n_active
            for j in 1:n_active
                live_map[i, j] = true
            end
        end
        return (live_map,)
    end

    # Control nodes for what primitives are computed and what are set to emtpy
    add_input!(graph, :upper_recipe, recipes.upper)
    add_input!(graph, :diagonal_recipe, recipes.diagonal)
    add_input!(graph, :lower_recipe, recipes.lower)

    add_input!(graph, :show_stats_upper, false)
    add_input!(graph, :show_stats_diag, false)
    add_input!(graph, :show_stats_lower, false)

    add_input!(graph, :triagonal_config, triagonal_config)
    add_input!(graph, :diagonal_config, diagonal_config)

    # Fixed per-real-dimension domain (see _estimate_prior_domain/
    # _domain_from_samples), used for both axis_limits_i below and the
    # incremental histogram recipes' bin edges. Empty until the caller
    # (init_visualizer! for the live path, bat_makie_plot/convert_arguments
    # for the static path) sets it from real information -- axis_limits_i
    # guards against reading it before then.
    add_input!(graph, :domain_lo, Float64[])
    add_input!(graph, :domain_hi, Float64[])

    for recipe in vcat(BAT_MAKIE_RECIPES_1D, BAT_MAKIE_RECIPES_2D)
        add_input!(graph, Symbol("$(typeof(recipe))"), recipe)
    end
    # ChainScatter2D is deliberately not in BAT_MAKIE_RECIPES_2D (see its own
    # registration below, in the (i,j) loop) since it needs an extra input
    # (:flat_chainids) the shared per-recipe registration loop above doesn't
    # thread through -- but it still needs this same per-type input node
    # (matching every other recipe) for consistency.
    add_input!(graph, Symbol("$(ChainScatter2D)"), ChainScatter2D())

    for i in 1:n
        #1D marginal views
        marg_sym = marg_symbol((i, i))

        map!(
            # Dead-branch view is over smpls.v.data itself (empty index sets,
            # so it's always safe regardless of current_idx/n_dof), not a
            # hardcoded ElasticMatrix placeholder: bat_makie_plot's static path
            # backs samples with a plain Matrix, and once vsel can change at
            # runtime there (via the picker), a live view (Matrix-backed) and a
            # dead view (previously hardcoded ElasticMatrix-backed) of the same
            # node would be incompatible types, erroring on the transition.
            (smpls, vsel_map, current_idx, live_map) -> (current_idx > 0 && live_map[i, i]) ? view(smpls.v.data, [vsel_map[i, i][1]], 1:current_idx) : view(smpls.v.data, Int[], 1:0),
            graph,
            [:flat_samples, :vsel_map, :current_idx, :live_map],
            marg_sym
        )

        # Fixed from the domain estimate/actual-data-range (see
        # _estimate_prior_domain/_domain_from_samples), not derived from the
        # currently-visible marg/current_idx -- this is what keeps limits
        # stable during live sampling and while panning the static plot's
        # sample-index slider, instead of jumping around as the visible min/
        # max changes. Falls back to (0,1) if the domain isn't set yet, or if
        # this grid position has no active variable (vsel_map's (0,0)
        # sentinel) -- both guard against indexing into an empty/mismatched
        # domain vector.
        map!(
            (lo, hi, vsel_map) -> begin
                v = vsel_map[i, i][1]
                (isempty(lo) || v == 0) && return (0.0, 1.0)
                margin = 0.05 * (hi[v] - lo[v])
                (lo[v] - margin, hi[v] + margin)
            end,
            graph,
            [:domain_lo, :domain_hi, :vsel_map],
            Symbol("axis_limits_$i")
        )

        primitive_symbols_1D = [primitive_symbol(recipe, (i, i)) for recipe in BAT_MAKIE_RECIPES_1D]

        for k in eachindex(primitive_symbols_1D)
            recipe = BAT_MAKIE_RECIPES_1D[k]
            # Persistent per-cell accumulator for incremental recipes --
            # captured by the closure below, one independent instance per
            # (cell, recipe), so it survives across ticks instead of being
            # rebuilt from scratch on every recompute. Stats recipes
            # (Mean1D/Std1D) and histogram recipes (Hist1D/QuantileHist1D)
            # need different underlying state -- see _make_running_state_1d.
            running_state = is_incremental(recipe) ? _make_running_state_1d(recipe) : nothing

            register_computation!(graph,
                [marg_sym, :flat_weights, :diagonal_recipe, :live_map, :diagonal_config, :vsel_map, :domain_lo, :domain_hi],
                [primitive_symbols_1D[k]]
            ) do inputs, changed, cached
                coords, weights, live_recipe, live_map, config, vsel_map, domain_lo, domain_hi = inputs
                cell_status = live_map[i, i] ? LiveCell() : DeadCell()
                recipe_status = determine_recipe_status(recipe, live_recipe())
                # filter=true isn't compatible with incremental accumulation
                # (see is_incremental's docs in makie_hist.jl) -- falls back to
                # a full recompute for that case even if the recipe otherwise
                # supports it.
                primitives = if cell_status isa LiveCell && is_incremental(recipe) && !config.filter
                    vsel = vsel_map[i, i][1]
                    if running_state isa _IncrementalHist1DState
                        # A live cell can still have zero samples (right after
                        # vsel activates, before the first batch flushes) -- the
                        # dead-shaped placeholder view isn't just empty but
                        # actually 0-row, so skip straight to the state's
                        # current (possibly still-empty) result rather than
                        # feeding it into _update_hist!.
                        if isempty(weights)
                            compute_hist_primitives(recipe, running_state, config)
                        else
                            (; nbins, closed) = config
                            eff_nbins = recipe isa Hist1D ? nbins + 1 : nbins
                            domain = (domain_lo[vsel], domain_hi[vsel])
                            _update_hist!(running_state, vec(coords), weights, vsel, domain, eff_nbins, closed)
                            compute_hist_primitives(recipe, running_state, config)
                        end
                    else
                        _update_stats!(running_state, vec(coords), weights, vsel)
                        compute_stats_primitives(recipe, running_state, config)
                    end
                else
                    compute_plotting_primitives(coords, weights, recipe, recipe_status, cell_status, config)
                end

                return (primitives,)
            end
        end

        for j in i+1:n
            marg_sym_2D = marg_symbol((j, i))
            map!(
                # See the 1D case above for why this is an empty view over the
                # real smpls.v.data rather than a hardcoded ElasticMatrix.
                (smpls, vsel_map, current_idx, live_map) -> (current_idx > 0 && live_map[j, i]) ? view(smpls.v.data, [vsel_map[j, i]...], 1:current_idx) : view(smpls.v.data, Int[], 1:0),
                graph,
                [:flat_samples, :vsel_map, :current_idx, :live_map],
                marg_sym_2D
            )

            primitive_symbols_2D = [primitive_symbol(recipe, (j, i)) for recipe in BAT_MAKIE_RECIPES_2D]

            for k in eachindex(primitive_symbols_2D)
                recipe = BAT_MAKIE_RECIPES_2D[k]
                # Per-(cell, recipe) persistent accumulator for the incremental
                # 2D recipes -- see _make_running_state_2d and the 1D case above.
                running_state = is_incremental(recipe) ? _make_running_state_2d(recipe) : nothing

                register_computation!(graph,
                    [marg_sym_2D, :flat_weights, :upper_recipe, :lower_recipe, :live_map, :triagonal_config, :vsel_map, :domain_lo, :domain_hi],
                    [primitive_symbols_2D[k]]
                ) do inputs, changed, cached
                    coords, weights, live_recipe_upper, live_recipe_lower, live_map, config, vsel_map, domain_lo, domain_hi = inputs
                    cell_status = live_map[i, j] ? LiveCell() : DeadCell()
                    recipe_status = determine_recipe_status(recipe, live_recipe_upper(), live_recipe_lower())
                    primitives = if cell_status isa LiveCell && is_incremental(recipe) && !config.filter
                        vsel = vsel_map[j, i]
                        if running_state isa _IncrementalHist2DState
                            # See the 1D case above: a live cell can still have
                            # zero samples, and the dead-shaped placeholder view
                            # is 0-row (not just 0-column), so indexing row 1/2
                            # of it directly would throw -- skip straight to the
                            # state's current result instead.
                            if isempty(weights)
                                compute_hist_primitives(recipe, running_state, config)
                            else
                                (; nbins, closed) = config
                                domain = ((domain_lo[vsel[1]], domain_hi[vsel[1]]), (domain_lo[vsel[2]], domain_hi[vsel[2]]))
                                x = view(coords, 1, :)
                                y = view(coords, 2, :)
                                _update_hist!(running_state, x, y, weights, vsel, domain, nbins, closed)
                                compute_hist_primitives(recipe, running_state, config)
                            end
                        else
                            _update_stats!(running_state, coords, weights, vsel)
                            compute_stats_primitives(recipe, running_state, config)
                        end
                    else
                        compute_plotting_primitives(coords, weights, recipe, recipe_status, cell_status, config)
                    end
                    return (primitives,)
                end
            end

            # ChainScatter2D registered separately from the shared loop above
            # (rather than added to BAT_MAKIE_RECIPES_2D) since it's the only
            # 2D recipe that needs an extra per-sample input (:flat_chainids)
            # -- folding it into the shared loop would mean adding a chainids
            # parameter to all 9 *other* 2D recipes' compute_plotting_primitives
            # methods too, even though they'd ignore it. Uses the same
            # primitive_symbol(recipe, (j,i)) naming as the shared loop, so
            # _init_gridlayout's graph[primitive_symbol(upper_recipe, (j,i))][]
            # lookup finds it transparently whenever a user actually selects
            # ChainScatter2D as the upper/lower recipe. Not an is_incremental
            # recipe (like plain Scatter2D, it just needs the raw point cloud
            # each time), so no running-state accumulator branch is needed.
            chainscatter_primitive_sym = primitive_symbol(ChainScatter2D(), (j, i))
            register_computation!(graph,
                [marg_sym_2D, :flat_weights, :flat_chainids, :upper_recipe, :lower_recipe, :live_map, :triagonal_config],
                [chainscatter_primitive_sym]
            ) do inputs, changed, cached
                coords, weights, chainids, live_recipe_upper, live_recipe_lower, live_map, config = inputs
                cell_status = live_map[i, j] ? LiveCell() : DeadCell()
                recipe_status = determine_recipe_status(ChainScatter2D(), live_recipe_upper(), live_recipe_lower())
                primitives = compute_plotting_primitives(coords, weights, chainids, ChainScatter2D(), recipe_status, cell_status, config)
                return (primitives,)
            end
        end
    end

    return graph
end

function _init_gridlayout(
    graph::ComputeGraph,
    n::Int64
)
    gridlayout = lift(
        graph[:current_idx],
        graph[:idxs], # re-render on vsel changes too, not just new sample batches --
        # otherwise toggling the picker only appears to work during live
        # sampling, as an accidental side effect of :current_idx also changing.
        graph[:upper_recipe],
        graph[:diagonal_recipe],
        graph[:lower_recipe],
        graph[:show_stats_upper],
        graph[:show_stats_diag],
        graph[:show_stats_lower]
    ) do idx, _idxs, upper_recipe, diagonal_recipe, lower_recipe, stats_upper, stats_diag, stats_lower
        matrix = Matrix{Any}(undef, n, n)
        triagonal_config = graph[:triagonal_config][]
        diagonal_config = graph[:diagonal_config][]

        # Deselecting a variable should visually remove its row/column and
        # let the remaining ones grow into the freed space. The naive way to
        # do that -- building an n_active x n_active S.GridLayout matrix
        # instead of always n x n -- hits a genuine Makie SpecApi
        # reconciliation bug: shrinking the matrix then growing it back to a
        # size it held *before* tries to reuse a previously-disconnected
        # block at that position instead of creating a fresh one, and it
        # never reappears (confirmed empirically via direct instrumentation
        # of both the compute graph and this very closure -- both correctly
        # recomputed n_active back to its original value, yet the rendered
        # scene stayed frozen at the smaller size).
        #
        # So the matrix itself always stays n x n (sidestepping that
        # reconciliation path entirely, since the *set of positions* never
        # changes) and inactive rows/columns are instead collapsed to
        # Fixed(0) via GridLayoutSpec's own rowsizes/colsizes -- the same
        # mechanism (and the same Fixed(0)-collapses-a-cell trick used
        # elsewhere in this file for the collapsible controls row) just
        # applied here to grid rows/columns instead of Figure rows.
        # GridLayoutBase then naturally redistributes the same total area
        # across only the n_active non-collapsed cells.
        n_active = length(_idxs)
        @assert n_active <= n "idxs has $(n_active) entries, exceeding the grid size N_max=$n"
        # Explicitly typed as Union{Auto,Fixed} (not left to infer as
        # Vector{Any}) -- GridLayoutBase.convert_contentsizes requires
        # Vector{<:ContentSize} and rejects a plain Vector{Any}, which is
        # what an untyped comprehension over two different concrete types
        # produces.
        cellsizes = Union{Auto,Fixed}[i <= n_active ? Auto() : Fixed(0) for i in 1:n]

        # Shared y-axis limit across all diagonal cells (0 to 1.1x the peak
        # value of any active diagonal's own recipe primitives -- see
        # _diag_y_extent's per-recipe methods), rather than each diagonal
        # auto-scaling to its own peak independently. nothing (Makie's usual
        # autolimits) until real data exists in at least one active diagonal
        # cell, matching the graceful-degradation-before-data pattern used
        # elsewhere in this function (e.g. axis_limits_i falling back to
        # (0,1)).
        diag_y_max = maximum(
            (_diag_y_extent(graph[primitive_symbol(diagonal_recipe, (i, i))][], diagonal_recipe()) for i in 1:n_active);
            init=0.0
        )
        diag_ylims = diag_y_max > 0 ? (0.0, 1.1 * diag_y_max) : nothing

        # GridLayoutBase only credits a grid's *structurally* first/last
        # row/column (via ismostin in gridlayout.jl) toward that side's
        # reported protrusion, regardless of whether that row/col is
        # actually active -- confirmed by direct layoutobservables.protrusions
        # inspection. Active columns are always the prefix 1:n_active, so
        # column 1 (left) is always active and its protrusion is always
        # correctly reported -- but row n (bottom) is only active when
        # n_active == n (fully selected); the instant a variable is
        # deselected, row n goes Fixed(0) and the grid's *reported* bottom
        # protrusion silently drops to 0, even though the new last-active row
        # still renders real tick/axis labels below it. Since fig.layout's
        # Outside(44,44,16,40) alignmode (in _build_fig) trades protrusion
        # against content size for a fixed total canvas, that erroneously
        # "freed" protrusion silently inflates the grid's Relative(0.8)/
        # Aspect(1,1)-computed size -- this is exactly the "small but real
        # size jump when the vsel selection shrinks/grows" previously left as
        # an open TODO in _build_fig (bottom protrusion flips between a real
        # value and exactly 0 depending only on whether n_active == n, with
        # the resulting size jump matching 0.8x that protrusion delta to five
        # decimal places).
        #
        # Fixed via GridLayoutBase's public alignmode=Mixed(bottom=Protrusion(...))
        # escape hatch first -- reverted. Confirmed empirically that this
        # GridLayoutBase version's update! unconditionally calls
        # determinedirsize(gl, Col()) on every relayout regardless of that
        # grid's own alignmode, and determinedirsize only handles Inside/
        # Outside, throwing "Unknown AlignMode of type Mixed" -- reproduced
        # as a deterministic crash on the very first live-sampling render
        # (not just an occasional edge case; two isolated static-path test
        # calls happened not to hit it, which is what made it look safe at
        # first).
        #
        # Fix instead keeps row n's diagonal cell's bottom decorations
        # logically "on" (so GridLayoutBase's own unmodified, already-correct
        # ismostin-based protrusion computation credits it) whenever it isn't
        # really active, but renders them fully transparent so nothing is
        # visibly drawn -- avoiding both the ghost-decoration bug this would
        # otherwise reintroduce and the Mixed/determinedirsize crash, while
        # staying entirely within the already-battle-tested Inside() code
        # path. Mirrors the real last-*active* row's own variable/limits
        # (rather than row n's own (0,1) placeholder) so the phantom
        # protrusion matches what's actually needed, not a guess.
        text_color = Colors.RGBA(Makie.to_color(Makie.current_default_theme()[:textcolor][]))
        transparent_text_color = Colors.RGBA(text_color.r, text_color.g, text_color.b, 0.0)

        for i in 1:n
            diagonal_primitives = graph[primitive_symbol(diagonal_recipe, (i, i))][]
            diagonal_plotspecs = compose_plotspecs(diagonal_primitives, diagonal_recipe(), diagonal_config)
            stats_specs_1D = stats_diag ? get_stats_plotspecs(graph, (i, i), Makie1DStats(), diagonal_config) : []
            append!(diagonal_plotspecs, stats_specs_1D)

            xlims = graph[Symbol("axis_limits_$i")][]
            # A Fixed(0) row/column (cellsizes above) collapses the cell's
            # own plotting area to zero, but ticks/tick-labels/gridlines are
            # protrusion content drawn *outside* that area -- they don't
            # automatically disappear just because the cell they're attached
            # to has shrunk to nothing (confirmed empirically: leftover tick
            # marks/labels from deselected variables were still visible).
            # Explicitly forcing every decoration off for an inactive cell,
            # not just relying on it having zero size, is what actually
            # removes them.
            cell_active = i <= n_active
            # See the long comment above the text_color/transparent_text_color
            # definitions for the full story: row n's bottom decorations are
            # kept logically "on" (for GridLayoutBase's own protrusion
            # bookkeeping) even when this cell isn't really active, but
            # rendered fully transparent -- and mirroring the real last-
            # *active* row's own variable/limits, not row n's own (0,1)
            # placeholder, so the phantom protrusion matches what's actually
            # needed.
            is_phantom_row = (i == n) && !cell_active && n_active > 0
            show_x_decor = cell_active || is_phantom_row
            effective_xlims = is_phantom_row ? graph[Symbol("axis_limits_$n_active")][] : xlims
            matrix[i, i] = S.Axis(
                plots=diagonal_plotspecs,
                # Matches the upper/lower 2D cells' aspect=1 below -- without
                # it, a diagonal cell has no fixed visual aspect ratio at all
                # (unlike a 2D cell, whose data-derived x/y limits happen to
                # somewhat constrain its shape) and stretches to fill whatever
                # rectangle the GridLayout/decorations leave it, typically
                # taller than wide for a 1D density/histogram.
                aspect=1,
                limits=(effective_xlims, diag_ylims),
                # Every cell shows its own bottom/left tick labels + axis
                # labels now (per explicit request), with the tick *marks*
                # themselves removed everywhere to keep the added clutter in
                # check -- xticksvisible/yticksvisible=false rather than
                # tying them to visibility of the labels.
                xticklabelsvisible=show_x_decor, xticksvisible=false,
                yticklabelsvisible=cell_active, yticksvisible=false,
                yticklabelrotation=pi / 2,
                ytickformat="{:.1f}",
                xticklabelcolor=is_phantom_row ? transparent_text_color : text_color,
                xgridvisible=cell_active,
                ygridvisible=cell_active,
                leftspinevisible=cell_active, rightspinevisible=cell_active,
                topspinevisible=cell_active, bottomspinevisible=cell_active,
                # v_i on x, p_i on y -- every diagonal cell shows both,
                # unconditionally (same "every cell always" rule as ticks
                # above): each diagonal is the unique anchor identifying
                # which variable its row/column represents.
                # Plain "" (not L"") for the inactive/hidden case -- an empty
                # LaTeXString crashes Makie's glyph-collection computation
                # even when xlabelvisible=false (confirmed empirically: the
                # visibility flag doesn't skip glyph layout for the
                # underlying text, only its own rendering).
                xlabel=if is_phantom_row
                    L"v_%$(_idxs[n_active])"
                elseif cell_active
                    L"v_%$(_idxs[i])"
                else
                    ""
                end,
                ylabel=cell_active ? L"p_%$(_idxs[i])" : "",
                xlabelvisible=show_x_decor,
                xlabelcolor=is_phantom_row ? transparent_text_color : text_color,
                ylabelvisible=cell_active,
            )

            for j in i+1:n
                # NOTE: upper/lower cells at mirrored grid positions reuse the same
                # computed primitive (row1=larger-index var, row2=smaller-index var),
                # to avoid computing each variable pair twice. Upper cells' axis
                # limits (below) follow that layout, but the lower-triangle cells
                # (see the `for j in 1:i-1` loop) assign x/y limits the other way
                # around, so a lower cell's x-axis doesn't align with its column's
                # diagonal cell -- inconsistent with the standard corner-plot
                # convention (every cell in column c shares column c's x-range).
                # Fixing this properly needs a transpose flag threaded through
                # every 2D recipe's compose_plotspecs (Hist2D, KDE2D, QuantileHist2D,
                # QuantileKDE2D, Hexbin2D, Scatter2D, Cov2D, Std2D, Mean2D,
                # Errorbars2D), not a local change here -- deferred as its own pass.
                upper_primitives = graph[primitive_symbol(upper_recipe, (j, i))][]
                upper_plotspecs = compose_plotspecs(upper_primitives, upper_recipe(), triagonal_config)
                stats_specs_2D = stats_upper ? get_stats_plotspecs(graph, (j, i), Makie2DStats(), triagonal_config) : PlotSpec[]
                append!(upper_plotspecs, stats_specs_2D)

                ylims = graph[Symbol("axis_limits_$j")][]
                # i < j always in this loop, so j <= n_active already implies
                # i <= n_active -- checking j alone is sufficient here.
                cell_active_upper = j <= n_active
                matrix[i, j] = S.Axis(
                    plots=upper_plotspecs,
                    aspect=1,
                    limits=(ylims, xlims),
                    # Every cell shows its own bottom/left ticks+labels now --
                    # see the diagonal cell's matching comment above. No
                    # xaxisposition/yaxisposition override either (was :top/
                    # :right), so this defaults to the same bottom/left as
                    # every other cell.
                    xticklabelsvisible=cell_active_upper, xticksvisible=false,
                    yticklabelsvisible=cell_active_upper, yticksvisible=false,
                    yticklabelrotation=pi / 2,
                    xgridvisible=cell_active_upper,
                    ygridvisible=cell_active_upper,
                    leftspinevisible=cell_active_upper, rightspinevisible=cell_active_upper,
                    topspinevisible=cell_active_upper, bottomspinevisible=cell_active_upper,
                    # See the diagonal cell's comment above re: plain "" vs L"".
                    xlabel=cell_active_upper ? L"v_%$(_idxs[j])" : "",
                    ylabel=cell_active_upper ? L"v_%$(_idxs[i])" : "",
                    xlabelvisible=cell_active_upper,
                    ylabelvisible=cell_active_upper,
                )
            end
            for j in 1:i-1
                lower_primitives = graph[primitive_symbol(lower_recipe, (i, j))][]
                lower_plotspecs = compose_plotspecs(lower_primitives, lower_recipe(), triagonal_config)
                stats_specs_2D = stats_lower ? get_stats_plotspecs(graph, (i, j), Makie2DStats(), triagonal_config) : PlotSpec[]
                append!(lower_plotspecs, stats_specs_2D)

                ylims = graph[Symbol("axis_limits_$j")][]
                # j < i always in this loop, so i <= n_active already implies
                # j <= n_active -- checking i alone is sufficient here.
                cell_active_lower = i <= n_active
                matrix[i, j] = S.Axis(
                    plots=lower_plotspecs,
                    aspect=1,
                    limits=(xlims, ylims),
                    # Every cell shows its own bottom/left ticks+labels now --
                    # see the diagonal cell's matching comment above (this
                    # cell already defaulted to bottom/left, unlike upper).
                    xticklabelsvisible=cell_active_lower, xticksvisible=false,
                    yticklabelsvisible=cell_active_lower, yticksvisible=false,
                    yticklabelrotation=pi / 2,
                    xgridvisible=cell_active_lower,
                    ygridvisible=cell_active_lower,
                    leftspinevisible=cell_active_lower, rightspinevisible=cell_active_lower,
                    topspinevisible=cell_active_lower, bottomspinevisible=cell_active_lower,
                    # See the diagonal cell's comment above re: plain "" vs L"".
                    xlabel=cell_active_lower ? L"v_%$(_idxs[i])" : "",
                    ylabel=cell_active_lower ? L"v_%$(_idxs[j])" : "",
                    xlabelvisible=cell_active_lower,
                    ylabelvisible=cell_active_lower,
                )
            end
        end
        return S.GridLayout(matrix; rowsizes=cellsizes, colsizes=cellsizes)
    end

    return gridlayout
end


# A color that visually separates from `bg` in whichever direction actually
# increases contrast, without hardcoding one shade per theme: darkening a
# fixed amount reads fine starting from a light color but is imperceptible
# starting from an already-near-black one (0.10 darkened by 0.08 is still
# 0.02, visually identical) -- so this shifts *away* from whichever extreme
# `bg` is already closer to (darker if bg is light, lighter if bg is dark).
# Used twice in a row (see bat_theme()/bat_theme_dark()) to build a 3-step
# ladder -- page background, then the UI panel a step further, then the
# widgets on top of the panel a further step still -- each step using the
# *previous* step's own color as its `bg`, so the two gaps (bg-to-panel,
# panel-to-widget) are always identical regardless of amount.
function _panel_bg_color(bg, amount::Real=0.08)
    rgb = Colors.RGB(Makie.to_color(bg))
    luminance = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b
    delta = luminance > 0.5 ? -amount : amount
    return Colors.RGB(clamp(rgb.r + delta, 0, 1), clamp(rgb.g + delta, 0, 1), clamp(rgb.b + delta, 0, 1))
end

function _build_fig(
    graph::ComputeGraph,
    gridlayout::Any,
    picker_info::Union{NamedTuple,Nothing}=nothing;
    has_chain_info::Bool=false
)
    # The whole column (main grid, toggle_row, controls_layout -- see below)
    # is locked to a single width via colsize!(fig.layout, 1, Aspect(1,1)),
    # which always resolves to whatever height row 1 (the grid) ends up with
    # -- entirely independent of the Figure's own declared width. A
    # significantly-wider-than-tall size like the previous (900, 700) default
    # left ~450px of pure dead canvas beside the content (confirmed
    # empirically: at (900,700) the grid+controls column resolves to only
    # ~446px wide, centered, with ~227px of untouched margin on each side).
    # (665, 850) is sized to exactly match that column's natural width at
    # that height plus the left/right margin set below -- both numbers need
    # re-measuring if the grid's Relative(0.8)/Aspect(1,1) row/col sizing
    # (rowsize!/colsize! below) or the controls row's content ever change,
    # since they're tied to this specific layout's geometry, not derived
    # analytically. The window remains fully resizable regardless (GLMakie)
    # or just affects the initial render size (CairoMakie).
    fig = Figure(size=(665, 850))
    # Widens Makie's default 16px figure margin on every side it's actually
    # needed on, confirmed empirically by measuring each side's peak
    # protrusion overflow (text rendered beyond the canvas edge -- silently
    # invisible rather than clipped-looking, since CairoMakie/GLMakie simply
    # don't rasterize anything outside the canvas). Every cell now shows its
    # own bottom/left tick labels + axis labels (see _init_gridlayout), so
    # only the true bottom row and true leftmost column's protrusion can ever
    # reach the Figure's own outer edge -- top/right stay at the default 16px
    # (nothing is ever positioned there anymore).
    #  - left: the leftmost column's p_i/v_i y-labels are sized off the
    #    *global* fontsize=20 (bat_theme), not the shrunk xticklabelsize/
    #    yticklabelsize=10 -- so they need much more room than the tick
    #    labels do, especially now that the Figure's width is tightly fit to
    #    the grid (no more incidental slack from a much-wider-than-tall
    #    default to absorb them in). 44px (not just enough for a
    #    single-digit "v_1") comfortably covers 2-digit variable indices too
    #    ("v_15" etc.) -- confirmed empirically across N_max=3/5/15 and both
    #    full and partial vsel selections.
    #  - right: mirrors left for symmetry/robustness (e.g. if a future change
    #    ever re-adds right-side content), though nothing currently needs it.
    #  - top: the diagonal's own p_i label sits above its tick labels, same
    #    shape of problem as the old top-of-grid case, just now on every
    #    diagonal cell instead of only the row-1 one -- 40px confirmed
    #    empirically sufficient (with margin) across the same scenarios above.
    #  - bottom: comfortably fits in the default 16px, no widening needed.
    # Re-measure all of this again if xlabelsize/ylabelsize, xticklabelsize/
    # yticklabelsize, or the Figure size above change.
    fig.layout.alignmode[] = Outside(44, 44, 16, 40)

    plot(fig[1, 1], gridlayout)

    colsize!(fig.layout, 1, Aspect(1, 1))
    # Relative(0.8) recomputes on every relayout, not just on real window
    # resizes -- this previously caused a small but real, noticeable size
    # jump in this row (and, via the Aspect(1,1) column lock above, the whole
    # column) whenever the vsel selection shrank/grew, with no window resize
    # involved. Root cause (fully isolated, not just worked around): a real
    # GridLayoutBase bug where the corner grid's *reported* bottom protrusion
    # depends on whether the grid's structurally-last row happens to be
    # active, not on the actual last *active* row's real content -- see the
    # phantom-protrusion fix (text_color/is_phantom_row) and its long comment
    # in _init_gridlayout for the full diagnosis. That fix keeps the grid's
    # own reported protrusion stable regardless of n_active, which is what
    # actually stopped Relative(0.8)'s resolved size from depending on
    # n_active at all -- Relative(0.8) itself was never the problem, just the
    # unstable input it was reacting to. Two fix attempts were tried and
    # discarded before this one: driving a Fixed row height off
    # `fig.scene.viewport` (verified only in headless CairoMakie, made things
    # worse in a real GLMakie window -- likely a DPI/framebuffer-vs-logical-
    # size mismatch), and GridLayoutBase's own alignmode=Mixed(bottom=
    # Protrusion(...)) escape hatch (confirmed correct in isolated tests, but
    # this GridLayoutBase version's update! unconditionally calls
    # determinedirsize on every relayout regardless of alignmode, and that
    # function doesn't handle Mixed at all -- crashed deterministically on
    # the very first live-sampling render). The current fix instead operates
    # entirely within the already-battle-tested Inside() alignmode code path
    # (no Mixed, no viewport/DPI/backend-specific concept involved), so it
    # should behave identically in GLMakie -- still confirm in a real
    # interactive session before treating this as fully closed, per the
    # lesson from the first attempt.
    rowsize!(fig.layout, 1, Relative(0.8))

    # Always-visible row holding just the controls-collapse toggle (and the
    # current-index slider, see below) -- kept outside controls_layout so it
    # never disappears along with the block it's meant to reveal. Visually
    # separated from the rest of the figure via a rounded, subtly-shaded
    # panel behind it: toggle_row itself is a 3x3 wrapper of Fixed(ui_box_pad)
    # margin rows/cols around a single inner content cell (toggle_row_content,
    # holding the actual button/label/slider below) purely so the Box
    # (assigned to the same fig[2,1] span, and thus filling the exact same
    # resolved bbox as toggle_row itself -- confirmed empirically that a Box
    # and a GridLayout can coexist at the same grid position, the Box acting
    # as a background since it's created first and Makie draws blocks in
    # creation order) shows a visible inset margin around the content instead
    # of a border tightly hugging it. GridLayout(3, 3) (explicit dims, not
    # left to auto-expand from content) is required for rowsize!/colsize! to
    # accept row/col 3 at all -- nothing is ever assigned directly into it.
    # ui_box_pad is a third of the ambient fontsize (not a fixed pixel guess),
    # so it scales with the theme rather than needing separate re-tuning.
    # rowgap!/colgap!(..., 0): GridLayoutBase's own default inter-cell gap
    # (applied *between* every pair of the 3 rows/cols regardless of their
    # own sizes) would otherwise stack on top of these margins and make the
    # padding come out uneven/much larger than ui_box_pad -- confirmed
    # empirically (padding was ~3.7x the intended value with the default gap
    # left in place).
    ui_box_pad = fig.scene.theme[:fontsize][] / 3
    toggle_row = fig[2, 1] = GridLayout(3, 3)
    rowgap!(toggle_row, 0)
    colgap!(toggle_row, 0)
    Box(fig[2, 1], color=_panel_bg_color(fig.scene.backgroundcolor[]), cornerradius=10, strokewidth=0)
    toggle_row_content = toggle_row[2, 2] = GridLayout()
    rowsize!(toggle_row, 1, Fixed(ui_box_pad))
    rowsize!(toggle_row, 3, Fixed(ui_box_pad))
    colsize!(toggle_row, 1, Fixed(ui_box_pad))
    colsize!(toggle_row, 3, Fixed(ui_box_pad))
    # Auto (not a hardcoded Fixed height guess) so this always matches
    # whatever toggle_row_content's actual content needs, regardless of
    # ui_box_pad or font-size changes -- toggle_row's own height is left at
    # its Block/GridLayout default (unlike width, never set to `nothing`
    # elsewhere in this file), which is exactly the recursively-computed
    # natural height Auto needs here (Fixed(pad) + content + Fixed(pad)).
    rowsize!(fig.layout, 2, Auto())

    # Everything else (recipe/stats menus and the vsel picker matrix) now
    # lives nested inside a single collapsible block, itself entirely within
    # column 1 -- the same column the main grid occupies and whose width is
    # locked to the plot's own aspect ratio (colsize! above). That means no
    # amount of UI content can ever push the figure wider than the corner
    # plot itself, and collapsing/expanding all of it is a single row-height
    # toggle rather than something tracked per-widget. Same rounded-panel
    # treatment as toggle_row above (Box behind a 3x3 Fixed(ui_box_pad)-margin
    # wrapper) -- see its comments for why each piece is needed. Matches
    # toggle_row's width automatically, with no extra effort: both defer to
    # the same Aspect(1,1)-locked column (colsize! above), so they're always
    # identical regardless of either box's own content.
    controls_layout = fig[3, 1] = GridLayout(3, 3)
    rowgap!(controls_layout, 0)
    colgap!(controls_layout, 0)
    controls_box = Box(fig[3, 1], color=_panel_bg_color(fig.scene.backgroundcolor[]), cornerradius=10, strokewidth=0)
    rowsize!(controls_layout, 1, Fixed(ui_box_pad))
    rowsize!(controls_layout, 3, Fixed(ui_box_pad))
    colsize!(controls_layout, 1, Fixed(ui_box_pad))
    colsize!(controls_layout, 3, Fixed(ui_box_pad))
    # Gap between the two boxes (toggle_row's above, controls_layout's below)
    # -- rowgap! with an explicit row index targets only *this* gap, unlike
    # the blanket rowgap!(gl, gap) form used to zero out the wrapper grids'
    # own internal gaps above. Only valid now that fig.layout actually has 3
    # rows (i.e. after controls_layout's own fig[3,1] assignment above) --
    # confirmed empirically that calling this any earlier throws ("invalid
    # row gap 2"), since GridLayoutBase only grows fig.layout's row count as
    # content gets assigned to new rows.
    rowgap!(fig.layout, 2, ui_box_pad)

    # ui_layout holds the recipe/stats menus (columns 1-3) *and* the vsel
    # picker's title/matrix (column 4, added in _build_vsel_picker! below) --
    # all in the same GridLayout, deliberately, so the picker's title shares
    # ui_layout's own row 1 (guaranteeing its top/bottom edges exactly match
    # the "Recipe"/"Stats overlay" header labels there) and the matrix spans
    # rows 2:4 (guaranteeing its bottom edge exactly matches menu_lower's
    # row), with no separate alignment bookkeeping needed -- two blocks in
    # the same grid row/rows share pixel-exact boundaries by construction.
    ui_layout = controls_layout[2, 2] = GridLayout()

    # A nested GridLayout's `width` attribute defaults to `Auto()`, just like
    # a Block's -- and just like Menu (see _set_block_visible!'s comments),
    # `Auto()` means "use my own bottom-up-computed size instead of whatever
    # the parent cell suggests", not "fill the parent cell". Confirmed
    # empirically: without this, these end up sized from their own content
    # instead of the intended column-1 width, overflowing past the main grid
    # in some cases and auto-centering short of it in others. `width =
    # nothing` (like Menu's own default) makes a block/layout defer to the
    # suggested bbox unconditionally, which is the actual "fill this cell"
    # behavior we want here -- ui_layout is included again now that it's
    # controls_layout's sole content (the picker moved inside it, see above).
    for gl in (toggle_row, toggle_row_content, controls_layout, ui_layout)
        gl.width[] = nothing
    end

    options2D = [
        ("QuantileHist", QuantileHist2D),
        ("Hist", Hist2D),
        ("Scatter", Scatter2D),
        ("Hexbin", Hexbin2D),
        ("QuantileKDE", QuantileKDE2D),
        ("KDE", KDE2D),
    ]
    # Only offered when the plotted samples actually carry chain identity
    # (see _samples_have_chain_ids in makie_scatter.jl) -- meaningless (and
    # would just show every point as a single color) for samples from a
    # non-MCMC sampler.
    has_chain_info && push!(options2D, ("Scatter (by chain)", ChainScatter2D))
    options1D = [
        ("QuantileHist", QuantileHist1D),
        ("Hist", Hist1D),
        ("KDE", KDE1D),
        ("QuantileKDE", QuantileKDE1D),
        ("PDF", PDF1D),
    ]

    default_upper = options2D[findfirst(x -> x[2] == graph[:upper_recipe][], options2D)][1]
    default_diag = options1D[findfirst(x -> x[2] == graph[:diagonal_recipe][], options1D)][1]
    default_lower = options2D[findfirst(x -> x[2] == graph[:lower_recipe][], options2D)][1]

    menu_upper = Menu(
        fig,
        options=options2D,
        default=default_upper
    )
    menu_diagonal = Menu(
        fig,
        options=options1D,
        default=default_diag
    )
    menu_lower = Menu(
        fig,
        options=options2D,
        default=default_lower
    )

    curr_idxs = graph[:current_idxs][]
    # curr_idxs[1] being non-empty only means the (single) chain has at least
    # one walker -- it says nothing about whether any samples have actually
    # been produced yet. Without also checking current_idx > 0, this builds a
    # Slider with an empty 1:0 range (crashes) whenever _build_fig runs before
    # any samples exist, which for the live path is always (it's called
    # synchronously at figure construction, before the first async flush).
    show_slider = length(curr_idxs) == 1 && !isempty(curr_idxs[1]) && graph[:current_idx][] > 0
    # The index slider lives in toggle_row (always visible, right of the
    # collapse button) rather than inside ui_layout/controls_layout, per
    # explicit request -- unlike the rest of the recipe/stats controls, it
    # should never disappear when those are collapsed.
    if show_slider
        # "Current Index" sits in its own row directly above the slider
        # (row 1), the slider itself in row 2. The label spans *both*
        # column 2 (the slider) and column 3 (the value display) rather than
        # just column 2 -- deliberately, so its centered position (halign
        # =:center, Label's default) is the midpoint of that whole 2:3 span,
        # which runs from the slider's left edge (immediately after the
        # fixed-width button) to the row's own right edge, both of which are
        # constant. Centering over column 2 alone instead would move the
        # label every time the value display's text width changes (e.g.
        # "500" vs "50000"), since that changes how the fixed remaining
        # width splits between columns 2 and 3 even though their *union*
        # doesn't move -- confirmed empirically.
        lbl_idx_title = Label(toggle_row_content[1, 2:3], "Current Index")
        slider_curr_idx = Slider(toggle_row_content[2, 2], range=1:graph[:current_idx][], startvalue=graph[:current_idx][])
        # No gap between the label/slider rows -- and each pinned to the
        # outer edge of its own row (valign=:top / :bottom) rather than the
        # default :center, which would otherwise leave slack split above the
        # label and below the slider (confirmed empirically: with the
        # default :center, the slider's bottom sat ~5px short of the button's
        # bottom, even with the row gap zeroed). Pinning outward is what
        # makes the label's top and the slider's bottom land exactly on the
        # button's own top/bottom edges below.
        rowgap!(toggle_row_content, 1, 0)
        lbl_idx_title.valign[] = :top
        slider_curr_idx.valign[] = :bottom
        # Deferring to whatever column 2 is given (rather than the slider's
        # own natural/Auto width) is what makes it actually span the row's
        # remaining width -- see the width[]=nothing comments above for why
        # Auto() wouldn't do this (Slider, like most Blocks, reports a real
        # non-nothing autosize, which would leave its column sized to just
        # that instead of expanding to fill whatever's left).
        slider_curr_idx.width[] = nothing
        # The label also can't be left reporting its own Auto() width to the
        # column: a column's Auto width is the max over every (single-span)
        # item placed in it across all rows, skipping only ones reporting
        # `nothing` -- confirmed empirically that leaving it in place made
        # column 2 "determined" at the label's own (much narrower) text
        # width, shrinking the slider to match instead of filling the row.
        # But unlike the slider, `width[]=nothing` is the wrong fix here: it
        # makes the label's *own* bbox span the full column too, and Label
        # always positions its text at `bbox.origin + 0.5*textwidth` (its own
        # halign attribute isn't consulted for single-line text placement,
        # only for word-wrap justification) -- so a full-width bbox renders
        # the text hard against the *left* edge instead of centered
        # (confirmed empirically). `tellwidth=false` instead removes it from
        # the column's width computation without touching its own width
        # attribute (still Auto(), i.e. sized tightly to its own text), so
        # its computed bbox stays text-sized and its halign=:center (Label's
        # own default) centers *that* box within column 2's full width.
        lbl_idx_title.tellwidth[] = false
        lbl_idx_value = Label(toggle_row_content[2, 3], lift(string, slider_curr_idx.value))
        # Column 1 (button) and 3 (value label) are left at their Auto()
        # default, so they size to their own content and column 2 (the only
        # one reporting no natural width) absorbs whatever's left over.
    else
        # No slider to fill the remaining width in this case -- the button's
        # own column then needs to explicitly claim the row's full width for
        # its halign=:left to be flush against the true left edge instead of
        # a shrink-wrapped, auto-centered one (same reasoning as above).
        colsize!(toggle_row_content, 1, Relative(1))
    end

    lbl_upper = Label(fig, "Upper")
    ui_layout[2, 1] = lbl_upper
    lbl_diag = Label(fig, "Diagonal")
    ui_layout[3, 1] = lbl_diag
    lbl_lower = Label(fig, "Lower")
    ui_layout[4, 1] = lbl_lower

    lbl_recipe = Label(fig, "Recipe")
    ui_layout[1, 2] = lbl_recipe
    ui_layout[2, 2] = menu_upper
    ui_layout[3, 2] = menu_diagonal
    ui_layout[4, 2] = menu_lower

    lbl_stats = Label(fig, "Stats overlay")
    ui_layout[1, 3] = lbl_stats
    toggle_upper = Toggle(ui_layout[2, 3], active=false)
    toggle_diag = Toggle(ui_layout[3, 3], active=false)
    toggle_lower = Toggle(ui_layout[4, 3], active=false)

    # Collected here (rather than only living as local variables) so the
    # whole-controls collapse toggle below can show/hide every one of them
    # together -- one mechanism, not one per widget.
    ui_blocks = Any[
        lbl_upper, lbl_diag, lbl_lower, lbl_recipe,
        menu_upper, menu_diagonal, menu_lower,
        lbl_stats, toggle_upper, toggle_diag, toggle_lower,
    ]

    colsize!(ui_layout, 1, Auto())
    colsize!(ui_layout, 2, 200)
    colsize!(ui_layout, 3, Auto())

    rowsize!(controls_layout, 2, Auto())

    if show_slider
        # show_slider guarantees a single chain, but that chain may still have
        # multiple walkers; pan all of them to the same position.
        on(slider_curr_idx.value) do curr_idx
            n_walkers_here = length(graph[:current_idxs][][1])
            update!(graph, current_idxs=[fill(curr_idx, n_walkers_here)])
        end
    end

    on(menu_upper.selection) do selected_recipe
        update!(graph, upper_recipe=selected_recipe)
    end
    on(menu_diagonal.selection) do selected_recipe
        update!(graph, diagonal_recipe=selected_recipe)
    end
    on(menu_lower.selection) do selected_recipe
        update!(graph, lower_recipe=selected_recipe)
    end

    on(toggle_upper.active) do is_live
        update!(graph, show_stats_upper=is_live)
    end

    on(toggle_diag.active) do is_live
        update!(graph, show_stats_diag=is_live)
    end

    on(toggle_lower.active) do is_live
        update!(graph, show_stats_lower=is_live)
    end

    rescale_picker! = nothing
    lbl_marginals = nothing
    if !isnothing(picker_info)
        (; N, N_max, initial_vsel, apply_vsel!) = picker_info
        picker_blocks, rescale_picker!, lbl_marginals = _build_vsel_picker!(
            fig, ui_layout, graph, N, N_max, initial_vsel, apply_vsel!
        )
        append!(ui_blocks, picker_blocks)
    end

    # Single toggle that collapses/expands the *entire* controls block
    # (recipe/stats menus and the always-visible vsel picker matrix) down to
    # zero height, so the main grid can use the full window when the UI
    # isn't needed.
    collapse_button = Button(fig, label="☰", halign=:left, valign=:top)
    if show_slider
        # Spans both of the "Current Index" label/slider rows so its own top
        # and bottom edges land exactly on theirs. `height=Relative(1)` (not
        # the Auto() default) is what makes it *fill* that combined span
        # exactly rather than centering its own natural height within it --
        # Relative sizing uses the assigned bbox height directly as-is, so
        # there's no leftover space left to offset the alignment. Only safe
        # here because rows 1/2 have other content (the label/slider) to
        # derive a real height from; with no slider, the button is the only
        # content in row 1 and needs its own natural (Auto) height instead.
        collapse_button.height[] = Relative(1)
        toggle_row_content[1:2, 1] = collapse_button
    else
        toggle_row_content[1, 1] = collapse_button
    end
    # ---- Reactive whole-UI scale factor -----------------------------------
    # Everything above (fontsize, ui_box_pad, the fixed 200px menu column,
    # widget width/height/size/markersize) is sized in absolute pixels tied to
    # bat_theme's design-time fontsize and this function's Figure(665, 850)
    # default -- correct at that design size, but none of it shrinks if the
    # actual window/screen ends up smaller. Collapsing the controls panel
    # avoids the problem (it's Fixed(0)), but *expanding* it demands the full
    # unscaled pixel content, which can clip past a small screen's bounds with
    # no window resize involved (reported directly: the panel clipped when
    # expanded on a smaller screen).
    #
    # ui_scale is current-figure-width / _UI_DESIGN_WIDTH, recomputed from
    # fig.layout's own resolved bbox -- the same kind of GridLayoutBase-
    # internal, logical-coordinate signal rescale_picker! above already uses
    # safely, deliberately NOT fig.scene.viewport (raw window/framebuffer
    # pixels): driving a row size directly off fig.scene.viewport already
    # caused one confirmed GLMakie-only regression elsewhere in this function
    # (see the rowsize!(fig.layout, 1, Relative(0.8)) comment above) -- a
    # likely DPI/framebuffer-vs-logical-size mismatch that CairoMakie has no
    # equivalent of, so it looked correct in every headless test there. This
    # mechanism instead only ever reacts to fig.layout's own top-level bbox,
    # which -- unlike picker_layout's local one above -- doesn't depend on any
    # of the nested widget sizes it drives (fig.layout's extent is set
    # directly by the Figure's own size, not derived from its content), so
    # there's no analogous re-notify-on-nested-resize loop risk here; the
    # dedup guard below is kept anyway, purely defensively.
    #
    # _UI_DESIGN_WIDTH and every base_* constant below were measured directly
    # (not derived analytically) against this function's actual
    # Figure(665, 850)/bat_theme(fontsize) defaults -- re-measure all of them
    # if the Figure size, fig.layout.alignmode's margins, the corner-grid's
    # Relative(0.8)/Aspect(1,1) sizing, or bat_theme's fontsize ever change.
    # Approximate, not exact: uses the *whole* figure's resolved width as the
    # scale reference without separately accounting for the fixed (never
    # itself scaled) Outside() margins -- fine for the "stop clipping on a
    # modestly smaller screen" goal this targets, not pixel-exact at extreme
    # window sizes.
    _UI_DESIGN_WIDTH = 665.0
    # Rows sized with Relative() in a GridLayout that also has Auto()/Fixed()
    # rows are each resolved as `fraction * (total - Auto/Fixed rows' own
    # heights and gaps)` *independently* of one another -- confirmed
    # empirically (not documented behavior found for this GridLayoutBase
    # version) -- they do NOT renormalize against each other, so if this
    # value plus the corner grid's own Relative(0.8) (rowsize!(fig.layout,
    # 1, ...) above) sum to more than 1.0, the panel overflows past the
    # bottom of the figure by exactly that excess fraction, REGARDLESS of
    # how much content it actually holds. 0.18 (measured to leave a small
    # margin under the 0.2 headroom Relative(0.8) leaves) was chosen this
    # way, not to match any particular content height -- content is instead
    # made to fit *within* whatever this resolves to, via the s_height
    # computation below.
    CONTROLS_RELATIVE_HEIGHT = 0.18
    base_menu_height = 36.0
    base_button_width = 34.0
    base_button_height = 39.0
    base_toggle_width = 32.0
    base_toggle_height = 18.0
    base_toggle_markersize = 18.0
    base_slider_height = 10.0
    base_ui_col2_width = 200.0
    current_ui_scale = Ref(1.0)

    # `expanded` is passed explicitly (rather than this closing over
    # controls_visible directly) so this can be defined -- and safely called
    # once immediately, see the reactive hook below -- before
    # controls_visible even exists.
    #
    # Menu/Toggle/Label/Button/Slider width-height writes are skipped
    # entirely while collapsed: _set_block_visible! (below) caches each
    # block's width/height into a WeakKeyDict the *first* time it's hidden,
    # then restores that exact cached pair on every later expand -- writing a
    # scaled value here while already collapsed would either get silently
    # cached as the new "natural" size (wrong: that's a collapsed-size-derived
    # value, not the true design size) or, if already cached from an earlier
    # expand, get clobbered right back to the stale pre-scale value on the
    # next expand anyway. controls_visible's own handler explicitly
    # re-invokes this function (with expanded=true) right after every expand
    # for exactly this reason -- confirmed empirically necessary: without it,
    # expanding after this ran while collapsed left every menu back at its
    # original unscaled Auto() height, exactly mirroring why rescale_picker!
    # is already re-invoked there too.
    function rescale_ui!(s::Real, expanded::Bool)
        current_ui_scale[] = s

        pad = ui_box_pad * s
        for gl in (toggle_row, controls_layout)
            rowsize!(gl, 1, Fixed(pad))
            rowsize!(gl, 3, Fixed(pad))
            colsize!(gl, 1, Fixed(pad))
            colsize!(gl, 3, Fixed(pad))
        end
        rowgap!(fig.layout, 2, pad)
        colsize!(ui_layout, 2, base_ui_col2_width * s)

        expanded || return nothing

        fontsize_scaled = fig.scene.theme[:fontsize][] * s
        # lbl_marginals ("Displayed Marginals") is a header sharing this same
        # row/style, deliberately excluded from rescale_picker!'s own cell-
        # driven fontsize scaling (see its comment) -- it needs to track this
        # group instead, or its unscaled width dominates ui_layout's column 4
        # Auto-width and drags the whole panel wider than the actual scaled
        # content (confirmed empirically: this is what caused the panel to
        # overflow past the figure bounds even after everything else here was
        # already scaling correctly).
        for lbl in (lbl_upper, lbl_diag, lbl_lower, lbl_recipe, lbl_stats, lbl_marginals)
            isnothing(lbl) || (lbl.fontsize[] = fontsize_scaled)
        end
        for menu in (menu_upper, menu_diagonal, menu_lower)
            menu.fontsize[] = fontsize_scaled
            menu.height[] = base_menu_height * s
        end
        for tgl in (toggle_upper, toggle_diag, toggle_lower)
            tgl.width[] = base_toggle_width * s
            tgl.height[] = base_toggle_height * s
            tgl.markersize[] = base_toggle_markersize * s
        end
        collapse_button.fontsize[] = fontsize_scaled
        collapse_button.width[] = base_button_width * s
        # height is left at its Relative(1) assignment above when
        # show_slider (it already fills the label+slider span, which is
        # itself scaled below -- overriding it with a numeric height here
        # would silently detach it from that span again).
        show_slider || (collapse_button.height[] = base_button_height * s)
        if show_slider
            lbl_idx_title.fontsize[] = fontsize_scaled
            lbl_idx_value.fontsize[] = fontsize_scaled
            slider_curr_idx.height[] = base_slider_height * s
        end
        return nothing
    end

    # Starts collapsed (false) per explicit request -- update=true is what
    # makes the handler actually apply that at construction time, since `on`
    # only fires on *future* notifications by default and every ui_block
    # starts out fully visible/expanded otherwise.
    controls_visible = Observable(false)
    on(controls_visible; update=true) do vis
        # Relative (not Auto()) when expanded -- bounds the panel to a fixed
        # *fraction* of whatever the current figure height actually is,
        # rather than letting it demand however much its content naturally
        # needs. Confirmed empirically necessary: with Auto(), shrinking the
        # window (see rescale_ui!'s ui_scale computation below, which needs
        # this bound to correctly account for vertical space too) still let
        # the panel's content overflow past the bottom edge, since Auto()
        # rows don't respect any total budget -- only the *individual*
        # widget heights rescale_ui! sets were shrinking, not their sum.
        # CONTROLS_RELATIVE_HEIGHT (below) is deliberately *not* chosen to
        # match this panel's natural content height -- see its own comment:
        # Relative() rows here don't renormalize against each other, so it's
        # bounded by the corner grid's Relative(0.8) leaving at most ~0.2 of
        # headroom, and content is instead scaled down (via s_height below)
        # to fit within whatever that bound resolves to, even at the design
        # size -- re-measure _UI_DESIGN_HEIGHT_BUDGET below if the controls
        # panel's row/column contents ever change.
        rowsize!(fig.layout, 3, vis ? Relative(CONTROLS_RELATIVE_HEIGHT) : Fixed(0))
        # controls_box doesn't go through _set_block_visible! (it's not part
        # of ui_blocks -- it's the always-created panel background, not a
        # collapsible widget), so its own visibility needs setting directly:
        # deferring to fig.layout row 3's Fixed(0) alone left a ~1px sliver
        # visible at the collapsed height (floating-point rounding in the
        # nested GridLayout size resolution, confirmed empirically -- never
        # exactly 0), which a plain flat-color Box has no automatic occlusion
        # for the way zero-area content naturally would.
        controls_box.visible[] = vis
        for b in ui_blocks
            _set_block_visible!(b, vis)
        end
        # The per-widget restore loop above processes ui_layout's menus
        # before the picker matrix's own blocks, so mid-loop the matrix's
        # available height is transiently smaller than its final value --
        # _set_block_visible! would otherwise leave checkboxes/labels sized
        # to that transient value instead of the correct final one (confirmed
        # empirically). This forces one final, correct pass once everything
        # else has fully settled.
        vis && !isnothing(rescale_picker!) && rescale_picker!()
        # Same idea, for the same reason, for the menus/labels/button/slider
        # this function's own rescale_ui! manages -- see its docstring above.
        vis && rescale_ui!(current_ui_scale[], true)
    end
    on(collapse_button.clicks) do _
        controls_visible[] = !controls_visible[]
    end

    # No dedicated initial call needed beyond `update=true` below: every
    # widget's already-constructed default state already *is* the correct
    # s=1 appearance (the base_* constants above were measured directly from
    # those same defaults), and controls_visible's own update=true firing just
    # above already applied the (still s=1 at this point) collapse.
    #
    # s is the *smaller* of a width-based and a height-based candidate --
    # confirmed empirically necessary: a width-only scale correctly shrunk
    # individual widgets, but a *short* window (height-constrained, not just
    # narrow) still overflowed vertically, since nothing was checking whether
    # the panel's now-bounded Relative(CONTROLS_RELATIVE_HEIGHT) row (see
    # controls_visible's handler above) actually had room for that width-
    # derived widget size. Using the min of both means neither dimension can
    # overflow, at the cost of sometimes shrinking more than strictly
    # necessary in the other dimension (e.g. a very short but wide window
    # shrinks fontsize based on height alone, leaving horizontal slack) --
    # an accepted trade-off for guaranteeing no clipping over pixel-perfect
    # sizing.
    #
    # _UI_DESIGN_HEIGHT_BUDGET is the height ui_layout's content would need
    # at s=1 (every base_* height above at its literal, unscaled value) --
    # deliberately *larger* than CONTROLS_RELATIVE_HEIGHT's own resolved
    # pixel height at the design (665, 850) size, so that s_height comes out
    # below 1.0 even at the design size and content is scaled down to
    # actually fit the bounded row rather than overflowing it. Calibrated
    # empirically by iterating against the real figure (not derived
    # analytically -- GridLayoutBase's exact Relative-row solve order wasn't
    # fully reverse-engineered, just enough of it to know the qualitative
    # relationship), with headroom added beyond the tightest value that
    # still fit at design size, since the fit was observed to degrade
    # somewhat unevenly (not perfectly linearly) as the window shrinks
    # further. Re-measure/re-tune both this and CONTROLS_RELATIVE_HEIGHT
    # together if the controls panel's row/column contents ever change.
    _UI_DESIGN_HEIGHT_BUDGET = 270.0
    last_ui_wh = Ref((-1.0, -1.0))
    on(fig.layout.layoutobservables.computedbbox; update=true) do bbox
        w, h = bbox.widths[1], bbox.widths[2]
        (w < 10 || h < 10 || (w, h) == last_ui_wh[]) && return
        last_ui_wh[] = (w, h)
        s_width = w / _UI_DESIGN_WIDTH
        s_height = (h * CONTROLS_RELATIVE_HEIGHT) / _UI_DESIGN_HEIGHT_BUDGET
        rescale_ui!(clamp(min(s_width, s_height), 0.3, 2.0), controls_visible[])
    end

    return fig
end

# Pure decision logic, kept separate from the widget wiring below so it can be
# tested directly without real mouse events (which CairoMakie can't produce).
# Checking cell (i,j) adds both i and j to the active set (a diagonal cell
# i==j just adds i); unchecking removes both. The active set -- not individual
# checkbox states -- is the source of truth, so removal is explicit rather
# than re-derived from sibling checkboxes that haven't been resynced yet
# (which would make e.g. unchecking a diagonal while some other still-checked
# pair references it snap the diagonal back on).
function _vsel_after_toggle(active_vars::Set{<:Integer}, i::Integer, j::Integer, is_on::Bool)
    return is_on ? union(active_vars, (i, j)) : setdiff(active_vars, (i, j))
end

# Whether cell (i,j) should display as checked given the active set: both
# endpoints must be selected (a diagonal cell i==j just needs i).
_checkbox_should_be_checked(active_vars::Set{<:Integer}, i::Integer, j::Integer) = (i in active_vars) && (j in active_vars)

# Most Blocks (Button/Menu/Toggle/Slider/Checkbox/...) have no direct
# `visible` attribute -- confirmed empirically (hasproperty(b, :visible) is
# false for all of these) -- so toggling their underlying blockscene's
# visibility is the general mechanism. Label/Box are the exception: they do
# have their own `visible` attribute directly, and use it instead.
#
# blockscene visibility alone isn't enough for these, though: an interactive
# block's own click handling (Menu's dropdown, Button's click, Toggle/Slider
# drag, Checkbox's toggle) is driven by a purely geometric hit-test against
# the block's *own computed bounding box* -- entirely independent of
# `visible` (confirmed empirically: a Menu still opens its dropdown from a
# click at its old screen position even with blockscene.visible[]=false).
# The reason the bbox doesn't shrink on its own: GridLayoutBase's `Fixed(0)`
# row/column size is only a *suggestion* to the block; whatever the block's
# own `width`/`height` attribute resolves to takes precedence, and computes
# its bbox from its natural/reported size regardless of how small the
# assigned cell actually is. Forcing width/height to a literal `0` collapses
# the computed bbox to zero area, which *is* geometrically unclickable.
#
# The original width/height must be *remembered per block* rather than
# restored to a hardcoded `Auto()`, though: not every block defaults to
# `Auto()` -- Menu defaults to `width === nothing`, which means "just use
# whatever the parent GridLayout cell suggests" (this is exactly what gives
# every recipe menu here its uniform, column-driven width, since ui_layout's
# column 2 is a fixed 200px). `Auto()` is a *different* value with different
# semantics: it means "use my own text-derived autosize if one is available",
# and Menu unconditionally always makes one available (it recomputes its
# autosize from its current selected-text bounding box on every change) --
# so overwriting `nothing` with `Auto()` on restore silently switches a Menu
# from column-width to shrink-wrapped-to-text width permanently, which is
# exactly the "Hist"-vs-"QuantileHist" width mismatch this once caused.
const _BLOCK_NATURAL_SIZE = WeakKeyDict{Any,Tuple{Any,Any}}()
_natural_size!(b) = get!(() -> (b.width[], b.height[]), _BLOCK_NATURAL_SIZE, b)

function _set_block_visible!(b, v::Bool)
    b.blockscene.visible[] = v
    w, h = _natural_size!(b)
    b.width[] = v ? w : 0
    b.height[] = v ? h : 0
    return nothing
end
_set_block_visible!(b::Union{Label,Box}, v::Bool) = (b.visible[] = v)

# Checkbox is a further exception to the generic method above: its clickable
# area is a *fixed*-size square (its `size` attribute) centered on
# computedbbox, independent of that bbox's own dimensions (confirmed in
# Makie's checkbox.jl source) -- so zeroing width/height alone still leaves a
# fully clickable checkbox sitting at the collapsed bbox's center point. Needs
# `size` zeroed/restored too; its natural size is remembered (weakly, so a
# checkbox that's later discarded can still be GC'd) the first time it's
# hidden, since that capture must happen before it's ever been zeroed.
const _CHECKBOX_NATURAL_SIZE = WeakKeyDict{Checkbox,Float64}()
function _set_block_visible!(b::Checkbox, v::Bool)
    b.blockscene.visible[] = v
    w, h = _natural_size!(b)
    b.width[] = v ? w : 0
    b.height[] = v ? h : 0
    if v
        b.size[] = get(_CHECKBOX_NATURAL_SIZE, b, b.size[])
    else
        haskey(_CHECKBOX_NATURAL_SIZE, b) || (_CHECKBOX_NATURAL_SIZE[b] = b.size[])
        b.size[] = 0
    end
    return nothing
end

# Builds the always-visible "Displayed Marginals" title + N x N variable
# picker matrix (N = total model dimensionality, not N_max), placed as a 4th
# column of ui_layout (see _build_fig): the title shares ui_layout's row 1
# with the "Recipe"/"Stats overlay" headers (so their top/bottom edges match
# exactly, being the same grid row) and the matrix spans rows 2:4 (so its
# bottom edge exactly matches menu_lower's row) -- both guaranteed by
# GridLayoutBase, not by manual pixel-matching. The lower triangle including
# the diagonal (i >= j) is interactive -- cell (i,j) and (j,i) are the same 2D
# marginal, so only one needs a checkbox; a diagonal cell (i,i) selects
# variable i on its own. Only the strict upper triangle is decorative.
# Checking a cell on/off means "include/exclude these variable(s) in vsel";
# the actual N_max x N_max corner-plot grid always shows every pairing among
# the currently selected variables, so there's no independent per-pair
# visibility beyond which variables are selected.
#
# Takes graph/N/N_max/initial_vsel/apply_vsel! directly rather than a
# BATVisualizer so it also works for the static bat_makie_plot path, which has
# no vis/content at all. Returns (blocks, rescale_picker!, lbl_marginals):
# every block it creates, so the caller's whole-controls collapse toggle can
# fold this matrix's visibility into its own (it's no longer independently
# collapsible); a callback the caller must invoke once after fully
# re-expanding the controls; and the "Displayed Marginals" title label itself,
# so the caller's own whole-UI rescale can keep its fontsize in step with the
# other column headers (deliberately excluded from rescale_picker!'s own
# cell-driven scaling below -- see its comment). That second (callback)
# return value is not redundant with the automatic bbox-driven rescaling
# above: collapsing/expanding processes ui_blocks (menus first, then this
# matrix's own blocks) one at a time, so
# mid-restore the matrix's available height is transiently smaller than its
# final value -- confirmed empirically to otherwise leave cells sized to that
# transient value instead of the correct final one.
function _build_vsel_picker!(
    fig::Figure,
    ui_layout::GridLayout,
    graph::ComputeGraph,
    N::Integer,
    N_max::Integer,
    initial_vsel::AbstractVector{<:Integer},
    apply_vsel!::Function
)
    lbl_marginals = Label(fig, "Displayed Marginals")
    ui_layout[1, 4] = lbl_marginals

    picker_layout = ui_layout[2:4, 4] = GridLayout()
    # Deferring to the assigned row-span (rather than the matrix's own small
    # natural size) is what makes it actually *fill* that span all the way
    # down to row 4's bottom edge, instead of centering its small natural
    # size somewhere in the middle of it -- see the width[]=nothing comments
    # above for why a GridLayout's default Auto() sizing wouldn't do this.
    picker_layout.width[] = nothing
    picker_layout.height[] = nothing

    # Warning label (e.g. "can't select more than N_max...") sits below the
    # matrix, spanning its full width.
    status_label = Label(fig, "", fontsize=12, color=:red)

    initial_vsel_set = Set(initial_vsel)
    checkboxes = Dict{Tuple{Int,Int},Checkbox}()
    all_blocks = Union{Checkbox,Label,Box}[status_label, lbl_marginals]
    updating_programmatically = Ref(false)

    for j in 1:N
        lbl = Label(fig, string(j), fontsize=12)
        picker_layout[1, j+1] = lbl
        push!(all_blocks, lbl)
    end
    for i in 1:N
        lbl = Label(fig, string(i), fontsize=12)
        picker_layout[i+1, 1] = lbl
        push!(all_blocks, lbl)
        for j in 1:N
            if i >= j
                cb = Checkbox(
                    picker_layout[i+1, j+1],
                    checked=(i in initial_vsel_set && j in initial_vsel_set),
                    roundness=0
                )
                checkboxes[(i, j)] = cb
                push!(all_blocks, cb)
            else
                bx = Box(fig, color=:gray70, width=20, height=20)
                picker_layout[i+1, j+1] = bx
                push!(all_blocks, bx)
            end
        end
    end
    picker_layout[N+2, 1:(N+1)] = status_label

    # Scales the matrix's cells (checkbox click-target size, row/column
    # index label fontsize, decorative box size) to the space picker_layout
    # actually ends up filling, rather than leaving them at their small
    # default size with dead space around them now that the matrix fills the
    # whole label-to-bottom-edge span above. Unconditional (no dedup) so the
    # explicit external call the caller makes after re-expanding (see this
    # function's docstring) always forces a fresh, correct application.
    function rescale_picker!()
        h = picker_layout.layoutobservables.computedbbox[].widths[2]
        h < 10 && return
        cell = h / (N + 2) # matrix rows (N+1) plus the status-label row
        cb_size = max(4.0, 0.75 * cell)
        lbl_fontsize = clamp(0.45 * cell, 6.0, 16.0)
        for b in all_blocks
            if b isa Checkbox
                b.size[] = cb_size
            elseif b isa Box
                b.width[] = cb_size
                b.height[] = cb_size
            elseif b !== lbl_marginals && b !== status_label
                b.fontsize[] = lbl_fontsize
            end
        end
        return nothing
    end
    # Reacts to picker_layout's own resolved bbox (not a one-off calculation
    # at construction time) so it also adapts to window resizes. last_h
    # dedupes re-entrant firings *here* specifically: changing a cell's size
    # above triggers its own relayout (Label/Checkbox both feed their size
    # back into the layout machinery), which unconditionally renotifies this
    # same computedbbox Observable even when its value comes out identical
    # (GridLayoutBase doesn't skip notification on equal values here) --
    # without this guard that's an infinite loop, confirmed empirically
    # (StackOverflowError). The explicit external call above bypasses this
    # dedup entirely (calling rescale_picker! directly, not through here),
    # which is required -- see this function's docstring for why.
    last_h = Ref(-1.0)
    on(picker_layout.layoutobservables.computedbbox) do bbox
        h = bbox.widths[2]
        (h < 10 || h == last_h[]) && return
        last_h[] = h
        rescale_picker!()
    end

    active_vars = Ref(initial_vsel_set)

    for (i, j) in keys(checkboxes)
        cb = checkboxes[(i, j)]
        on(cb.checked) do is_on
            updating_programmatically[] && return

            new_vars = _vsel_after_toggle(active_vars[], i, j, is_on)

            if is_on && length(new_vars) > N_max
                updating_programmatically[] = true
                cb.checked[] = false
                updating_programmatically[] = false
                status_label.text[] = "Can't select more than $(N_max) variables at once -- deselect one first."
                return
            end

            status_label.text[] = ""
            active_vars[] = new_vars
            apply_vsel!(sort(collect(new_vars)))

            # Resync every checkbox to the new active set -- e.g. checking
            # (2,1) also checks (1,1) and (2,2) automatically, and checking
            # both (1,1) and (2,2) also checks (2,1) automatically.
            updating_programmatically[] = true
            for ((i2, j2), cb2) in checkboxes
                cb2.checked[] = _checkbox_should_be_checked(new_vars, i2, j2)
            end
            updating_programmatically[] = false
        end
    end

    return all_blocks, rescale_picker!, lbl_marginals
end

function register_state_for_vis!(
    vis::BATVisualizer{BATMakieVisualization},
    mcmc_state::MCMCState,
    samples#::Vector{DensitySampleVector}
)
    (; graph, chain_ids, output_buffer) = vis.content

    empty_walker_outputs = _empty_chain_outputs(mcmc_state)
    samples_new = _append_walker_outputs(mcmc_state, empty_walker_outputs, samples)

    samples_graph = graph[:samples][]
    push!(samples_graph, samples_new)
    update!(graph, samples=samples_graph)

    current_idxs = length.(samples)
    current_idxs_graph = graph[:current_idxs][]
    push!(current_idxs_graph, current_idxs)
    update!(graph, current_idxs=current_idxs_graph)

    push!(chain_ids, mcmc_state.chain_state.info.id)
    empty_chain_output = _empty_chain_outputs(mcmc_state)
    push!(output_buffer, empty_chain_output)
end

# Core of the vsel choke point, with no dependency on a live BATVisualizer --
# usable directly by the static bat_makie_plot path, which has no vis/content
# (and no concurrent listener task to race, so no locking needed there).
# Clamps the requested selection against both the model's dimensionality and
# the grid size (N_max), and only touches the graph if the (clamped) selection
# actually differs from what's already there.
function _apply_vsel_to_graph!(graph::ComputeGraph, n_dof::Integer, N_max::Integer, requested_vsel::AbstractVector{<:Integer})
    clamped = _clamp_vsel(requested_vsel, n_dof, N_max)
    if clamped != graph[:idxs][]
        update!(graph, idxs=clamped)
    end
    return clamped
end

# Live-path wrapper: this is what a variable-selection UI widget should call to
# change vsel while the plot is live; it's also used below for the initial
# activation once real samples exist. Guarded by buffer_lock so a future
# widget running on a different task can't race the listener's own tick.
function _apply_vsel!(vis::BATVisualizer{BATMakieVisualization}, requested_vsel::AbstractVector{<:Integer})
    (; graph, buffer_lock, n_dof) = vis.content
    lock(buffer_lock) do
        _apply_vsel_to_graph!(graph, n_dof[], vis.backend.N_max, requested_vsel)
    end
    return nothing
end

function BAT.init_visualizer!(
    vis::BATVisualizer{BATMakieVisualization};
    mcmc_states::Vector{<:MCMCState},
    outputs,#::Vector{Vector{DensitySampleVector}},
    f_pretransform::Function
)
    warmup_makie_shaders()

    vis.content.n_dof[] = totalndof(varshape(mcmc_target(mcmc_states[1])))

    # Fixed axis-limit/histogram-bin-edge domain, estimated from the prior
    # before any real samples exist (see _estimate_prior_domain) -- set here,
    # before the figure is first built, so the very first render already uses
    # it rather than some other placeholder.
    domain_lo, domain_hi = _estimate_prior_domain(mcmc_states, vis.content.n_dof[])
    update!(vis.content.graph, domain_lo=domain_lo, domain_hi=domain_hi)

    for (i, state) in enumerate(mcmc_states)
        register_state_for_vis!(vis, state, _transform_walker_outputs(f_pretransform, outputs[i]))
    end

    (; graph, buffer_lock, buffer_cond, output_buffer, n_buffer_samples, effective_batch_size, is_live, listener_task) = vis.content
    (; poll_interval, adaptive_batching, batch_growth_rate) = vis.backend

    picker_info = (
        N=vis.content.n_dof[],
        N_max=vis.backend.N_max,
        initial_vsel=vis.backend.vsel,
        (apply_vsel!)=new_vsel -> _apply_vsel!(vis, new_vsel),
    )

    # This entry point is only ever reached from MCMC sampling (mcmc_states::
    # Vector{<:MCMCState} above), so it's tempting to assume chain info is
    # always present -- checked via the real sample data anyway (same
    # predicate the static path uses), so this stays correct even if some
    # future MCMC-adjacent sampler reuses this visualizer without actually
    # producing chain-tagged samples. .info is unaffected by
    # _transform_walker_outputs (that only transforms .v), so checking the
    # untransformed first chain/walker's output is equivalent to checking
    # what's actually registered.
    has_chain_info = _samples_have_chain_ids(outputs[1][1])

    with_theme(vis.backend.dark ? bat_theme_dark() : bat_theme()) do
        gridlayout = _init_gridlayout(graph, vis.backend.N_max)
        fig = _build_fig(graph, gridlayout, picker_info; has_chain_info=has_chain_info)
        display(fig)
    end

    # force=false: only flush once effective_batch_size[] samples are buffered
    # (the normal per-tick check). force=true: flush whatever is currently
    # buffered regardless of threshold, used once for the final drain below so
    # the last partial batch isn't silently dropped when is_live[] flips false.
    function flush_buffer!(; force::Bool=false)
        lock(buffer_lock)
        update_graph = force ? n_buffer_samples[] > 0 : n_buffer_samples[] >= effective_batch_size[]
        if update_graph
            # Shallow copy: output_buffer's slots get replaced (not mutated)
            # below, so the extracted inner vectors are never touched again --
            # no need to deep-copy the actual sample data out of them.
            extracted_output_buffer = copy(output_buffer)
            output_buffer .= _empty_chain_outputs.(mcmc_states)
            n_buffer_samples[] = 0
            # Geometric growth (not on the forced final drain -- there's no
            # "next tick" for it to matter to): the same amortized-doubling
            # trick as dynamic array growth, so the number of full-dataset
            # recomputes over a run is O(log N) instead of O(N), keeping total
            # redraw work near-linear instead of quadratic in sample count.
            if !force && adaptive_batching
                effective_batch_size[] = ceil(Int, effective_batch_size[] * batch_growth_rate)
            end
            # Wakes any sampling threads blocked on the high-watermark in
            # update_visualizer_impl! -- notify while still holding buffer_lock,
            # matching Threads.Condition's contract.
            notify(buffer_cond, all=true)
        end
        unlock(buffer_lock)

        if update_graph
            fresh_batch_trafo = _transform_chain_outputs(f_pretransform, extracted_output_buffer)

            samples = graph[:samples][]
            samples_new = _append_chain_outputs(mcmc_states[1], samples, fresh_batch_trafo)

            update!(graph, samples=samples_new)

            # Derived from the actual merged length (not the raw batch length):
            # checked_push! inside _append_chain_outputs can collapse a sample at
            # the batch boundary into a weight increment instead of a new row.
            current_idxs_new = [length.(chain_samples) for chain_samples in samples_new]
            update!(graph, current_idxs=current_idxs_new)

            # Checked against only this batch (not the full accumulated
            # dataset), so this stays cheap regardless of how far into the run
            # we are -- see _widen_domain!'s docs.
            new_flat_views = Any[]
            for chain_batch in fresh_batch_trafo
                for walker_batch in chain_batch
                    push!(new_flat_views, walker_batch)
                end
            end
            _widen_domain!(graph, vcat(new_flat_views...))
        end
        return nothing
    end

    # Started only after the figure above has fully resolved its initial state --
    # the listener's first tick can mutate :idxs/:samples via _apply_vsel!, which
    # would otherwise race the still-in-progress initial construction (observed
    # in practice specifically on a cold/slow first JIT compile of the compute
    # graph's closures, where construction can take longer than poll_interval).
    listener_task[] = errormonitor(
        @async begin
            while is_live[]
                sleep(poll_interval)

                # Decoupled from the sample-batch flush below so a vsel change
                # (vis.backend.vsel mutated by a future UI widget) is picked up
                # promptly instead of waiting for the next full batch.
                _apply_vsel!(vis, vis.backend.vsel)

                flush_buffer!()
            end

            # is_live[] can flip false between two ticks (bat_sample_impl sets it
            # right before waiting on this task), which would otherwise drop
            # whatever's currently buffered -- below n_batch, or even a full
            # batch this tick hadn't gotten to yet -- leaving the display behind
            # the true final sample count. One last unconditional flush closes
            # that gap before the task returns.
            flush_buffer!(force=true)
        end
    )
end

function BAT.update_visualizer_impl!(
    vis::BATVisualizer{BATMakieVisualization};
    chain_state::MCMCChainState,
    nonzero_weights::Bool
)
    (; buffer_lock, buffer_cond, output_buffer, chain_ids, n_buffer_samples, effective_batch_size, buffer_ratio) = vis.content
    output_id = findfirst(x -> x == chain_state.info.id, chain_ids)
    n_smpls_start = sum(length.(output_buffer[output_id]))
    get_samples!(output_buffer[output_id], chain_state, nonzero_weights)
    n_smpls_end = sum(length.(output_buffer[output_id]))

    lock(buffer_lock)
    n_new = n_smpls_end - n_smpls_start
    n_buffer_samples[] += n_new
    # Backpressure: block this chain's sampling task once the buffer has grown
    # far enough ahead of the listener that the display would otherwise lag
    # noticeably behind the true latest samples. Bounded, not per-sample --
    # sampling still runs in free bursts up to max_buffered before it has to
    # wait -- so the listener remains the sole bottleneck only when it's
    # genuinely falling behind, not on every single step.
    #
    # Derived from effective_batch_size[] (scaled by the fixed configured
    # ratio) rather than a separately-tracked/grown value: this threshold must
    # always stay >= the current flush trigger, or sampling would permanently
    # block below a threshold the listener is never allowed to reach --
    # deriving it from the same value the flush trigger uses makes that
    # invariant automatic instead of something two independently-growing
    # counters could drift out of.
    # Recomputed every iteration (not hoisted out of the loop) so a waiter
    # woken by one flush re-checks against whatever effective_batch_size[]
    # grew to by the time it wakes, not a value that was already stale then.
    while n_buffer_samples[] >= ceil(Int, effective_batch_size[] * buffer_ratio)
        wait(buffer_cond)
    end
    unlock(buffer_lock)
end


function _marginal_view_dist(
    locations::SubArray,
    weights::SubArray,
    filter::Bool,
    bins::Union{Tuple{Vararg{Int64}},Int64},
    closed::Symbol,
    normalization::Symbol
)
    if filter
        mask = _low_weight_mask(weights)
        locations = view(locations, :, mask)
        weights = view(weights, mask)
    end

    cols = Tuple(eachrow(locations))
    edges = if isa(bins, Integer)
        _get_edges(cols, (bins,), closed)
    elseif bins isa Tuple
        Tuple(_get_edges(cols[i], bins[i], closed) for i in 1:length(bins))
    else
        (_get_edges(cols, bins, closed),)
    end

    hist = fit(Histogram, cols, FrequencyWeights(weights), edges, closed=closed)
    h_norm = normalization == :none ? hist : StatsBase.normalize(hist, mode=normalization)
    return h_norm
end

# Mirrors BAT.drop_low_weight_samples, but returns a mask over a bare weight
# vector instead of indexing a DensitySampleVector (locations/weights are
# already split apart into separate views by the time they get here).
function _low_weight_mask(weights::AbstractVector, fraction::Real=10^-5, threshold::Real=10^-2)
    W = float(weights)
    if minimum(W) / maximum(W) > threshold
        return trues(length(W))
    end
    W_s = sort(W)
    Q = cumsum(W_s)
    Q ./= maximum(Q)
    ind = searchsortedlast(Q, fraction)
    ind == 0 && return trues(length(W))
    thresh = W_s[ind]
    return W .>= thresh
end

function _get_bin_centers(hist::Histogram)
    edges = hist.edges
    dims = ndims(hist.weights)

    centers = [[edges[d][i] + 0.5 * (edges[d][i+1] - edges[d][i]) for i in 1:length(edges[d])-1] for d in 1:dims]

    return centers
end


function warmup_makie_shaders()
    @info "Warming up Makie shaders"
    fig = Figure()
    ax = Axis(fig[1, 1])

    barplot!(ax, [0.0], [0.0])
    stairs!(ax, [0.0, 1.0], [0.0, 0.0])
    vlines!(ax, [0.0])
    hlines!(ax, [0.0])
    errorbars!(ax, [0.0], [0.0], [0.1])

    scatter!(ax, [0.0], [0.0])
    lines!(ax, [0.0, 1.0], [0.0, 1.0])
    linesegments!(ax, [0.0, 1.0], [0.0, 1.0])

    heatmap!(ax, [0.0, 1.0], [0.0, 1.0], [0.0 1.0; 1.0 0.0])
    contourf!(ax, [0.0, 1.0], [0.0, 1.0], [0.0 1.0; 1.0 0.0])
    hexbin!(ax, [0.0], [0.0]; bins=2)
    poly!(ax, Point2f[(0, 0), (1, 0), (0, 1)])

    Makie.colorbuffer(fig)
    return nothing
end
