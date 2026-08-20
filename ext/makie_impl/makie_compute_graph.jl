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

# Untruncated counterpart of marg_symbol -- see Trace2D's own registration
# and :flat_samples_full's comment below for why it needs the *full*
# completed dataset rather than the shared current_idx-truncated view every
# other recipe uses.
function marg_full_symbol(vsel::Tuple{Int64,Int64})
    return Symbol("marg_full_$(vsel[1])$(vsel[2])")
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
# hard bound either way -- see the domain recompute in flush_buffer! below
# (init_visualizer!), which callers must rely on rather than treating this
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
#
# Non-finite values are filtered per dimension, mirroring _domain_including
# below (and for the same reason documented there at length): a completed run
# containing a single Inf/NaN sample value would otherwise poison this
# domain via raw minimum/maximum -- NaN propagates outright, and an Inf span
# silently defeats the eps-based degenerate-edges guard downstream
# (eps(Inf) is NaN) -- re-creating on the static path exactly the failure
# class the live path was hardened against. A dimension with NO finite
# values degrades to the same (0, 1) placeholder axis_limits_i already uses
# before any domain exists.
function _domain_from_samples(data::AbstractMatrix, n_dof::Integer)
    lo = Vector{Float64}(undef, n_dof)
    hi = Vector{Float64}(undef, n_dof)
    for d in 1:n_dof
        row = view(data, d, :)
        finite_row = Iterators.filter(isfinite, row)
        if isempty(finite_row)
            lo[d], hi[d] = 0.0, 1.0
        else
            lo[d], hi[d] = extrema(finite_row)
        end
    end
    return lo, hi
end

# Recomputes the graph's domain_lo/domain_hi from scratch against the FULL
# accumulated sample set on every flush, rather than incrementally widening
# a separate piece of mutable state from just the newest batch (the earlier
# design, removed). That incremental state was the root cause of two
# separate OutOfMemoryError incidents: a single non-finite sample value
# (e.g. an accepted sample that overflowed to +-Inf under a poorly-tuned
# constrained<->unconstrained transform -- not necessarily an "extreme
# model") reaching a batch-local extrema computation could corrupt
# domain_hi/domain_lo to +-Inf permanently, since widening only ever grows
# it and never re-validates -- which then fed a degenerate span into
# StatsBase.histrange downstream (eps(Inf) is NaN, which silently defeated
# the degenerate-domain guard meant to catch exactly this). Recomputing
# fresh from the authoritative sample data every time removes that state
# entirely: there is nothing for a single bad value to corrupt beyond its
# own recompute, since non-finite values are filtered out fresh on every
# call rather than needing to be caught once and remembered correctly
# forever.
#
# `prior_lo`/`prior_hi` (the one-time, never-mutated prior-based estimate
# from _estimate_prior_domain -- see init_visualizer!) are unioned in as a
# floor/ceiling so the domain still only ever grows over a run, exactly as
# the old incremental version guaranteed, rather than the axis limits
# shrinking-then-growing as the real accumulated sample range happens to be
# narrower than that initial guess early in a run.
#
# Cost: O(total samples so far) per flush instead of O(batch size) -- an
# explicit, accepted trade for removing the incremental state entirely.
# Sample counts in this live-plotting context are small enough (typically
# thousands, not millions) for this to be unmeasurable in practice.
# PURE function (no graph access) so the caller can fold the recomputed
# domain into the same single batched update! as samples/current_idxs -- an
# earlier version updated the graph itself as a separate update! call, which
# (because ComputePipeline eagerly resolves observable-attached nodes like
# :gridlayout inside every update!, under the graph lock) made every flush
# pay a third full-grid resolve just for the domain. No "did it change" check
# is needed here either: ComputePipeline's input-side is_same uses isequal
# for distinct arrays, so handing update! a freshly-allocated but
# value-identical domain vector is dropped by the framework as a complete
# no-op for that input.
function _domain_including(all_samples, prior_lo::Vector{Float64}, prior_hi::Vector{Float64})
    data = all_samples.v.data
    new_lo = similar(prior_lo)
    new_hi = similar(prior_hi)
    for d in eachindex(prior_lo)
        row = view(data, d, :)
        finite_row = Iterators.filter(isfinite, row)
        if isempty(finite_row)
            new_lo[d] = prior_lo[d]
            new_hi[d] = prior_hi[d]
        else
            lo, hi = extrema(finite_row)
            new_lo[d] = min(lo, prior_lo[d])
            new_hi[d] = max(hi, prior_hi[d])
        end
    end
    return new_lo, new_hi
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

    # Window start (per-walker end position is current_idxs above, already
    # existing) -- the "Current Index" slider used to always reveal samples
    # from position 1 up to its single value; it's now an IntervalSlider, so
    # this is the low end of that interval. Defaults to 1 ("from the
    # beginning", i.e. exactly the old behavior) and is never touched at all
    # outside of that slider's own callback -- every other caller (live
    # multi-chain runs with no slider shown, precompilation, etc.) sees
    # identical behavior to before this was added.
    add_input!(graph, :window_start, 1)

    register_computation!(graph,
        [:samples, :current_idxs, :window_start],
        [:flat_samples],
    ) do inputs, changed, cached
        samples = inputs.samples
        current_idxs = inputs.current_idxs
        window_start = inputs.window_start

        walker_views = Any[] #Vector{DensitySampleVector}()
        for i in eachindex(samples)
            for j in eachindex(samples[i])
                wend = current_idxs[i][j]
                # clamp: a shorter walker (e.g. early in a live run, before
                # every walker has produced equally many samples) must never
                # see a start position past its own end -- 1:0-style empty
                # ranges are fine (a valid empty UnitRange), a start > end
                # the other way (e.g. 5:3) is not.
                wstart = clamp(window_start, 1, max(wend, 1))
                push!(walker_views, view(samples[i][j], wstart:wend))
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
    map!(
        # Per-sample step number, for Trace2D -- see :flat_chainids' comment
        # for the general availability-degradation pattern. `stepno` is the
        # step at which a stored row's position was *first* reached (frozen
        # at acceptance, not advanced by later rejections at that same spot)
        # -- Trace2D reconstructs each row's true *last* occupied step as
        # `stepno + weight - 1` itself, this just supplies the raw field.
        (smpls, idx) -> hasfield(eltype(smpls.info), :stepno) ? Int64[s.stepno for s in view(smpls.info, 1:idx)] : Int64[],
        graph,
        [:flat_samples, :current_idx],
        :flat_stepnos
    )
    map!(
        # Per-sample walker id, for Trace2D -- needed alongside chainid since
        # a single chain can have multiple concurrent walkers stepping in
        # lock-step (sharing overlapping stepno ranges), so chainid alone
        # can't disentangle one walker's own trajectory. Degrades to an
        # empty Int32[] whenever chainid/stepno themselves aren't available
        # (same condition as :flat_chainids/:flat_stepnos, kept consistent
        # so all three always agree on whether trace support exists and stay
        # the same length when it does) -- and, only for samples that *do*
        # have chain identity but happen not to expose a walker id under
        # either of the two field names BAT's own SampleID subtypes use
        # (`walkerid` for MCMCSampleID, `walker` for AHMCSampleID -- a
        # pre-existing naming inconsistency between the two, not introduced
        # here), falls back to an all-zeros vector (i.e. "assume one walker
        # per chain") rather than erroring.
        (smpls, idx) -> begin
            T = eltype(smpls.info)
            if !hasfield(T, :chainid) || !hasfield(T, :stepno)
                Int32[]
            elseif hasfield(T, :walkerid)
                Int32[s.walkerid for s in view(smpls.info, 1:idx)]
            elseif hasfield(T, :walker)
                Int32[s.walker for s in view(smpls.info, 1:idx)]
            else
                zeros(Int32, idx)
            end
        end,
        graph,
        [:flat_samples, :current_idx],
        :flat_walkerids
    )

    # Untruncated (full-completed-dataset) counterparts of :flat_samples/
    # :flat_weights/:flat_chainids/:flat_stepnos/:flat_walkerids above, used
    # only by Trace2D (see its own registration below) so it can reveal every
    # chain *proportionally* as :current_idx pans back and forth, instead of
    # via one shared row-position cutoff into the flattened array.
    #
    # This matters specifically for the static bat_makie_plot/Makie.plot path
    # reviewing a *completed* multi-chain run: BAT's merged multi-chain
    # DensitySampleVector is chain-block-concatenated (all of chain A's rows,
    # then all of chain B's, ...), not time-interleaved across chains --
    # confirmed empirically (only 3 chainid transitions across 515 row-pairs
    # for a 4-chain run). A single shared :current_idx cutoff into that
    # array therefore reveals one chain's *entire* block before any of the
    # next chain's, so panning the "Current Index" slider makes
    # already-fully-revealed chains appear to freeze while only the
    # currently-being-revealed chain's trace visibly moves -- a real,
    # reported symptom, but not a bug in Trace2D's own compute/caching
    # (verified directly: a frozen chain's own row count is mathematically
    # identical at every :current_idx past its block's end, and the
    # currently-active chain's own row count *does* change over the same
    # span, ruling out a stale-recompute bug). Reinterpreting :current_idx as
    # a *fraction* of the full dataset (current_idx / length(full)) and
    # applying that same fraction to each chain's own full-length group
    # (computed here) fixes this: every chain now reveals its own history in
    # lockstep, proportionally, regardless of which block it occupies.
    #
    # A no-op for the live path: current_idx there is always exactly
    # length(:flat_samples) (see the current_idx map! above) -- either
    # because show_slider is only ever enabled for a single (chain, walker)
    # (live single-chain or the static path, where "proportional" and "raw"
    # reveal are identical for the one existing group), or because
    # current_idxs is never manually rewound below each walker's own true
    # current length in live multi-chain runs (no slider is shown there at
    # all) -- so the reveal fraction computed from these nodes is always 1.0
    # exactly when it would otherwise matter.
    # The trace toggles are declared HERE, not with the other show_* control
    # inputs further down, because :flat_samples_full's registration directly
    # below lists them as inputs and register_computation! requires its
    # inputs to already exist. (Trace2D has no diagonal counterpart -- it's
    # an inherently 2D concept, a path through a 2D marginal -- so there's
    # no show_trace_diag.)
    add_input!(graph, :show_trace_upper, false)
    add_input!(graph, :show_trace_lower, false)

    # Gated on the trace toggles: this whole _full node family (this node,
    # the four per-sample map!s below, and the per-pair marg_full views)
    # exists ONLY for Trace2D, yet -- because ComputePipeline resolves a
    # node's inputs before its callback can early-return -- it used to pay a
    # full O(total samples) dataset copy plus several O(n) per-sample
    # comprehensions on every flush even with both trace toggles off (the
    # default). With the toggles as declared inputs, the off state produces
    # 1:0 views instead: same view/vcat types as the on state (so the
    # TypedEdge-locked output type is identical across off->on), just empty
    # -- and every downstream _full node is O(input length), so empty-in/
    # empty-out with no changes needed there. Consecutive off-state empties
    # are isequal, so ComputePipeline stops even running the downstream
    # callbacks after the first off-resolve. Flipping a toggle on dirties
    # this node -> full recompute -> the trace renders exactly as before.
    register_computation!(graph,
        [:samples, :show_trace_upper, :show_trace_lower],
        [:flat_samples_full],
    ) do inputs, changed, cached
        samples = inputs.samples
        trace_on = inputs.show_trace_upper || inputs.show_trace_lower
        walker_views = Any[]
        for i in eachindex(samples)
            for j in eachindex(samples[i])
                rng = trace_on ? (1:length(samples[i][j])) : (1:0)
                push!(walker_views, view(samples[i][j], rng))
            end
        end
        return (vcat(walker_views...),)
    end
    map!(
        smpls -> view(smpls.weight, 1:length(smpls)),
        graph,
        :flat_samples_full,
        :flat_weights_full
    )
    map!(
        smpls -> hasfield(eltype(smpls.info), :chainid) ? Int32[s.chainid for s in view(smpls.info, 1:length(smpls))] : Int32[],
        graph,
        :flat_samples_full,
        :flat_chainids_full
    )
    map!(
        smpls -> hasfield(eltype(smpls.info), :stepno) ? Int64[s.stepno for s in view(smpls.info, 1:length(smpls))] : Int64[],
        graph,
        :flat_samples_full,
        :flat_stepnos_full
    )
    map!(
        smpls -> begin
            T = eltype(smpls.info)
            # Named n_smpls, deliberately NOT n: this closure lives in the same
            # top-level scope as _init_compute_graph's own `n` (the grid size)
            # parameter, which is captured by many other closures registered
            # in this function (vsel_map, live_map, ...). An anonymous
            # function's own local assignment to a name doesn't get its own
            # independent binding here -- Julia treats `n = ...` inside this
            # lambda as *assigning to the same captured variable* every other
            # closure in this scope shares (confirmed directly: a minimal
            # reproduction of exactly this shape showed every other closure's
            # `n` change to match this lambda's last-assigned value after a
            # single call). That's exactly the mechanism behind this session's
            # real, reproducible bug: every time this computation re-ran (on
            # every new sample batch), it silently overwrote the shared grid
            # size with the current sample count, so soon after the first live
            # sample batch, every other computation's "n" (E.g. vsel_map's
            # `idxs has $(length(idxs)) entries, exceeding the grid size
            # N_max=$n` assertion) started reading a stale, wrong value
            # instead of the true, constant grid size.
            n_smpls = length(smpls)
            if !hasfield(T, :chainid) || !hasfield(T, :stepno)
                Int32[]
            elseif hasfield(T, :walkerid)
                Int32[s.walkerid for s in view(smpls.info, 1:n_smpls)]
            elseif hasfield(T, :walker)
                Int32[s.walker for s in view(smpls.info, 1:n_smpls)]
            else
                zeros(Int32, n_smpls)
            end
        end,
        graph,
        :flat_samples_full,
        :flat_walkerids_full
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

    # (:show_trace_upper/:show_trace_lower are declared earlier, above
    # :flat_samples_full's registration, which needs them as inputs.)

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
    # Trace2D is likewise not in BAT_MAKIE_RECIPES_2D -- it's an always-live
    # overlay (like Mean2D/Std2D/Cov2D, via its own determine_recipe_status
    # override in makie_trace.jl), not a selectable main recipe, and needs
    # the same extra chain-identity inputs ChainScatter2D does, plus
    # :flat_stepnos/:flat_walkerids.
    add_input!(graph, Symbol("$(Trace2D)"), Trace2D())

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
                # 0.05*(hi[v]-lo[v]) is exactly 0 when hi[v]==lo[v] (a
                # variable that's fixed/frozen, or hasn't moved yet in a live
                # run) -- confirmed directly that this renders the whole
                # panel completely blank (a literally zero-width Axis limits
                # argument), silently, with no error. Falling back to an
                # absolute margin scaled to the value's own magnitude when
                # the span is zero keeps the panel showing something.
                span = hi[v] - lo[v]
                margin = iszero(span) ? max(abs(hi[v]), one(hi[v])) * 0.05 : 0.05 * span
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
                [marg_sym, :flat_weights, :diagonal_recipe, :live_map, :diagonal_config, :vsel_map, :domain_lo, :domain_hi, :window_start],
                [primitive_symbols_1D[k]]
            ) do inputs, changed, cached
                coords, weights, live_recipe, live_map, config, vsel_map, domain_lo, domain_hi, window_start = inputs
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
                            _update_hist!(running_state, vec(coords), weights, vsel, domain, eff_nbins, closed, window_start)
                            compute_hist_primitives(recipe, running_state, config)
                        end
                    else
                        _update_stats!(running_state, vec(coords), weights, vsel, window_start)
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
            # Untruncated counterpart, for Trace2D only -- see
            # :flat_samples_full's comment above.
            marg_sym_2D_full = marg_full_symbol((j, i))
            map!(
                (smpls, vsel_map, live_map) -> live_map[j, i] ? view(smpls.v.data, [vsel_map[j, i]...], 1:length(smpls)) : view(smpls.v.data, Int[], 1:0),
                graph,
                [:flat_samples_full, :vsel_map, :live_map],
                marg_sym_2D_full
            )

            primitive_symbols_2D = [primitive_symbol(recipe, (j, i)) for recipe in BAT_MAKIE_RECIPES_2D]

            for k in eachindex(primitive_symbols_2D)
                recipe = BAT_MAKIE_RECIPES_2D[k]
                # Per-(cell, recipe) persistent accumulator for the incremental
                # 2D recipes -- see _make_running_state_2d and the 1D case above.
                running_state = is_incremental(recipe) ? _make_running_state_2d(recipe) : nothing

                register_computation!(graph,
                    [marg_sym_2D, :flat_weights, :upper_recipe, :lower_recipe, :live_map, :triagonal_config, :vsel_map, :domain_lo, :domain_hi, :window_start],
                    [primitive_symbols_2D[k]]
                ) do inputs, changed, cached
                    coords, weights, live_recipe_upper, live_recipe_lower, live_map, config, vsel_map, domain_lo, domain_hi, window_start = inputs
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
                                _update_hist!(running_state, x, y, weights, vsel, domain, nbins, closed, window_start)
                                compute_hist_primitives(recipe, running_state, config)
                            end
                        else
                            _update_stats!(running_state, coords, weights, vsel, window_start)
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

            # Trace2D registered separately, the same way ChainScatter2D is,
            # since it needs the same extra per-sample chain-identity input
            # plus :flat_stepnos/:flat_walkerids on top. Unlike ChainScatter2D
            # (a selectable main recipe, live only when actually chosen),
            # Trace2D's *availability* is always-live regardless of what
            # upper_recipe/lower_recipe currently is -- see its
            # determine_recipe_status override in makie_trace.jl, exactly
            # like Mean2D/Std2D/Cov2D. Whether it's actually *drawn* is a
            # separate, purely presentation-layer decision made in
            # _init_gridlayout's lift below (show_trace_upper/lower),
            # matching how the stats overlay toggle works.
            #
            # BUT unlike Mean2D/Std2D/Cov2D (whose computation is cheap
            # running-statistics accumulation, negligible whether the toggle
            # is on or not), Trace2D's own compute (grouping + windowing over
            # the *entire* accumulated dataset -- see its own file's header
            # comment) is real, non-incremental work. Explicitly threading
            # :show_trace_upper/:show_trace_lower in and early-returning the
            # empty sentinel when both are off (below) means this cost is
            # only ever paid while the overlay is actually visible, rather
            # than on every single sample batch for every active variable
            # pair regardless of whether any user ever touches this toggle.
            #
            # Uses the *_full inputs (untruncated) plus :current_idx directly
            # (rather than the shared current_idx-truncated marg_sym_2D/
            # flat_weights/flat_chainids/etc every other 2D recipe depends on)
            # so it can reveal each chain proportionally as current_idx pans
            # -- see :flat_samples_full's comment above for why a raw shared
            # truncation makes already-revealed chains appear to freeze.
            trace_primitive_sym = primitive_symbol(Trace2D(), (j, i))
            register_computation!(graph,
                [marg_sym_2D_full, :flat_weights_full, :flat_chainids_full, :flat_walkerids_full, :flat_stepnos_full, :current_idx, :upper_recipe, :lower_recipe, :live_map, :triagonal_config, :show_trace_upper, :show_trace_lower],
                [trace_primitive_sym]
            ) do inputs, changed, cached
                coords, weights, chainids, walkerids, stepnos, current_idx, live_recipe_upper, live_recipe_lower, live_map, config, show_trace_upper, show_trace_lower = inputs
                if !(show_trace_upper || show_trace_lower)
                    return (_empty_trace2d_primitives(),)
                end
                cell_status = live_map[i, j] ? LiveCell() : DeadCell()
                recipe_status = determine_recipe_status(Trace2D(), live_recipe_upper(), live_recipe_lower())
                primitives = compute_plotting_primitives(coords, weights, chainids, walkerids, stepnos, current_idx, Trace2D(), recipe_status, cell_status, config)
                return (primitives,)
            end
        end
    end

    return graph
end
