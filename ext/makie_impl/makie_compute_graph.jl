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

# The two grid indices are separated by an underscore ("_prim_12_3", not
# "_prim_123"): concatenating bare digits makes distinct index pairs
# ambiguous once indices reach two digits -- no colliding pair is actually
# generated under the current (bigger, smaller)/diagonal conventions, but
# that safety was accidental, not structural.
function primitive_symbol(recipe, vsel::Tuple{Int64,Int64})
    return Symbol(string(recipe), "_prim_", vsel[1], "_", vsel[2])
end

function primitive_symbol(recipe::R, vsel::Tuple{Int64,Int64}) where {R<:BATMakieRecipe}
    return Symbol(string(typeof(recipe)), "_prim_", vsel[1], "_", vsel[2])
end

function marg_symbol(vsel::Tuple{Int64,Int64})
    return Symbol("marg_$(vsel[1])_$(vsel[2])")
end

# CLOSURE-CAPTURE HAZARD, learned the hard way (a once-real, reproducible
# bug): every closure registered inside _init_compute_graph below shares that
# function's top-level scope, including its `n` (grid size) parameter. An
# assignment like `n = length(...)` inside any of those lambdas does NOT
# create an independent local -- it reassigns the shared captured variable
# every other closure reads (confirmed via a minimal repro; the historical
# symptom was vsel_map's grid-size check reading back the sample count).
# When adding closures there, never assign to a bare name that exists in the
# enclosing scope -- use a distinct local name.

# Rows of `walker` (restricted to 1:wend, the rows the live path has already
# revealed) whose DWELL intersects the real-MCMC-step window [wlo, whi]. A
# stored row with step number s and weight w occupies steps s : s+w-1, so it
# belongs to the window iff s <= whi and s+w-1 >= wlo -- including rows whose
# dwell merely straddles a window edge, with their FULL weight (the mass
# error is at most one dwell per walker per edge, an accepted trade against
# rebuilding the weight arrays on every slider drag). Both s and s+w-1 are
# monotone within one walker (each MCMC step either appends a row or
# increments the last row's weight; nonzero-weight row dropping only creates
# gaps, never reordering), so the window is a single contiguous row range,
# found by two binary searches. Samples without a stepno field (IID/Sobol --
# and AHMC, which has chain ids but no step numbers) fall back to "row index
# == step", reproducing the previous row-window semantics for those sources
# exactly. info elements are accessed one-by-one (info[k].stepno) since the
# live path's per-walker outputs store info as a plain Vector{MCMCSampleID},
# not a field-array StructVector.
function _step_window_rows(walker, wend::Integer, wlo::Integer, whi::Integer)
    wend <= 0 && return 1:0
    # Fast path for the untouched default window ("show everything") -- the
    # common case, hit on every live flush while the slider is untouched.
    (wlo <= 1 && whi == typemax(Int)) && return 1:wend

    info = walker.info
    if !isempty(info) && hasfield(eltype(info), :stepno)
        wt = walker.weight
        # First row whose dwell end (stepno + weight - 1) reaches wlo:
        lo, h = 1, wend + 1
        while lo < h
            m = (lo + h) >> 1
            if Int(info[m].stepno) + Int(wt[m]) - 1 < wlo
                lo = m + 1
            else
                h = m
            end
        end
        # Last row whose stepno is still within the window:
        l2, h2, hi = 1, wend, 0
        while l2 <= h2
            m = (l2 + h2) >> 1
            if Int(info[m].stepno) <= whi
                hi = m
                l2 = m + 1
            else
                h2 = m - 1
            end
        end
        return lo:hi
    else
        return clamp(wlo, 1, wend + 1):min(whi, wend)
    end
end

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
#
# `target` must be the ORIGINAL, untransformed measure (bat_sample's own `m`,
# threaded through init_visualizer!'s `target` kwarg) -- NOT
# mcmc_target(mcmc_states[1]), which an earlier version used: mcmc states
# carry the PRETRANSFORMED measure (default pretransform for RandomWalk/MALA
# is PriorToNormal), while the displayed samples go through
# inverse(f_pretransform) into original space, so a prior domain estimated
# from the transformed target's initsrc was standard-normal-scaled regardless
# of the real prior's scale -- e.g. a prior at 100 +- 5 got a domain floor
# near -3, silently inflating every axis until real samples widened past it.
function _estimate_prior_domain(target, n_dof::Integer; n_prior_samples::Integer=2000, tail_prob::Real=0.0015)
    initsrc = BAT.get_initsrc_from_target(target)
    shape = varshape(initsrc)
    draws = [ValueShapes.unshaped(rand(initsrc), shape) for _ in 1:n_prior_samples]
    M = reduce(hcat, draws)
    lo = [quantile(view(M, d, :), tail_prob) for d in 1:n_dof]
    hi = [quantile(view(M, d, :), 1 - tail_prob) for d in 1:n_dof]
    return lo, hi
end

# Per-dimension HARD support bounds of a measure's prior, in the unshaped
# original-space coordinates the visualizer displays -- (+-Inf for anything
# unbounded or undeterminable). Used to drive KDE boundary reflection (see
# _weighted_kde1d/_weighted_kde2d in makie_kde.jl): without it, a KDE of a
# bounded parameter (e.g. a Uniform prior) leaks half its kernel mass past
# each hard bound, rendering the density at the boundary at roughly half its
# true value. Every fallback here is CONSERVATIVE (+-Inf = "unknown" = no
# correction, exactly the pre-existing behavior), so an exotic prior this
# walk doesn't understand degrades to today's uncorrected rendering, never
# to a wrong correction.
function _support_bounds(target, n_dof::Integer)
    d = _support_root_dist(target)
    isnothing(d) && return (fill(-Inf, n_dof), fill(Inf, n_dof))
    lo, hi = _component_support(d)
    # A shape mismatch means the walk didn't line up with the displayed
    # unshaped dims -- don't guess, disable correction outright.
    length(lo) == n_dof || return (fill(-Inf, n_dof), fill(Inf, n_dof))
    return lo, hi
end

# Unwraps a measure down to the underlying prior Distribution the support can
# be read from. Mirrors get_initsrc_from_target's unwrap chain (posterior ->
# prior), but lands on the raw Distribution instead of a measure, since
# per-component bounds need Distributions.minimum/maximum on the components.
_support_root_dist(m::AbstractPosteriorMeasure) = _support_root_dist(getprior(m))
_support_root_dist(m::BATDistMeasure) = m.dist
_support_root_dist(m::BATWeightedMeasure) = _support_root_dist(m.base)
_support_root_dist(d::Distribution) = d
_support_root_dist(@nospecialize(m)) = nothing

# Per-component (lo, hi) vectors covering that component's unshaped dims.
# The NamedTupleDist walk (components x accessors x view_idxs) follows
# truncate_dist_hard's own walk over the same structure
# (src/measures/truncate_batmeasure.jl) -- the accessors' view_idxs ARE the
# unshaped-dim layout, so bounds land at exactly the dims the displayed
# flat samples use.
_component_support(d::UnivariateDistribution) =
    (Float64[minimum(d)], Float64[maximum(d)])
_component_support(d::Distributions.Product) =
    (Float64[minimum(c) for c in d.v], Float64[maximum(c) for c in d.v])
_component_support(d::ValueShapes.UnshapedNTD) = _component_support(d.shaped)
# ConstValueDist occupies zero unshaped dims -- contributes nothing.
_component_support(::ConstValueDist) = (Float64[], Float64[])
# Conservative fallback for anything without a per-dim support notion
# (correlated multivariates, matrix-variates, ...): unbounded everywhere.
_component_support(d::Distribution) = (fill(-Inf, length(d)), fill(Inf, length(d)))

function _component_support(d::NamedTupleDist)
    n = totalndof(varshape(d))
    lo = fill(-Inf, n)
    hi = fill(Inf, n)
    dists = values(d)
    accessors = values(varshape(d))
    for (dd, acc) in zip(dists, accessors)
        # view_idxs returns a bare Int for scalar accessors (a range only
        # for array-shaped ones) -- normalize so the assignment below is
        # uniformly vector-shaped.
        raw_idxs = ValueShapes.view_idxs(1:n, acc)
        idxs = raw_idxs isa Integer ? (raw_idxs:raw_idxs) : raw_idxs
        clo, chi = _component_support(dd)
        # A component whose own walk came back mismatched (a nested exotic
        # dist) keeps its dims at the +-Inf default instead of corrupting
        # neighboring components' slots.
        length(clo) == length(idxs) || continue
        lo[idxs] .= clo
        hi[idxs] .= chi
    end
    return (lo, hi)
end

# Normalizes the static path's user-facing `support` kwarg (bat_makie_plot/
# Makie.plot) into the graph's (support_lo, support_hi) vectors: `nothing`
# (the default) -> empty = unknown = no correction; a vector of per-dim
# (lo, hi) pairs (tuples or intervals) -> validated directly; anything else
# (a posterior/prior measure or a Distribution) -> derived via
# _support_bounds. Explicit pairs are validated loudly -- unlike the
# measure-derived path, a hand-written bounds list that doesn't fit the
# model is a caller mistake, not an exotic prior to degrade gracefully on.
_support_vectors(::Nothing, n_dof::Integer) = (Float64[], Float64[])

function _support_vectors(support::AbstractVector, n_dof::Integer)
    length(support) == n_dof || throw(ArgumentError(
        "`support` has $(length(support)) entries but the model has $n_dof dimensions"))
    lo = Float64[first(s) for s in support]
    hi = Float64[last(s) for s in support]
    for d in 1:n_dof
        lo[d] <= hi[d] || throw(ArgumentError(
            "`support` entry $d has lo > hi ($(lo[d]) > $(hi[d]))"))
    end
    return lo, hi
end

_support_vectors(support, n_dof::Integer) = _support_bounds(support, n_dof)

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
        chain_ids=Vector{Int32}(),  # MCMCChainStateInfo ids are Int32
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

    curr_idxs = Vector{Vector{Int}}()
    add_input!(graph, :current_idxs, curr_idxs)

    # The display window in REAL MCMC STEPS (the "Step Range" slider's
    # interval), applied per walker via _step_window_rows: step numbers are
    # the one clock all walkers of a chain share (they step in lock-step but
    # store different row counts, so any row-based window would misalign
    # them in time). Written ONLY by the slider; the default sentinel means
    # "everything" and hits _step_window_rows' fast path, so live flushes
    # pay no search cost while the slider is untouched. :current_idxs above
    # stays the "rows that exist per walker" bookkeeping, written only by
    # registration/flush -- the slider and the flush no longer fight over a
    # shared input as they did when the slider wrote current_idxs directly.
    add_input!(graph, :window_steps, (1, typemax(Int)))

    register_computation!(graph,
        [:samples, :current_idxs, :window_steps],
        [:flat_samples],
    ) do inputs, changed, cached
        samples = inputs.samples
        current_idxs = inputs.current_idxs
        wlo, whi = inputs.window_steps

        walker_views = Any[] #Vector{DensitySampleVector}()
        for i in eachindex(samples)
            for j in eachindex(samples[i])
                walker = samples[i][j]
                # min(...): the graph's row bookkeeping can lag the walker's
                # true length mid-flush; never view past the real end.
                wend = min(current_idxs[i][j], length(walker))
                push!(walker_views, view(walker, _step_window_rows(walker, wend, wlo, whi)))
            end
        end
        return (vcat(walker_views...),)
    end

    # Highest real MCMC step any walker has reached so far (its last row's
    # stepno + weight - 1; the row count for sources without step numbers)
    # -- the "Step Range" slider's range end.
    register_computation!(graph,
        [:samples, :current_idxs],
        [:max_step],
    ) do inputs, changed, cached
        samples = inputs.samples
        current_idxs = inputs.current_idxs
        max_step = 0
        for i in eachindex(samples)
            for j in eachindex(samples[i])
                walker = samples[i][j]
                wend = min(current_idxs[i][j], length(walker))
                wend <= 0 && continue
                info = walker.info
                if !isempty(info) && hasfield(eltype(info), :stepno)
                    max_step = max(max_step, Int(info[wend].stepno) + Int(walker.weight[wend]) - 1)
                else
                    max_step = max(max_step, wend)
                end
            end
        end
        return (max_step,)
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


    add_input!(graph, :idxs, Int[])

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
        # A real throw, not @assert: this guards user-influencable state and
        # asserts may be compiled out.
        length(idxs) <= n || throw(ArgumentError("idxs has $(length(idxs)) entries, exceeding the grid size N_max=$n"))

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
        length(idxs) <= n || throw(ArgumentError("idxs has $(length(idxs)) entries, exceeding the grid size N_max=$n"))

        # Symmetric square block by construction -- consumers uniformly index
        # it as live_map[j, i], matching the (bigger, smaller) pair convention
        # of vsel_map/the marginal views. If this construction ever becomes
        # non-symmetric, every consumer's indexing must be revisited.
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

    # Trace2D has no diagonal counterpart -- it's an inherently 2D concept
    # (a path through a 2D marginal), so there's no show_trace_diag.
    add_input!(graph, :show_trace_upper, false)
    add_input!(graph, :show_trace_lower, false)

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

    # Per-real-dimension HARD prior-support bounds (+-Inf where unbounded/
    # unknown -- see _support_bounds), set once by the same callers that set
    # the domain. Distinct from domain_lo/domain_hi: the domain is a soft
    # display range (data extrema / prior quantiles) that grows over a live
    # run, while these are the fixed truncation boundaries of the measure
    # itself, used for KDE boundary reflection. Empty = not provided = no
    # correction anywhere.
    add_input!(graph, :support_lo, Float64[])
    add_input!(graph, :support_hi, Float64[])

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

        # Every recipe is a FULL recompute over the current sample view on
        # every invocation -- the per-cell incremental accumulators
        # (_IncrementalHist*/Uv/MvState and their n/vsel/domain/wstart
        # staleness protocol) were deliberately removed: the flush already
        # pays O(total samples) data assembly regardless, so the accumulators
        # only saved the (cheap) fold while forcing every incremental recipe
        # to compute even when not selected, duplicating identical
        # histogram/moment folds across sibling recipes, and repeatedly
        # producing staleness bugs (the zero-weight crashes and the
        # interval-slider pan silently showing stale data were both
        # accumulator-state bugs). Now only recipes that actually resolve to
        # LiveRecipe do real work -- switching recipes costs one O(n)
        # recompute of the newly selected recipe, milliseconds at this
        # extension's scales -- and there is no persistent state to go stale.
        for k in eachindex(primitive_symbols_1D)
            recipe = BAT_MAKIE_RECIPES_1D[k]
            register_computation!(graph,
                [marg_sym, :flat_weights, :diagonal_recipe, :live_map, :diagonal_config, :vsel_map, :domain_lo, :domain_hi, :support_lo, :support_hi],
                [primitive_symbols_1D[k]]
            ) do inputs, changed, cached
                # Field access by name, not positional destructuring -- a
                # positional unpack silently mis-binds if the input-symbol
                # list above is ever reordered. The marginal view's key is
                # per-cell, so it's read via getproperty.
                coords = getproperty(inputs, marg_sym)
                weights = inputs.flat_weights
                config = inputs.diagonal_config
                (; live_map, vsel_map, domain_lo, domain_hi, support_lo, support_hi) = inputs
                cell_status = live_map[i, i] ? LiveCell() : DeadCell()
                recipe_status = determine_recipe_status(recipe, inputs.diagonal_recipe())
                # The cell's fixed per-variable domain rides along inside the
                # config -- only the histogram-family recipes read it, for
                # STABLE domain-derived bin edges (see _marginal_view_dist's
                # domain parameter). Skipped when the domain isn't known yet
                # or the cell has no active variable; those recipes then fall
                # back to data-derived edges.
                v = vsel_map[i, i][1]
                cfg = (v == 0 || isempty(domain_lo)) ? config :
                    (; config..., domain=(domain_lo[v], domain_hi[v]))
                # The variable's hard support bounds ride along the same way
                # -- only the KDE-family recipes read these, for boundary
                # reflection (see _weighted_kde1d).
                cfg = (v == 0 || isempty(support_lo)) ? cfg :
                    (; cfg..., support=(support_lo[v], support_hi[v]))
                return (compute_plotting_primitives(coords, weights, recipe, recipe_status, cell_status, cfg),)
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
                # Full recompute per invocation, no persistent accumulator --
                # see the 1D loop's comment above for the rationale.
                register_computation!(graph,
                    [marg_sym_2D, :flat_weights, :upper_recipe, :lower_recipe, :live_map, :triagonal_config, :vsel_map, :domain_lo, :domain_hi, :support_lo, :support_hi],
                    [primitive_symbols_2D[k]]
                ) do inputs, changed, cached
                    # By-name access -- see the 1D loop's matching comment.
                    coords = getproperty(inputs, marg_sym_2D)
                    weights = inputs.flat_weights
                    config = inputs.triagonal_config
                    (; upper_recipe, lower_recipe, live_map, vsel_map, domain_lo, domain_hi, support_lo, support_hi) = inputs
                    cell_status = live_map[j, i] ? LiveCell() : DeadCell()
                    recipe_status = determine_recipe_status(recipe, upper_recipe(), lower_recipe())
                    # See the 1D loop above -- the pair's fixed domain rides
                    # along inside the config for the histogram recipes' bin
                    # edges, and the pair's hard support bounds for the KDE
                    # recipes' boundary reflection. Tuple order matches the
                    # marginal view's row order (vsel[1] = x, vsel[2] = y).
                    vsel = vsel_map[j, i]
                    cfg = (vsel[1] == 0 || isempty(domain_lo)) ? config :
                        (; config..., domain=((domain_lo[vsel[1]], domain_hi[vsel[1]]), (domain_lo[vsel[2]], domain_hi[vsel[2]])))
                    cfg = (vsel[1] == 0 || isempty(support_lo)) ? cfg :
                        (; cfg..., support=((support_lo[vsel[1]], support_hi[vsel[1]]), (support_lo[vsel[2]], support_hi[vsel[2]])))
                    return (compute_plotting_primitives(coords, weights, recipe, recipe_status, cell_status, cfg),)
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
            # ChainScatter2D as the upper/lower recipe.
            chainscatter_primitive_sym = primitive_symbol(ChainScatter2D(), (j, i))
            register_computation!(graph,
                [marg_sym_2D, :flat_weights, :flat_chainids, :upper_recipe, :lower_recipe, :live_map, :triagonal_config],
                [chainscatter_primitive_sym]
            ) do inputs, changed, cached
                # By-name access -- see the 1D loop's matching comment.
                coords = getproperty(inputs, marg_sym_2D)
                (; flat_weights, flat_chainids, upper_recipe, lower_recipe, live_map, triagonal_config) = inputs
                cell_status = live_map[j, i] ? LiveCell() : DeadCell()
                recipe_status = determine_recipe_status(ChainScatter2D(), upper_recipe(), lower_recipe())
                primitives = compute_plotting_primitives(coords, flat_weights, flat_chainids, ChainScatter2D(), recipe_status, cell_status, triagonal_config)
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
            # _init_gridlayout (show_trace_upper/lower), matching how the
            # stats overlay toggle works -- but since Trace2D's own compute
            # (grouping + windowing) is real, non-incremental work, the
            # toggles are ALSO threaded in here so the sentinel early-return
            # below skips that cost entirely while the overlay is off.
            #
            # Uses the ordinary windowed per-sample inputs: with samples
            # registered per (chain, walker) and the step window applied per
            # walker (see :window_steps/_step_window_rows), every group's
            # rows here are its own chronological, time-aligned slice -- an
            # older design needed untruncated _full input copies plus a
            # proportional reveal fraction to work around the merged static
            # dataset being one chain-block-concatenated pseudo-walker; both
            # are gone.
            trace_primitive_sym = primitive_symbol(Trace2D(), (j, i))
            register_computation!(graph,
                [marg_sym_2D, :flat_weights, :flat_chainids, :flat_walkerids, :flat_stepnos, :upper_recipe, :lower_recipe, :live_map, :triagonal_config, :show_trace_upper, :show_trace_lower],
                [trace_primitive_sym]
            ) do inputs, changed, cached
                # By-name access -- see the 1D loop's matching comment.
                (; show_trace_upper, show_trace_lower) = inputs
                if !(show_trace_upper || show_trace_lower)
                    return (_empty_trace2d_primitives(),)
                end
                coords = getproperty(inputs, marg_sym_2D)
                (; flat_weights, flat_chainids, flat_walkerids, flat_stepnos, upper_recipe, lower_recipe, live_map, triagonal_config) = inputs
                cell_status = live_map[j, i] ? LiveCell() : DeadCell()
                recipe_status = determine_recipe_status(Trace2D(), upper_recipe(), lower_recipe())
                primitives = compute_plotting_primitives(coords, flat_weights, flat_chainids, flat_walkerids, flat_stepnos, Trace2D(), recipe_status, cell_status, triagonal_config)
                return (primitives,)
            end
        end
    end

    return graph
end
