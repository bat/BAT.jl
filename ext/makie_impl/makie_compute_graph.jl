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

# The underscore between the indices keeps multi-digit index pairs unambiguous.
function primitive_symbol(recipe, vsel::Tuple{Int64,Int64})
    return Symbol(string(recipe), "_prim_", vsel[1], "_", vsel[2])
end

function primitive_symbol(recipe::R, vsel::Tuple{Int64,Int64}) where {R<:BATMakieRecipe}
    return Symbol(string(typeof(recipe)), "_prim_", vsel[1], "_", vsel[2])
end

function marg_symbol(vsel::Tuple{Int64,Int64})
    return Symbol("marg_$(vsel[1])_$(vsel[2])")
end

# Closures registered in _init_compute_graph share its scope (including `n`):
# assigning to a bare enclosing-scope name in one reassigns the shared variable
# for all -- always use distinct local names.

# Initial per-dim axis/bin domain from prior-sample tail quantiles; not a hard
# bound. `target` must be the ORIGINAL untransformed measure (bat_sample's own
# `m`), NOT mcmc_target(...): mcmc states carry the PRETRANSFORMED measure,
# giving a standard-normal-scaled domain regardless of the real prior's scale.
function _estimate_prior_domain(rng::AbstractRNG, target, n_dof::Integer; n_prior_samples::Integer = 2000, tail_prob::Real = 0.0015)
    initsrc = BAT.get_initsrc_from_target(target)
    shape = varshape(initsrc)
    draws = [ValueShapes.unshaped(rand(rng, initsrc), shape) for _ in 1:n_prior_samples]
    M = reduce(hcat, draws)
    lo = [quantile(view(M, d, :), tail_prob) for d in 1:n_dof]
    hi = [quantile(view(M, d, :), 1 - tail_prob) for d in 1:n_dof]
    return lo, hi
end

# Per-dim HARD support bounds of the prior in unshaped display coords, driving
# KDE boundary reflection. Fallbacks are CONSERVATIVE: +-Inf = no correction.
function _support_bounds(target, n_dof::Integer)
    d = _support_root_dist(target)
    isnothing(d) && return (fill(-Inf, n_dof), fill(Inf, n_dof))
    lo, hi = _component_support(d)
    # Shape mismatch: don't guess, disable correction outright.
    length(lo) == n_dof || return (fill(-Inf, n_dof), fill(Inf, n_dof))
    return lo, hi
end

# Use the original measure's prior, never an EvaluatedMeasure's empirical or
# transformed approximation: support is expressed in displayed coordinates.
_support_root_dist(m::BAT.EvaluatedMeasure) = _support_root_dist(BAT.unevaluated(m))
_support_root_dist(m::AbstractPosteriorMeasure) = _support_root_dist(getprior(m))
_support_root_dist(m::BATDistMeasure) = m.dist
_support_root_dist(m::BATWeightedMeasure) = _support_root_dist(m.base)
_support_root_dist(d::Distribution) = d
_support_root_dist(@nospecialize(m)) = nothing

# Per-component (lo, hi) covering its unshaped dims; the NamedTupleDist walk
# mirrors truncate_dist_hard's, so bounds land at the displayed flat-sample dims.
function _component_support(d::UnivariateDistribution)
    lo, hi = _kde_support_bounds((minimum(d), maximum(d)))
    return (Float64[lo], Float64[hi])
end
function _component_support(d::Distributions.Product)
    bounds = [_kde_support_bounds((minimum(c), maximum(c))) for c in d.v]
    return (Float64[first(s) for s in bounds], Float64[last(s) for s in bounds])
end
_component_support(d::ValueShapes.UnshapedNTD) = _component_support(d.shaped)
# ConstValueDist occupies zero unshaped dims -- contributes nothing.
_component_support(::ConstValueDist) = (Float64[], Float64[])
# Conservative fallback for dists without per-dim support: unbounded everywhere.
_component_support(d::Distribution) = (fill(-Inf, length(d)), fill(Inf, length(d)))

function _component_support(d::NamedTupleDist)
    n = totalndof(varshape(d))
    lo = fill(-Inf, n)
    hi = fill(Inf, n)
    dists = values(d)
    accessors = values(varshape(d))
    for (dd, acc) in zip(dists, accessors)
        # view_idxs returns a bare Int for scalar accessors -- normalize to a range.
        raw_idxs = ValueShapes.view_idxs(1:n, acc)
        idxs = raw_idxs isa Integer ? (raw_idxs:raw_idxs) : raw_idxs
        clo, chi = _component_support(dd)
        # A mismatched component walk keeps its dims at the +-Inf default.
        length(clo) == length(idxs) || continue
        lo[idxs] .= clo
        hi[idxs] .= chi
    end
    return (lo, hi)
end

# Normalizes the `support` kwarg into (support_lo, support_hi): nothing -> none;
# explicit pairs validated loudly (caller mistake); a measure -> _support_bounds.
_support_vectors(::Nothing, n_dof::Integer) = (Float64[], Float64[])

function _support_vectors(support::AbstractVector, n_dof::Integer)
    length(support) == n_dof || throw(ArgumentError(
        "`support` has $(length(support)) entries but the model has $n_dof dimensions"))
    any(isnothing, support) && throw(ArgumentError(
        "`support` entries must be (lo, hi) pairs; use (-Inf, Inf) for an unconstrained dimension"))
    bounds = map(_kde_support_bounds, support)
    lo = Float64[first(s) for s in bounds]
    hi = Float64[last(s) for s in bounds]
    return lo, hi
end

_support_vectors(support, n_dof::Integer) = _support_bounds(support, n_dof)

# Static-path domain from the true per-dim extrema. Non-finite values must be
# filtered: an Inf span defeats the eps-based degenerate-edges guard downstream
# (eps(Inf) is NaN).
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

# Scan each owned walker matrix without materializing the whole history. Start
# from the fixed prior each time: replacing pending samples can remove extrema.
# No graph access; the caller publishes these fresh bounds with the same samples.
function _domain_including(walkers, prior_lo::Vector{Float64}, prior_hi::Vector{Float64})
    lo, hi = copy(prior_lo), copy(prior_hi)
    for samples in walkers
        isempty(samples) && continue
        _include_finite_domain!(lo, hi, samples.v.data)
    end
    return lo, hi
end

# Function barrier for the type-erased walker containers in live publication.
function _include_finite_domain!(lo::Vector{Float64}, hi::Vector{Float64}, data::AbstractMatrix)
    for d in eachindex(lo)
        row = view(data, d, :)
        finite_row = Iterators.filter(isfinite, row)
        if !isempty(finite_row)
            sample_lo, sample_hi = extrema(finite_row)
            lo[d] = min(sample_lo, lo[d])
            hi[d] = max(sample_hi, hi[d])
        end
    end
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
        graph=Ref{Union{Nothing,ComputeGraph}}(graph),
        buffer_lock=buffer_lock,
        # Shares buffer_lock: waiters and notifiers must use one lock (Threads.Condition).
        buffer_cond=Threads.Condition(buffer_lock),
        chain_ids=Vector{Int32}(),  # MCMCChainStateInfo ids are Int32
        output_buffer=Vector{Vector{DensitySampleVector}}(),
        # Latest owned current-state snapshot per chain/walker, including dwell
        # not yet emitted by the sampler. Slots are replaced under buffer_lock.
        pending_samples=Vector{Vector{DensitySampleVector}}(),
        output_timing=SampleStepSpans(),
        pending_timing=SampleStepSpans(),
        step_mode=Ref(false),
        capture_init=Ref(false),
        init_segments=Dict{Tuple{Int32,Int32},Vector{InitStepSegment}}(),
        invalid_init=Set{Tuple{Int32,Int32}}(),
        n_buffer_steps=Ref(0),
        buffer_dirty=Ref(false),
        # Flush-trigger threshold (grows under adaptive_batching); the backpressure
        # ceiling derives from it, so the sampler never blocks on an unreachable trigger.
        effective_batch_size=Ref(vis.n_batch),
        buffer_ratio=vis.max_buffered / vis.n_batch,
        lifecycle=_MakieLiveControl(),
        # Set in init_visualizer!; read by _apply_vsel! to validate vsel changes.
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
    n = Int(n)
    n > 0 || throw(ArgumentError("N_max must be positive (got $n)"))
    graph = ComputeGraph()

    smpls = Any[]

    add_input!(graph, :samples, smpls)

    curr_idxs = Vector{Vector{Int}}()
    add_input!(graph, :current_idxs, curr_idxs)

    # Nothing means All, independently of the current data extent. The same
    # range selects local steps with timing, or stored rows without timing.
    add_input!((key, value) -> Ref{Union{Nothing,Tuple{Int64,Int64}}}(value), graph, :window_steps, nothing)
    add_input!((key, value) -> Ref{Union{Nothing,SampleStepSpans}}(value), graph, :sample_timing, nothing)
    register_computation!(graph, [:samples, :current_idxs, :sample_timing], [:window_data]) do inputs, changed, cached
        @nospecialize(changed, cached)
        return (_sample_window_data(inputs.samples, inputs.current_idxs, inputs.sample_timing),)
    end
    register_computation!(graph, [:window_data, :window_steps], [:flat_samples, :window_segments]) do inputs, changed, cached
        @nospecialize(changed, cached)
        return _window_selection(inputs.window_data, inputs.window_steps)
    end
    map!(data -> data.limit, graph, :window_data, :max_step)
    map!(data -> data.timed, graph, :window_data, :has_step_timing)

    map!(smpls -> length(smpls),
        graph,
        :flat_samples,
        :current_idx
    )
    map!(
        # View of the real weight vector, not a hardcoded-eltype placeholder.
        (smpls, idx) -> view(smpls.weight, 1:idx),
        graph,
        [:flat_samples, :current_idx],
        :flat_weights
    )
    map!(
        # Per-sample chain id for ChainScatter2D; Int32[] when info has no chain
        # identity, so the graph can resolve before availability is checked.
        (smpls, idx) -> _samples_have_chain_ids(smpls) ? Int32[s.chainid for s in view(smpls.info, 1:idx)] : Int32[],
        graph,
        [:flat_samples, :current_idx],
        :flat_chainids
    )
    add_input!(graph, :idxs, Int[])

    register_computation!(graph,
        [:idxs],
        [:vsel_map],
    ) do inputs, changed, cached
        @nospecialize(changed, cached)
        idxs = inputs.idxs
        # A real throw, not @assert: guards user-influencable state; asserts may be compiled out.
        length(idxs) <= n || throw(ArgumentError("idxs has $(length(idxs)) entries, exceeding the grid size N_max=$n"))

        n_active = length(idxs)
        vsel_map = Matrix{Tuple{Int,Int}}(undef, n, n)
        for i in 1:n, j in 1:n
            # (0, 0) sentinel beyond the selection -- a live_map bypass fails loudly (index 0).
            vsel_map[i, j] = (i <= n_active && j <= n_active) ? (idxs[i], idxs[j]) : (0, 0)
        end

        return (vsel_map,)
    end

    register_computation!(graph,
        [:idxs],
        [:live_map],
    ) do inputs, changed, cached
        @nospecialize(changed, cached)
        idxs = inputs.idxs
        length(idxs) <= n || throw(ArgumentError("idxs has $(length(idxs)) entries, exceeding the grid size N_max=$n"))

        # Symmetric by construction -- consumers uniformly index it as live_map[j, i].
        live_map = fill(false, n, n)

        n_active = length(idxs)
        for i in 1:n_active
            for j in 1:n_active
                live_map[i, j] = true
            end
        end
        return (live_map,)
    end

    add_input!(graph, :upper_recipe, recipes.upper)
    add_input!(graph, :diagonal_recipe, recipes.diagonal)
    add_input!(graph, :lower_recipe, recipes.lower)

    add_input!(graph, :show_stats_upper, false)
    add_input!(graph, :show_stats_diag, false)
    add_input!(graph, :show_stats_lower, false)

    # Trace2D is inherently 2D -- no show_trace_diag.
    add_input!(graph, :show_trace_upper, false)
    add_input!(graph, :show_trace_lower, false)

    add_input!(graph, :triagonal_config, triagonal_config)
    add_input!(graph, :diagonal_config, diagonal_config)

    # Fixed per-dim domain for axis limits and histogram bin edges; empty until
    # the caller sets it -- axis_limits_i guards the unset case.
    add_input!(graph, :domain_lo, Float64[])
    add_input!(graph, :domain_hi, Float64[])

    # Hard prior-support bounds for KDE boundary reflection (+-Inf = unknown);
    # fixed, unlike the growing display domain above. Empty = no correction.
    add_input!(graph, :support_lo, Float64[])
    add_input!(graph, :support_hi, Float64[])

    for recipe in vcat(BAT_MAKIE_RECIPES_1D, BAT_MAKIE_RECIPES_2D)
        add_input!(graph, Symbol("$(typeof(recipe))"), recipe)
    end
    # Registered separately below (extra inputs), but still needs a per-type input node.
    add_input!(graph, Symbol("$(ChainScatter2D)"), ChainScatter2D())
    # Trace2D likewise -- always-live overlay, registered separately below.
    add_input!(graph, Symbol("$(Trace2D)"), Trace2D())
    register_computation!(graph,
        [:window_segments, :triagonal_config, :show_trace_upper, :show_trace_lower], [:trace_groups]
    ) do inputs, changed, cached
        @nospecialize(changed, cached)
        return (_window_trace_groups(
            inputs.show_trace_upper || inputs.show_trace_lower ? inputs.window_segments : WindowStepGroup[],
            inputs.triagonal_config.trace_nsteps),)
    end


    for i in 1:n
        marg_sym = marg_symbol((i, i))

        map!(
            # Dead branch: empty view over the real smpls.v.data, not a hardcoded
            # placeholder -- live/dead views of this node must share a type.
            (smpls, vsel_map, current_idx, live_map) -> (current_idx > 0 && live_map[i, i]) ? view(smpls.v.data, [vsel_map[i, i][1]], 1:current_idx) : view(smpls.v.data, Int[], 1:0),
            graph,
            [:flat_samples, :vsel_map, :current_idx, :live_map],
            marg_sym
        )

        # Limits come from the fixed domain, not the visible data, keeping axes
        # stable; (0, 1) when the domain is unset or the cell has no variable.
        map!(
            (lo, hi, vsel_map) -> begin
                v = vsel_map[i, i][1]
                (isempty(lo) || v == 0) && return (0.0, 1.0)
                # Zero span would render a blank panel -- use a magnitude-scaled margin.
                span = hi[v] - lo[v]
                margin = iszero(span) ? max(abs(hi[v]), one(hi[v])) * 0.05 : 0.05 * span
                (lo[v] - margin, hi[v] + margin)
            end,
            graph,
            [:domain_lo, :domain_hi, :vsel_map],
            Symbol("axis_limits_$i")
        )

        primitive_symbols_1D = [primitive_symbol(recipe, (i, i)) for recipe in BAT_MAKIE_RECIPES_1D]

        # Full recompute per invocation -- deliberately no per-cell accumulator state.
        for k in eachindex(primitive_symbols_1D)
            recipe = BAT_MAKIE_RECIPES_1D[k]
            register_computation!(graph,
                [marg_sym, :flat_weights, :diagonal_recipe, :live_map, :diagonal_config, :vsel_map, :domain_lo, :domain_hi, :support_lo, :support_hi],
                [primitive_symbols_1D[k]]
            ) do inputs, changed, cached
                @nospecialize(changed, cached)
                # By-name access -- a positional unpack silently mis-binds if inputs are reordered.
                coords = getproperty(inputs, marg_sym)
                weights = inputs.flat_weights
                config = inputs.diagonal_config
                (; live_map, vsel_map, domain_lo, domain_hi, support_lo, support_hi) = inputs
                cell_status = live_map[i, i] ? LiveCell() : DeadCell()
                recipe_status = determine_recipe_status(recipe, inputs.diagonal_recipe())
                # The cell's fixed domain rides along in the config (histogram recipes' stable bin edges).
                v = vsel_map[i, i][1]
                cfg = (v == 0 || isempty(domain_lo)) ? config :
                    (; config..., domain=(domain_lo[v], domain_hi[v]))
                # The hard support bounds ride along the same way (KDE boundary reflection).
                cfg = (v == 0 || isempty(support_lo)) ? cfg :
                    (; cfg..., support=(support_lo[v], support_hi[v]))
                return (compute_plotting_primitives(coords, weights, recipe, recipe_status, cell_status, cfg),)
            end
        end

        for j in i+1:n
            marg_sym_2D = marg_symbol((j, i))
            map!(
                # Empty view over the real smpls.v.data -- see the 1D case above.
                (smpls, vsel_map, current_idx, live_map) -> (current_idx > 0 && live_map[j, i]) ? view(smpls.v.data, [vsel_map[j, i]...], 1:current_idx) : view(smpls.v.data, Int[], 1:0),
                graph,
                [:flat_samples, :vsel_map, :current_idx, :live_map],
                marg_sym_2D
            )
            primitive_symbols_2D = [primitive_symbol(recipe, (j, i)) for recipe in BAT_MAKIE_RECIPES_2D]

            for k in eachindex(primitive_symbols_2D)
                recipe = BAT_MAKIE_RECIPES_2D[k]
                # Full recompute, no persistent accumulator -- see the 1D loop.
                register_computation!(graph,
                    [marg_sym_2D, :flat_weights, :upper_recipe, :lower_recipe, :live_map, :triagonal_config, :vsel_map, :domain_lo, :domain_hi, :support_lo, :support_hi],
                    [primitive_symbols_2D[k]]
                ) do inputs, changed, cached
                    @nospecialize(changed, cached)
                    # By-name access -- see the 1D loop's matching comment.
                    coords = getproperty(inputs, marg_sym_2D)
                    weights = inputs.flat_weights
                    config = inputs.triagonal_config
                    (; upper_recipe, lower_recipe, live_map, vsel_map, domain_lo, domain_hi, support_lo, support_hi) = inputs
                    cell_status = live_map[j, i] ? LiveCell() : DeadCell()
                    recipe_status = determine_recipe_status(recipe, upper_recipe(), lower_recipe())
                    # Domain/support ride along in the config as in the 1D loop;
                    # tuple order matches the view's row order (vsel[1]=x, vsel[2]=y).
                    vsel = vsel_map[j, i]
                    cfg = (vsel[1] == 0 || isempty(domain_lo)) ? config :
                        (; config..., domain=((domain_lo[vsel[1]], domain_hi[vsel[1]]), (domain_lo[vsel[2]], domain_hi[vsel[2]])))
                    cfg = (vsel[1] == 0 || isempty(support_lo)) ? cfg :
                        (; cfg..., support=((support_lo[vsel[1]], support_hi[vsel[1]]), (support_lo[vsel[2]], support_hi[vsel[2]])))
                    return (compute_plotting_primitives(coords, weights, recipe, recipe_status, cell_status, cfg),)
                end
            end

            # Registered separately (needs :flat_chainids); same primitive_symbol
            # naming, so _init_gridlayout's lookup finds it transparently.
            chainscatter_primitive_sym = primitive_symbol(ChainScatter2D(), (j, i))
            register_computation!(graph,
                [marg_sym_2D, :flat_weights, :flat_chainids, :window_data, :upper_recipe, :lower_recipe, :live_map, :triagonal_config],
                [chainscatter_primitive_sym]
            ) do inputs, changed, cached
                @nospecialize(changed, cached)
                # By-name access -- see the 1D loop's matching comment.
                coords = getproperty(inputs, marg_sym_2D)
                (; flat_weights, flat_chainids, upper_recipe, lower_recipe, live_map, triagonal_config) = inputs
                cell_status = live_map[j, i] ? LiveCell() : DeadCell()
                recipe_status = determine_recipe_status(ChainScatter2D(), upper_recipe(), lower_recipe())
                primitives = compute_plotting_primitives(coords, flat_weights, flat_chainids, ChainScatter2D(), recipe_status, cell_status, triagonal_config)
                return ((; primitives..., all_chainids=inputs.window_data.chainids),)
            end

            trace_primitive_sym = primitive_symbol(Trace2D(), (j, i))
            register_computation!(graph,
                [:window_data, :trace_groups, :vsel_map, :live_map], [trace_primitive_sym]
            ) do inputs, changed, cached
                @nospecialize(changed, cached)
                if !inputs.live_map[j, i] || isempty(inputs.trace_groups)
                    return (_empty_trace2d_primitives(),)
                end
                return (_timed_trace_primitives(inputs.window_data, inputs.trace_groups, inputs.vsel_map[j, i]),)
            end

        end
    end

    return graph
end
