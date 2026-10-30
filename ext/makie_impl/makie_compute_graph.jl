# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# Bounds for the whole selected histogram grid, not the visible window.
# Keep the statistical domain unchanged; only the display envelope grows.
function _extend_hist_axis_limit(limits, edges)
    lo, hi = Float64(first(edges)), Float64(last(edges))
    isfinite(lo) && isfinite(hi) && lo < hi ||
        throw(ArgumentError("Histogram display edges must have finite ordered endpoints"))
    return (min(limits[1], lo), max(limits[2], hi))
end


function _histogram_axis_limits(lo, hi, idxs, n, diagonal_recipe, upper_recipe,
    lower_recipe, diagonal_config, triagonal_config)
    limits = fill((0.0, 1.0), n)
    length(idxs) <= n ||
        throw(ArgumentError("idxs has $(length(idxs)) entries, exceeding the grid size N_max=$n"))
    (isempty(lo) || isempty(idxs)) && return limits
    for (i, v) in enumerate(idxs)
        span = hi[v] - lo[v]
        margin = iszero(span) ? max(abs(hi[v]), one(hi[v])) * 0.05 : 0.05 * span
        limits[i] = (lo[v] - margin, hi[v] + margin)
    end
    if diagonal_recipe in (Hist1D, QuantileHist1D)
        nbins = diagonal_config.nbins
        bins = diagonal_recipe === Hist1D ? Base.Checked.checked_add(nbins, 1) : nbins
        for (i, v) in enumerate(idxs)
            domain = (lo[v], hi[v])
            edges = only(_hist_edges(domain, nothing, bins, diagonal_config.closed))
            limits[i] = _extend_hist_axis_limit(limits[i], edges)
        end
    end
    if length(idxs) > 1 &&
        (upper_recipe in (Hist2D, QuantileHist2D) || lower_recipe in (Hist2D, QuantileHist2D))
        bins = triagonal_config.nbins
        for i in eachindex(idxs), j in i+1:length(idxs)
            # Pair primitives use (larger slot, smaller slot); their edge
            # fallback is coupled across dimensions, so plan the full pair.
            domain = ((lo[idxs[j]], hi[idxs[j]]), (lo[idxs[i]], hi[idxs[i]]))
            xedges, yedges = _hist_edges(domain, nothing, bins, triagonal_config.closed)
            limits[j] = _extend_hist_axis_limit(limits[j], xedges)
            limits[i] = _extend_hist_axis_limit(limits[i], yedges)
        end
    end
    return limits
end


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
function primitive_symbol(recipe, vsel::Tuple{Integer,Integer})
    return Symbol(string(recipe), "_prim_", vsel[1], "_", vsel[2])
end


function primitive_symbol(recipe::R, vsel::Tuple{Integer,Integer}) where {R<:BATMakieRecipe}
    return Symbol(string(typeof(recipe)), "_prim_", vsel[1], "_", vsel[2])
end


function marg_symbol(vsel::Tuple{Integer,Integer})
    return Symbol("marg_$(vsel[1])_$(vsel[2])")
end


# Estimate display limits from prior quantiles in the original coordinates.
# These limits do not define hard support.
function _estimate_prior_domain(rng::AbstractRNG, target, n_dof::Integer;
    n_prior_samples::Integer = 2000, tail_prob::Real = 0.0015,
    display_shape = missing, f_display = identity)
    initsrc = BAT.get_initsrc_from_target(target)
    # A pushforward posterior initializes from its origin's prior, then transforms.
    context = initsrc isa BAT.BATPushFwdMeasure && !BAT.supports_rand(initsrc) ?
        BAT.BATContext(rng = rng) : nothing
    # A raw empirical NamedTuple source need not carry container shape metadata
    # when the caller already supplied the original target's shape.
    shape = ismissing(display_shape) || f_display !== identity ? varshape(initsrc) : display_shape
    output_shape = ismissing(display_shape) ? shape : display_shape
    draws = map(1:n_prior_samples) do _
        value = isnothing(context) ? rand(rng, initsrc) :
            BAT.bat_initval_impl(initsrc, BAT.InitFromTarget(), context).result
        displayed = f_display === identity ? value : f_display(unshaped(value, shape))
        unshaped(displayed, output_shape)
    end
    M = reduce(hcat, draws)
    size(M, 1) == n_dof ||
        throw(ArgumentError("Domain draws do not match the original display shape"))
    lo = [quantile(view(M, d, :), tail_prob) for d in 1:n_dof]
    hi = [quantile(view(M, d, :), 1 - tail_prob) for d in 1:n_dof]
    return lo, hi
end


# Retain original prior-support bounds until sample membership is checked.
# Infinite endpoints disable KDE reflection at that boundary.
function _support_bounds(target, n_dof::Integer)
    d = _support_root_dist(target)
    isnothing(d) && return (fill(-Inf, n_dof), fill(Inf, n_dof))
    lo, hi = _component_support(d)
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

# Component support follows the unshaped coordinate order, as in truncate_dist_hard.
function _component_support(d::UnivariateDistribution)
    lo, hi = _kde_support_bounds((minimum(d), maximum(d)))
    return ([lo], [hi])
end


function _component_support(d::Distributions.Product)
    bounds = [_kde_support_bounds((minimum(c), maximum(c))) for c in d.v]
    return (Real[first(s) for s in bounds], Real[last(s) for s in bounds])
end


_component_support(d::ValueShapes.UnshapedNTD) = _component_support(d.shaped)
# Constant components have no unshaped coordinates.
_component_support(::ConstValueDist) = (Float64[], Float64[])
_component_support(d::Distribution) = (fill(-Inf, length(d)), fill(Inf, length(d)))

function _component_support(d::NamedTupleDist)
    n = totalndof(varshape(d))
    lo = Real[-Inf for _ in 1:n]
    hi = Real[Inf for _ in 1:n]
    dists = values(d)
    accessors = values(varshape(d))
    for (dd, acc) in zip(dists, accessors)
        # Scalar accessors return an Int; use a range for assignment below.
        raw_idxs = ValueShapes.view_idxs(1:n, acc)
        idxs = raw_idxs isa Integer ? (raw_idxs:raw_idxs) : raw_idxs
        clo, chi = _component_support(dd)
        # Leave unmatched components unbounded.
        length(clo) == length(idxs) || continue
        lo[idxs] .= clo
        hi[idxs] .= chi
    end
    return (lo, hi)
end


# Validate explicit support pairs or infer per-coordinate bounds from a measure.
_support_vectors(::Nothing, n_dof::Integer) = (Float64[], Float64[])

function _support_vectors(support::AbstractVector, n_dof::Integer)
    length(support) == n_dof || throw(ArgumentError(
        "`support` has $(length(support)) entries but the model has $n_dof dimensions"))
    any(isnothing, support) && throw(ArgumentError(
        "`support` entries must be (lo, hi) pairs; use (-Inf, Inf) for an unconstrained dimension"))
    bounds = map(_kde_support_bounds, support)
    lo = Real[first(s) for s in bounds]
    hi = Real[last(s) for s in bounds]
    return lo, hi
end


_support_vectors(support, n_dof::Integer) = _support_bounds(support, n_dof)

# Use finite coordinate extrema; infinite spans would make degenerate-edge padding NaN.
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


function BAT.BATVisualizer(vis::BATMakieVisualization)
    (; recipes, N_max, triagonal_config, diagonal_config) = vis


    graph = _init_compute_graph(
        recipes,
        triagonal_config,
        diagonal_config,
        N_max,
    )

    buffer_lock = ReentrantLock()
    content = (
        graph = Ref{Union{Nothing,ComputeGraph}}(graph),
        buffer_lock = buffer_lock,
        # Waiters and notifiers must share buffer_lock.
        buffer_cond = Threads.Condition(buffer_lock),
        chain_ids = Vector{Int32}(),  # MCMCChainStateInfo ids are Int32
        output_buffer = Vector{Vector{DensitySampleVector}}(),
        # Latest owned current-state snapshot per chain/walker, including dwell
        # not yet emitted by the sampler. Slots are replaced under buffer_lock.
        pending_samples = Vector{Vector{DensitySampleVector}}(),
        output_timing = SampleStepSpans(),
        pending_timing = SampleStepSpans(),
        step_mode = Ref(false),
        capture_init = Ref(false),
        init_segments = Dict{Tuple{Int32,Int32},Vector{InitStepSegment}}(),
        invalid_init = Set{Tuple{Int32,Int32}}(),
        n_buffer_steps = Ref(0),
        buffer_dirty = Ref(false),
        # Derive backpressure from the flush threshold so the trigger remains reachable.
        effective_batch_size = Ref(vis.n_batch),
        buffer_ratio = vis.max_buffered / vis.n_batch,
        lifecycle = _MakieLiveControl(),
        # Set in init_visualizer!; read by _apply_vsel! to validate vsel changes.
        n_dof = Ref{Int}(0)
    )

    return BATVisualizer(vis, content)
end


# Registered closures share this scope; keep captured variable names distinct.
function _init_compute_graph(
    recipes::NamedTuple, triagonal_config::NamedTuple, diagonal_config::NamedTuple, n::Integer
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
    register_computation!(
        graph, [:samples, :current_idxs, :sample_timing], [:window_data]
    ) do inputs, changed, cached
        @nospecialize(changed, cached)
        return (_sample_window_data(inputs.samples, inputs.current_idxs, inputs.sample_timing),)
    end
    register_computation!(
        graph, [:window_data, :window_steps], [:flat_samples, :window_segments]
    ) do inputs, changed, cached
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
        (smpls, idx) -> view(smpls.weight, 1:idx),
        graph,
        [:flat_samples, :current_idx],
        :flat_weights
    )
    map!(
        # Empty chain IDs let the graph resolve before recipe availability is checked.
        (smpls, idx) -> _samples_have_chain_ids(smpls) ?
            Int32[s.chainid for s in view(smpls.info, 1:idx)] : Int32[],
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
        length(idxs) <= n ||
            throw(ArgumentError("idxs has $(length(idxs)) entries, exceeding the grid size N_max=$n"))

        n_active = length(idxs)
        vsel_map = Matrix{Tuple{Int,Int}}(undef, n, n)
        for i in 1:n, j in 1:n
            # Zero indices mark inactive slots and cannot select sample coordinates.
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
        length(idxs) <= n ||
            throw(ArgumentError("idxs has $(length(idxs)) entries, exceeding the grid size N_max=$n"))

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

    add_input!(graph, :show_trace_upper, false)
    add_input!(graph, :show_trace_lower, false)

    add_input!(graph, :triagonal_config, triagonal_config)
    add_input!(graph, :diagonal_config, diagonal_config)

    # Display domain for axis limits and histogram edges; empty until setup.
    # Live publications can replace it; window selection leaves it unchanged.
    add_input!(graph, :domain_lo, Float64[])
    add_input!(graph, :domain_hi, Float64[])

    # Hard-support bounds for KDE reflection, separate from the display domain.
    # Infinite endpoints are unbounded; empty vectors disable correction.
    add_input!(graph, :support_lo, Float64[])
    add_input!(graph, :support_hi, Float64[])

    for recipe in vcat(BAT_MAKIE_RECIPES_1D, BAT_MAKIE_RECIPES_2D)
        add_input!(graph, Symbol("$(typeof(recipe))"), recipe)
    end
    # Chain scatter and trace use additional inputs and are registered separately.
    add_input!(graph, Symbol("$(ChainScatter2D)"), ChainScatter2D())
    add_input!(graph, Symbol("$(Trace2D)"), Trace2D())
    register_computation!(graph,
        [:window_segments, :triagonal_config, :show_trace_upper, :show_trace_lower], [:trace_groups]
    ) do inputs, changed, cached
        @nospecialize(changed, cached)
        return (_window_trace_groups(
            inputs.show_trace_upper || inputs.show_trace_lower ? inputs.window_segments : WindowStepGroup[],
            inputs.triagonal_config.trace_nsteps),)
    end


    # Window changes reuse this envelope. Domain, recipe, configuration or
    # variable-selection changes recompute it for the whole grid; sample data
    # affect it only through a changed domain.
    register_computation!(graph,
        [:domain_lo, :domain_hi, :idxs, :diagonal_recipe, :upper_recipe,
         :lower_recipe, :diagonal_config, :triagonal_config], [:axis_limits],
    ) do inputs, changed, cached
        @nospecialize(changed, cached)
        return (_histogram_axis_limits(inputs.domain_lo, inputs.domain_hi,
            inputs.idxs, n, inputs.diagonal_recipe, inputs.upper_recipe,
            inputs.lower_recipe, inputs.diagonal_config, inputs.triagonal_config),)
    end

    for i in 1:n
        marg_sym = marg_symbol((i, i))

        map!(
            # Empty views must retain the live node's storage type.
            (smpls, vsel_map, current_idx, live_map) ->
                (current_idx > 0 && live_map[i, i]) ?
                    view(smpls.v.data, [vsel_map[i, i][1]], 1:current_idx) :
                    view(smpls.v.data, Int[], 1:0),
            graph,
            [:flat_samples, :vsel_map, :current_idx, :live_map],
            marg_sym
        )

        map!(limits -> limits[i], graph, [:axis_limits], Symbol("axis_limits_$i"))

        primitive_symbols_1D = [primitive_symbol(recipe, (i, i)) for recipe in BAT_MAKIE_RECIPES_1D]

        for k in eachindex(primitive_symbols_1D)
            recipe = BAT_MAKIE_RECIPES_1D[k]
            register_computation!(graph,
                [
                    marg_sym, :flat_weights, :diagonal_recipe, :show_stats_diag, :live_map,
                    :diagonal_config, :vsel_map, :domain_lo, :domain_hi,
                    :support_lo, :support_hi
                ],
                [primitive_symbols_1D[k]]
            ) do inputs, changed, cached
                @nospecialize(changed, cached)
                coords = getproperty(inputs, marg_sym)
                weights = inputs.flat_weights
                config = inputs.diagonal_config
                (; live_map, vsel_map, domain_lo, domain_hi, support_lo, support_hi) = inputs
                cell_status = live_map[i, i] ? LiveCell() : DeadCell()
                recipe_status = determine_recipe_status(
                    recipe, inputs.diagonal_recipe(), inputs.show_stats_diag
                )
                v = vsel_map[i, i][1]
                cfg = (v == 0 || isempty(domain_lo)) ? config :
                    (; config..., domain = (domain_lo[v], domain_hi[v]))
                cfg = (v == 0 || isempty(support_lo)) ? cfg :
                    (; cfg..., support = (support_lo[v], support_hi[v]))
                return (compute_plotting_primitives(
                    coords, weights, recipe, recipe_status, cell_status, cfg
                ),)
            end
        end

        for j in i+1:n
            marg_sym_2D = marg_symbol((j, i))
            map!(
                (smpls, vsel_map, current_idx, live_map) ->
                    (current_idx > 0 && live_map[j, i]) ?
                        view(smpls.v.data, [vsel_map[j, i]...], 1:current_idx) :
                        view(smpls.v.data, Int[], 1:0),
                graph,
                [:flat_samples, :vsel_map, :current_idx, :live_map],
                marg_sym_2D
            )
            primitive_symbols_2D = [primitive_symbol(recipe, (j, i)) for recipe in BAT_MAKIE_RECIPES_2D]

            for k in eachindex(primitive_symbols_2D)
                recipe = BAT_MAKIE_RECIPES_2D[k]
                register_computation!(graph,
                    [
                        marg_sym_2D, :flat_weights, :upper_recipe, :lower_recipe,
                        :show_stats_upper, :show_stats_lower,
                        :live_map, :triagonal_config, :vsel_map, :domain_lo, :domain_hi,
                        :support_lo, :support_hi
                    ],
                    [primitive_symbols_2D[k]]
                ) do inputs, changed, cached
                    @nospecialize(changed, cached)
                    coords = getproperty(inputs, marg_sym_2D)
                    weights = inputs.flat_weights
                    config = inputs.triagonal_config
                    (;
                        upper_recipe, lower_recipe, live_map, vsel_map,
                        domain_lo, domain_hi, support_lo, support_hi
                    ) = inputs
                    cell_status = live_map[j, i] ? LiveCell() : DeadCell()
                    recipe_status = determine_recipe_status(
                        recipe, upper_recipe(), lower_recipe(),
                        inputs.show_stats_upper || inputs.show_stats_lower
                    )
                    # Domain and support follow the coordinate view's row order: x, then y.
                    vsel = vsel_map[j, i]
                    cfg = (vsel[1] == 0 || isempty(domain_lo)) ? config :
                        (; config..., domain = (
                            (domain_lo[vsel[1]], domain_hi[vsel[1]]),
                            (domain_lo[vsel[2]], domain_hi[vsel[2]])
                        ))
                    cfg = (vsel[1] == 0 || isempty(support_lo)) ? cfg :
                        (; cfg..., support = (
                            (support_lo[vsel[1]], support_hi[vsel[1]]),
                            (support_lo[vsel[2]], support_hi[vsel[2]])
                        ))
                    return (compute_plotting_primitives(
                        coords, weights, recipe, recipe_status, cell_status, cfg
                    ),)
                end
            end

            chainscatter_primitive_sym = primitive_symbol(ChainScatter2D(), (j, i))
            register_computation!(graph,
                [
                    marg_sym_2D, :flat_weights, :flat_chainids, :window_data,
                    :upper_recipe, :lower_recipe, :live_map, :triagonal_config
                ],
                [chainscatter_primitive_sym]
            ) do inputs, changed, cached
                @nospecialize(changed, cached)
                coords = getproperty(inputs, marg_sym_2D)
                (;
                    flat_weights, flat_chainids, upper_recipe, lower_recipe,
                    live_map, triagonal_config
                ) = inputs
                cell_status = live_map[j, i] ? LiveCell() : DeadCell()
                recipe_status = determine_recipe_status(ChainScatter2D(), upper_recipe(), lower_recipe())
                primitives = compute_plotting_primitives(
                    coords, flat_weights, flat_chainids, ChainScatter2D(),
                    recipe_status, cell_status, triagonal_config
                )
                return ((; primitives..., all_chainids = inputs.window_data.chainids),)
            end

            trace_primitive_sym = primitive_symbol(Trace2D(), (j, i))
            register_computation!(graph,
                [:window_data, :trace_groups, :vsel_map, :live_map], [trace_primitive_sym]
            ) do inputs, changed, cached
                @nospecialize(changed, cached)
                if !inputs.live_map[j, i] || isempty(inputs.trace_groups)
                    return (_empty_trace2d_primitives(),)
                end
                return (_timed_trace_primitives(
                    inputs.window_data, inputs.trace_groups, inputs.vsel_map[j, i]
                ),)
            end

        end
    end

    return graph
end
