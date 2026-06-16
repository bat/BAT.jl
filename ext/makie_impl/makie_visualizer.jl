
const BAT_MAKIE_RECIPES_1D = [
    Hist1D(),
    QuantileHist1D(),
    KDE1D(),
    QuantileKDE1D(),
    Std1D(),
    Mean1D(),
    Errorbars1D(),
    #PDF1D()
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


function BATVisualizer(vis::BATMakieVisualization)
    (; recipes, N_max, triagonal_config, diagonal_config) = vis

    # TODO: Think about whether or not the more expressive vsel symbol approach is desired, or integers are sufficient
    # idxs = vsel isa AbstractVector{<:Int64} ? vsel : reduce(vcat, asindex.(Ref(vs), vsel))
    #idxs = filter(<=(totalndof(vs)), vsel)

    graph = _init_compute_graph(
        recipes,
        triagonal_config,
        diagonal_config,
        N_max,
    )

    content = (
        graph=graph,
        buffer_lock=ReentrantLock(),
        chain_ids=Vector{Integer}(),
        output_buffer=Vector{Vector{DensitySampleVector}}(),
        n_buffer_samples=Ref(0),
        is_live=Threads.Atomic{Bool}(true)
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
        (smpls, idx) -> idx > 0 ? view(smpls.weight, 1:idx) : view(Int64[0], 1:1), # TODO: MD, think of something smarter to determine the weight type. Ad hoc fix to fixed Int64
        graph,
        [:flat_samples, :current_idx],
        :flat_weights
    )

    add_input!(graph, :idxs, Integer[])

    register_computation!(graph,
        [:idxs],
        [:vsel_map],
    ) do inputs, changed, cached
        idxs = inputs.idxs
        if isnothing(cached)
            vsel_map = [(i, j) for i in 1:n, j in 1:n]
        else
            vsel_map = cached.vsel_map
        end

        # TODO: Refine update logic to avoid redundant recomputations
        for (i, v_1) in enumerate(idxs)
            for (j, v_2) in enumerate(idxs)
                vsel_map[i, j] = (v_1, v_2)
            end
        end

        return (vsel_map,)
    end

    register_computation!(graph,
        [:idxs],
        [:live_map],
    ) do inputs, changed, cached
        # TODO: Refine update logic together with the vsel_map.

        live_map = fill(false, n, n)

        idxs = inputs.idxs
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

    for recipe in vcat(BAT_MAKIE_RECIPES_1D, BAT_MAKIE_RECIPES_2D)
        add_input!(graph, Symbol("$(typeof(recipe))"), recipe)
    end

    for i in 1:n
        #1D marginal views
        marg_sym = marg_symbol((i, i))

        map!(
            (smpls, vsel_map, current_idx) -> current_idx > 0 ? view(smpls.v.data, [vsel_map[i, i][1]], 1:current_idx) : view(ElasticMatrix{Float64,Vector{Float64}}(undef, 1, 0), [1], 1:0),
            graph,
            [:flat_samples, :vsel_map, :current_idx],
            marg_sym
        )

        map!(
            (marg, current_idx) -> isempty(marg) ? (0.0, 1.0) : (minimum(marg) - 0.1 * abs(minimum(marg)), maximum(marg) + 0.1 * abs(minimum(marg))),
            graph,
            [marg_sym, :current_idx],
            Symbol("axis_limits_$i")
        )

        primitive_symbols_1D = [primitive_symbol(recipe, (i, i)) for recipe in BAT_MAKIE_RECIPES_1D]

        for k in eachindex(primitive_symbols_1D)
            register_computation!(graph,
                [marg_sym, :flat_weights, :diagonal_recipe, :live_map, :diagonal_config],
                [primitive_symbols_1D[k]]
            ) do inputs, changed, cached
                coords, weights, live_recipe, live_map, config = inputs
                recipe = BAT_MAKIE_RECIPES_1D[k]
                cell_status = live_map[i, i] ? LiveCell() : DeadCell()
                recipe_status = determine_recipe_status(recipe, live_recipe())
                primitives = compute_plotting_primitives(coords, weights, recipe, recipe_status, cell_status, config)

                return (primitives,)
            end
        end

        for j in i+1:n
            marg_sym_2D = marg_symbol((j, i))
            map!(
                (smpls, vsel_map, current_idx) -> current_idx > 0 ? view(smpls.v.data, [vsel_map[j, i]...], 1:current_idx) : view(ElasticMatrix{Float64,Vector{Float64}}(undef, 2, 0), [1, 2], 1:0),
                graph,
                [:flat_samples, :vsel_map, :current_idx],
                marg_sym_2D
            )

            primitive_symbols_2D = [primitive_symbol(recipe, (j, i)) for recipe in BAT_MAKIE_RECIPES_2D]

            for k in eachindex(primitive_symbols_2D)
                register_computation!(graph,
                    [marg_sym_2D, :flat_weights, :upper_recipe, :lower_recipe, :live_map, :triagonal_config],
                    [primitive_symbols_2D[k]]
                ) do inputs, changed, cached
                    coords, weights, live_recipe_upper, live_recipe_lower, live_map, config = inputs
                    recipe = BAT_MAKIE_RECIPES_2D[k]
                    cell_status = live_map[i, j] ? LiveCell() : DeadCell()
                    recipe_status = determine_recipe_status(recipe, live_recipe_upper(), live_recipe_lower())
                    primitives = compute_plotting_primitives(coords, weights, recipe, recipe_status, cell_status, config)
                    return (primitives,)
                end
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
        graph[:upper_recipe],
        graph[:diagonal_recipe],
        graph[:lower_recipe],
        graph[:show_stats_upper],
        graph[:show_stats_diag],
        graph[:show_stats_lower]
    ) do idx, upper_recipe, diagonal_recipe, lower_recipe, stats_upper, stats_diag, stats_lower
        matrix = Matrix{Any}(undef, n, n)
        triagonal_config = graph[:triagonal_config][]
        diagonal_config = graph[:diagonal_config][]

        for i in 1:n
            diagonal_primitives = graph[primitive_symbol(diagonal_recipe, (i, i))][]
            diagonal_plotspecs = compose_plotspecs(diagonal_primitives, diagonal_recipe(), diagonal_config)
            stats_specs_1D = stats_diag ? get_stats_plotspecs(graph, (i, i), Makie1DStats(), diagonal_config) : []
            append!(diagonal_plotspecs, stats_specs_1D)

            xlims = graph[Symbol("axis_limits_$i")][]
            show_x_cosmetics = (i == n) || (i == 1)
            matrix[i, i] = S.Axis(
                plots=diagonal_plotspecs,
                limits=(xlims, nothing),
                xticklabelsvisible=show_x_cosmetics, xticksvisible=show_x_cosmetics,
                yticklabelsvisible=true,
                yticklabelrotation=pi / 2,
                ytickformat="{:.1f}",
                xgridvisible=true,
                ygridvisible=true,
                xaxisposition=(i == 1) ? :top : :bottom
            )

            for j in i+1:n
                # TODO: Figure out a way to flip the upper plots along the diagonal
                upper_primitives = graph[primitive_symbol(upper_recipe, (j, i))][]
                upper_plotspecs = compose_plotspecs(upper_primitives, upper_recipe(), triagonal_config)
                stats_specs_2D = stats_upper ? get_stats_plotspecs(graph, (j, i), Makie2DStats(), triagonal_config) : PlotSpec[]
                append!(upper_plotspecs, stats_specs_2D)

                ylims = graph[Symbol("axis_limits_$j")][]
                show_y_cosmetics = (j == n)
                matrix[i, j] = S.Axis(
                    plots=upper_plotspecs,
                    aspect=1,
                    limits=(ylims, xlims),
                    xticklabelsvisible=show_x_cosmetics, xticksvisible=show_x_cosmetics,
                    yticklabelsvisible=show_y_cosmetics, yticksvisible=show_y_cosmetics,
                    yticklabelrotation=pi / 2,
                    xgridvisible=true,
                    ygridvisible=true,
                    xaxisposition=:top,
                    yaxisposition=:right,
                )
            end
            for j in 1:i-1
                lower_primitives = graph[primitive_symbol(lower_recipe, (i, j))][]
                lower_plotspecs = compose_plotspecs(lower_primitives, lower_recipe(), triagonal_config)
                stats_specs_2D = stats_lower ? get_stats_plotspecs(graph, (i, j), Makie2DStats(), triagonal_config) : PlotSpec[]
                append!(lower_plotspecs, stats_specs_2D)

                ylims = graph[Symbol("axis_limits_$j")][]
                show_y_cosmetics = (j == 1)
                matrix[i, j] = S.Axis(
                    plots=lower_plotspecs,
                    aspect=1,
                    limits=(xlims, ylims),
                    xticklabelsvisible=show_x_cosmetics, xticksvisible=show_x_cosmetics,
                    yticklabelsvisible=show_y_cosmetics, yticksvisible=show_y_cosmetics,
                    yticklabelrotation=pi / 2,
                    xgridvisible=true,
                    ygridvisible=true,
                )
            end
        end
        return S.GridLayout(matrix)
    end

    return gridlayout
end


function _build_fig(graph::ComputeGraph, gridlayout::Any)
    fig = Figure()

    plot(fig[1, 1], gridlayout)

    colsize!(fig.layout, 1, Aspect(1, 1))
    rowsize!(fig.layout, 1, Relative(0.8))

    ui_layout = fig[2, 1] = GridLayout()

    options2D = [
        ("QuantileHist", QuantileHist2D),
        ("Hist", Hist2D),
        ("Scatter", Scatter2D),
        ("Hexbin", Hexbin2D),
        ("QuantileKDE", QuantileKDE2D),
        ("KDE", KDE2D),
    ]
    options1D = [
        ("QuantileHist", QuantileHist1D),
        ("Hist", Hist1D),
        ("KDE", KDE1D),
        ("QuantileKDE", QuantileKDE1D),
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
    show_slider = !isempty(curr_idxs[1]) && length(curr_idxs) == 1
    if show_slider
        slider_curr_idx = Slider(fig, range=1:graph[:current_idx][], startvalue=graph[:current_idx][])
    end

    ui_layout[2, 1] = Label(fig, "Upper")
    ui_layout[3, 1] = Label(fig, "Diagonal")
    ui_layout[4, 1] = Label(fig, "Lower")

    ui_layout[1, 2] = Label(fig, "Recipe")
    ui_layout[2, 2] = menu_upper
    ui_layout[3, 2] = menu_diagonal
    ui_layout[4, 2] = menu_lower

    ui_layout[1, 3] = Label(fig, "Stats overlay")
    toggle_upper = Toggle(ui_layout[2, 3], active=false)
    toggle_diag = Toggle(ui_layout[3, 3], active=false)
    toggle_lower = Toggle(ui_layout[4, 3], active=false)

    if show_slider
        ui_layout[5, 1] = Label(fig, "Current Idx")
        ui_layout[5, 2] = slider_curr_idx
    end

    colsize!(ui_layout, 1, Auto())
    colsize!(ui_layout, 2, 200)
    colsize!(ui_layout, 3, Auto())

    rowsize!(fig.layout, 2, Auto())

    if show_slider
        on(slider_curr_idx.value) do curr_idx
            if length(graph[:current_idxs][]) > 1
                println("Figure out how to pan through mulit walker samples.")
            else
                update!(graph, current_idxs=[[curr_idx]])
            end
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

    return fig
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

function BAT.init_visualizer!(
    vis::BATVisualizer{BATMakieVisualization};
    mcmc_states::Vector{<:MCMCState},
    outputs,#::Vector{Vector{DensitySampleVector}},
    f_pretransform::Function
)
    warmup_makie_shaders()

    for (i, state) in enumerate(mcmc_states)
        #register_state_for_vis!(vis, state, _transform_walker_outputs(f_pretransform, outputs[i]))
        register_state_for_vis!(vis, state, _unshape_walker_outputs(outputs[i]))
    end

    (; graph, chain_ids, buffer_lock, output_buffer, n_buffer_samples, is_live) = vis.content

    errormonitor(
        @async begin
            while is_live[]
                # TODO: MD, Discuss update scheme
                # TODO: MD, Figure out good sleep time
                sleep(0.1)

                update_graph = n_buffer_samples[] >= 50

                n_smpls_buff_1 = sum(length.(output_buffer[1]))
                # println("Buffer slot 1 has $n_smpls_buff_1 samples")
                #println("n_buffer_samples = $n_buffer_samples[]")
                if update_graph
                    lock(buffer_lock)
                    extracted_output_buffer = deepcopy(output_buffer)
                    output_buffer .= _empty_chain_outputs.(mcmc_states)
                    n_buffer_samples[] = 0

                    # fresh_batch_trafo = _transform_chain_outputs(f_pretransform, extracted_output_buffer)
                    fresh_batch_trafo = _unshape_chain_outputs(extracted_output_buffer)

                    samples = graph[:samples][]
                    samples_new = _append_chain_outputs(mcmc_states[1], samples, fresh_batch_trafo)

                    update!(graph, samples=samples_new)

                    batch_lengths = [length.(chain_batch) for chain_batch in fresh_batch_trafo]
                    current_idxs = graph[:current_idxs][]
                    current_idxs_new = [current_idxs[i] .+ batch_lengths[i] for i in eachindex(current_idxs)]
                    update!(graph, current_idxs=current_idxs_new)

                    #TODO: MD, figure out more graceful check. And discuss if it is even desired to activate the vis if no samples exist
                    exist_samples_for_vis = any([any(.!isempty.(chain_output)) for chain_output in samples])
                    if exist_samples_for_vis
                        update!(graph, idxs=vis.backend.vsel)
                    else
                        update!(graph, idxs=Integer[])
                    end

                    unlock(buffer_lock)
                end
            end
        end
    )

    with_theme(bat_theme()) do
        gridlayout = _init_gridlayout(graph, vis.backend.N_max)
        fig = _build_fig(graph, gridlayout)
        display(fig)
    end
end

function BAT.update_visualizer_impl!(
    vis::BATVisualizer{BATMakieVisualization};
    chain_state::MCMCChainState,
    nonzero_weights::Bool
)
    (; buffer_lock, output_buffer, chain_ids, n_buffer_samples) = vis.content
    output_id = findfirst(x -> x == chain_state.info.id, chain_ids)
    n_smpls_start = sum(length.(output_buffer[output_id]))
    get_samples!(output_buffer[output_id], chain_state, nonzero_weights)
    n_smpls_end = sum(length.(output_buffer[output_id]))

    lock(buffer_lock)
    n_new = n_smpls_end - n_smpls_start
    n_buffer_samples[] += n_new
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
        # TODO: Figure out how to drop low weight samples from the view
        # marg_samples = BAT.drop_low_weight_samples(marg_samples)
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
