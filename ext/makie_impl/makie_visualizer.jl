
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


struct BATMakieVisualizerState
    vis::BATMakieVisualizer
    graph::Any
    gridlayout::Any
end


function BATMakieVisualizer()
    recipes = (upper=QuantileHist2D, diagonal=Hist1D, lower=Hist2D)
    vsel = [1, 2, 3] # Figure out default values. Pass samples to determine number of parameters?
    N_max = 3 # TODO: Can cause errors when the number of dimensions in the data is smaller than N_max. Figure out a way to make safe.
    n_batch = 10

    triagonal_config = (
        weights=nothing,
        nsigma=1.0,
        nbins=(50, 50),
        closed=:left,
        normalization=:pdf,
        levels=cdf.(Chi(2), 0:3),
        filter=false,
        colormap=:Blues,
        color=Makie.wong_colors()[1],
        color_stats=:red,
        alpha=1.0,
        rev=false,
        threshold=nothing,
        markersize=2.0,
        edge=false,
        strokecolor=Makie.wong_colors()[1],
        strokewidth=1.0,
        strokestyle_stats=:solid,
        strokewidth_stats=2.0,
        color_mean=:black,
        strokestyle_mean=:dot,
        strokewidth_mean=2.0,
        color_ebars=:red,
        whiskerwidth=10
    )

    diagonal_config = (
        weights=nothing,
        nsigma=1.0,
        nbins=30,
        closed=:left,
        normalization=:pdf,
        levels=cdf.(Chi(1), 0:3),
        filter=false,
        color=Makie.wong_colors()[1],
        color_stats=:red,
        colormap=:Blues,
        alpha=1.0,
        filled=true,
        edge=false,
        strokecolor=Makie.wong_colors()[1],
        strokewidth=1.0,
        strokestyle_stats=:solid,
        strokewidth_stats=2.0,
        strokestyle_mean=:dot,
        strokewidth_mean=2.0,
        y_ebars=0.0,
        color_ebars=:red,
        whiskerwidth=10,
        filled_pdf=true,
        npoints_pdf=300,
        rev=false
    )

    vis = BATMakieVisualizer(
        recipes,
        vsel,
        N_max,
        n_batch,
        triagonal_config,
        diagonal_config,
    )
    return vis
end


function Makie.convert_arguments(
    ::Type{<:AbstractPlot},
    vis::BATMakieVisualizer
)
    (; axspecs, ui_controls) = vis

    gridspec = S.GridLayout(axspecs) # Embedd UI controls, but later.

    return gridspec
end


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

function init_visualizer(
    vis::BATMakieVisualizer,
    smpls::DensitySampleVector,
    sampling::SA
) where {SA<:AbstractSamplingAlgorithm}
    (; recipes, vsel, N_max, n_batch, triagonal_config, diagonal_config) = vis
    vs = varshape(smpls)

    # TODO: Think about whether or not the more expressive vsel symbol approach is desired, or integers are sufficient
    # idxs = vsel isa AbstractVector{<:Int64} ? vsel : reduce(vcat, asindex.(Ref(vs), vsel))
    idxs = filter(<=(totalndof(vs)), vsel)

    graph = _init_compute_graph(
        smpls,
        idxs,
        recipes,
        triagonal_config,
        diagonal_config,
        N_max,
        n_batch
    )

    gridlayout = _init_gridlayout(
        graph,
        N_max
    )

    vis_state = BATMakieVisualizerState(
        vis,
        graph,
        gridlayout
    )

    fig = _build_fig(vis_state)

    return vis_state, fig
end


function _init_compute_graph(
    smpls::DensitySampleVector,
    idxs::Vector{Integer},
    recipes::NamedTuple,
    triagonal_config::NamedTuple,
    diagonal_config::NamedTuple,
    n::Integer,
    n_batch_init
)
    graph = ComputeGraph()

    batch = smpls[end-n_batch_init:end]

    add_input!(graph, :samples, smpls)
    add_input!(graph, :current_idx, length(smpls))

    map!(
        (smpls, curr_idx) -> view(smpls.weight, 1:curr_idx),
        graph,
        [:samples, :current_idx],
        :weights
    )
    map!(
        (smpls, curr_idx) -> view(smpls.logd, 1:curr_idx),
        graph,
        [:samples, :current_idx],
        :logds
    )


    add_input!(graph, :idxs, idxs)

    register_computation!(graph,
        [:idxs],
        [:vsel_map],
    ) do inputs, changed, cached
        idxs = inputs.idxs
        if isnothing(cached)
            vsel_map = [(i, j) for i in 1:n, j in 1:n]
        else
            vsel_map = cached
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

        idxs = inputs.idxs
        live_map = Matrix{CellStatus}(undef, n, n)

        # TODO: Refine update logic together with the vsel_map.
        n_active = length(idxs)
        for i in 1:n
            for j in 1:n
                live_map[i, j] = LiveCell()
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
            (smpls, vsel_map, current_idx) -> view(smpls.v.data, [vsel_map[i, i][1]], 1:current_idx),
            graph,
            [:samples, :vsel_map, :current_idx],
            marg_sym
        )

        map!(
            (marg, current_idx) -> (minimum(marg) - 0.1 * abs(minimum(marg)), maximum(marg) + 0.1 * abs(minimum(marg))),
            graph,
            [marg_sym, :current_idx],
            Symbol("axis_limits_$i")
        )

        primitive_symbols_1D = [primitive_symbol(recipe, (i, i)) for recipe in BAT_MAKIE_RECIPES_1D]

        for k in eachindex(primitive_symbols_1D)
            register_computation!(graph,
                [marg_sym, :weights, :diagonal_recipe, :live_map, :diagonal_config],
                [primitive_symbols_1D[k]]
            ) do inputs, changed, cached
                coords, weights, live_recipe, live_map, config = inputs
                recipe = BAT_MAKIE_RECIPES_1D[k]
                cell_status = live_map[i, i]
                recipe_status = determine_recipe_status(recipe, live_recipe())
                primitives = compute_plotting_primitives(coords, weights, recipe, recipe_status, cell_status, config)

                return (primitives,)
            end
        end

        for j in i+1:n
            marg_sym_2D = marg_symbol((j, i))
            map!(
                (smpls, vsel_map, current_idx) -> view(smpls.v.data, [vsel_map[j, i]...], 1:current_idx),
                graph,
                [:samples, :vsel_map, :current_idx],
                marg_sym_2D
            )

            primitive_symbols_2D = [primitive_symbol(recipe, (j, i)) for recipe in BAT_MAKIE_RECIPES_2D]

            for k in eachindex(primitive_symbols_2D)
                register_computation!(graph,
                    [marg_sym_2D, :weights, :upper_recipe, :lower_recipe, :live_map, :triagonal_config],
                    [primitive_symbols_2D[k]]
                ) do inputs, changed, cached
                    coords, weights, live_recipe_upper, live_recipe_lower, live_map, config = inputs
                    recipe = BAT_MAKIE_RECIPES_2D[k]
                    cell_status = live_map[i, j]
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
                stats_specs_2D = stats_upper ? get_stats_plotspecs(graph, (j, i), Makie2DStats(), triagonal_config) : []
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
                stats_specs_2D = stats_lower ? get_stats_plotspecs(graph, (i, j), Makie2DStats(), triagonal_config) : []
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


function _build_fig(vis_state::BATMakieVisualizerState)
    fig = Figure()
    ui_layout = fig[2, 1] = GridLayout(tellwidth=false)

    options2D = [
        ("QuantileHist", QuantileHist2D),
        ("Hist", Hist2D),
        ("Scatter", Scatter2D),
        ("Hexbin", Hexbin2D),
        ("QuantileKDE", QuantileKDE2D),
        ("KDE", KDE2D),
        #("Cov", Symbol(Cov2D)),
        #("Std", Symbol(Std2D)),
        #("Mean", Symbol(Mean2D)),
        #("Errorbars", Symbol(Errorbars2D)),
    ]
    options1D = [
        ("QuantileHist", QuantileHist1D),
        ("Hist", Hist1D),
        ("KDE", KDE1D),
        ("QuantileKDE", QuantileKDE1D),
        #("Std", Symbol(Std1D)),
        #("Mean", Symbol(Mean1D)),
        #("Errorbars", Symbol(Errorbars1D)),
        #("PDF", Symbol(PDF1D)), TODO: Make work
    ]

    menu_upper = Menu(
        fig,
        options=options2D
    )
    menu_diagonal = Menu(
        fig,
        options=options1D
    )
    menu_lower = Menu(
        fig,
        options=options2D
    )

    slider_curr_idx = Slider(fig, range=1:100000)

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

    ui_layout[5, 1] = Label(fig, "Current Idx")
    ui_layout[5, 2] = slider_curr_idx


    plot(fig[1, 1], vis_state.gridlayout)

    on(slider_curr_idx.value) do curr_idx
        update!(vis_state.graph, current_idx=curr_idx)
    end

    on(menu_upper.selection) do selected_recipe
        update!(vis_state.graph, upper_recipe=selected_recipe)
    end
    on(menu_diagonal.selection) do selected_recipe
        update!(vis_state.graph, diagonal_recipe=selected_recipe)
    end
    on(menu_lower.selection) do selected_recipe
        update!(vis_state.graph, lower_recipe=selected_recipe)
    end

    on(toggle_upper.active) do is_live
        update!(vis_state.graph, show_stats_upper=is_live)
    end

    on(toggle_diag.active) do is_live
        update!(vis_state.graph, show_stats_diag=is_live)
    end

    on(toggle_lower.active) do is_live
        update!(vis_state.graph, show_stats_lower=is_live)
    end

    return fig
end


function update_visualizer_impl(vis::BATMakieVisualizer; kwargs)
    return nothing
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

