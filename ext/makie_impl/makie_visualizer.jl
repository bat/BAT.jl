
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


struct BATMakieVisualizerState
    recipes::NamedTuple
    vsel::Vector{Integer}
    N_max::Integer
    n_batch::Integer
    graph::Any
    gridlayout::Any
end


function BATMakieVisualizer()
    recipes = (upper=QuantileHist2D(), diagonal=Hist1D(), lower=Hist2D())
    vsel = [1, 2, 3] # Figure out default values. Pass samples to determine number of parameters?
    N_max = 3 # TODO: Can cause errors when the number of dimensions in the data is smaller than N_max. Figure out a way to make safe.
    n_batch = 10

    upper_config = (
        nbins=(30, 30),)

    diagonal_config = (
        weights=nothing,
        nbins=30,
        closed=:left,
        normalization=:pdf,
        filter=false,
        color=Makie.wong_colors()[1],
        alpha=1.0,
        filled=true,
        edge=false,
        strokecolor=Makie.wong_colors()[1],
        strokewidth=1
    )
    lower_config = (1,)

    vis = BATMakieVisualizer(
        recipes,
        vsel,
        N_max,
        n_batch,
        upper_config,
        diagonal_config,
        lower_config,
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

function primitive_symbol(recipe::R, vsel::Tuple{Int64,Int64}) where {R<:BATMakieRecipe}
    return Symbol(string(typeof(recipe)) * "_prim" * "_" * "$(vsel[1])" * "$(vsel[2])")
end

function islive_symbol(recipe::R) where {R<:BATMakieRecipe}
    return Symbol(string(typeof(recipe)) * "_islive")
end

function marg_symbol(vsel::Tuple{Int64,Int64})
    return Symbol("marg_$(vsel[1])$(vsel[2])")
end

function init_visualizer(
    vis::BATMakieVisualizer,
    smpls::DensitySampleVector,
    sampling::SA
) where {SA<:AbstractSamplingAlgorithm}
    (; recipes, vsel, N_max, n_batch, upper_config, diagonal_config, lower_config) = vis
    vs = varshape(smpls)

    # TODO: Think about whether or not the more expressive vsel symbol approach is desired, or integers are sufficient
    # idxs = vsel isa AbstractVector{<:Int64} ? vsel : reduce(vcat, asindex.(Ref(vs), vsel))
    idxs = filter(<=(totalndof(vs)), vsel)

    graph = _init_compute_graph(
        smpls,
        idxs,
        recipes,
        N_max,
        n_batch
    )

    gridlayout = _init_gridlayout(
        graph,
        N_max
    )

    vis_state = BATMakieVisualizerState(
        recipes,
        vsel,
        N_max,
        n_batch,
        upper_config,
        diagonal_config,
        lower_config,
        graph,
        gridlayout
    )

    fig = _build_fig(vis_state)

    return vis_state, fig
end


using Makie: ComputeGraph, add_input!, register_computation!


function _init_compute_graph(
    smpls::DensitySampleVector,
    idxs::Vector{Integer},
    recipes::NamedTuple,
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
        if isnothing(cached)
            live_map = fill(false, n, n)
        else
            live_map = cached
        end

        # TODO: Refine update logic together with the vsel_map.
        n_active = length(idxs)
        for i in 1:n
            for j in 1:n
                live_map[i, j] = i <= n_active && j <= n_active
            end
        end
        return (live_map,)
    end

    # Control nodes for what primitives are computed and what are set to emtpy
    add_input!(graph, :upper_recipe, Symbol("$(typeof(recipes.upper))"))
    add_input!(graph, :diagonal_recipe, Symbol("$(typeof(recipes.diagonal))"))
    add_input!(graph, :lower_recipe, Symbol("$(typeof(recipes.lower))"))

    for recipe in vcat(BAT_MAKIE_RECIPES_1D, BAT_MAKIE_RECIPES_2D)
        add_input!(graph, Symbol("$(typeof(recipe))"), recipe)
    end

    possible_recipes_1D = BAT_MAKIE_RECIPES_1D
    possible_recipes_2D = BAT_MAKIE_RECIPES_2D

    recipe_islives_1D = islive_symbol.(BAT_MAKIE_RECIPES_1D)
    recipe_islives_2D = islive_symbol.(BAT_MAKIE_RECIPES_2D)

    for i in eachindex(possible_recipes_1D)
        map!(
            recipe -> recipe == Symbol("$(typeof(possible_recipes_1D[i]))"),
            graph,
            :diagonal_recipe,
            recipe_islives_1D[i]
        )
    end

    for i in eachindex(possible_recipes_2D)
        map!(
            (upper, lower) -> upper == Symbol("$(typeof(possible_recipes_2D[i]))") || lower == Symbol("$(typeof(possible_recipes_2D[i]))"),
            graph,
            [:upper_recipe, :lower_recipe],
            recipe_islives_2D[i]
        )
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
                [marg_sym, :weights, recipe_islives_1D[k], :live_map],
                [primitive_symbols_1D[k]]
            ) do inputs, changed, cached
                coords, weights, recipe_islive, live_map = inputs
                recipe = BAT_MAKIE_RECIPES_1D[k]
                islive = recipe_islive && live_map[i, i]
                if islive
                    primitives = compute_plotting_primitives(coords, weights, recipe)
                    return (primitives,)
                else
                    empty_primitives = compute_plotting_primitives(nothing, nothing, recipe)
                    return (empty_primitives,)
                end
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
                    [marg_sym_2D, :weights, recipe_islives_2D[k], :live_map],
                    [primitive_symbols_2D[k]]
                ) do inputs, changed, cached
                    coords, weights, recipe_islive, live_map = inputs
                    recipe = BAT_MAKIE_RECIPES_2D[k]
                    islive = recipe_islive && (live_map[j, i] || live_map[i, j])

                    if islive
                        primitives = compute_plotting_primitives(coords, weights, recipe)
                        return (primitives,)
                    else
                        empty_primitives = compute_plotting_primitives(nothing, nothing, recipe)
                        return (empty_primitives,)
                    end
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
        graph[:lower_recipe]
    ) do idx, upper_recipe_sym, diagonal_recipe_sym, lower_recipe_sym
        matrix = Matrix{Any}(undef, n, n)
        upper_recipe = graph[upper_recipe_sym][]
        diagonal_recipe = graph[diagonal_recipe_sym][]
        lower_recipe = graph[lower_recipe_sym][]

        for i in 1:n
            diagonal_primitives = graph[primitive_symbol(diagonal_recipe, (i, i))][]
            diagonal_plotspecs = compose_plotspecs(diagonal_primitives, diagonal_recipe)

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
                upper_plotspecs = compose_plotspecs(upper_primitives, upper_recipe)

                ylims = graph[Symbol("axis_limits_$j")][]
                show_y_cosmetics = (j == n)
                matrix[i, j] = S.Axis(
                    plots=upper_plotspecs,
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
                lower_plotspecs = compose_plotspecs(lower_primitives, lower_recipe)

                ylims = graph[Symbol("axis_limits_$j")][]
                show_y_cosmetics = (j == 1)
                matrix[i, j] = S.Axis(
                    plots=lower_plotspecs,
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
    ui_layout = fig[2, 1] = GridLayout()

    options2D = [
        ("QuantileHist", Symbol(QuantileHist2D)),
        ("Hist", Symbol(Hist2D)),
    ]
    options1D = [
        ("QuantileHist", Symbol(QuantileHist1D)),
        ("Hist", Symbol(Hist1D)),
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

    ui_layout[1, 1] = Label(fig, "Upper Recipe")
    ui_layout[1, 2] = menu_upper
    ui_layout[1, 3] = Label(fig, "Diagonal Recipe")
    ui_layout[1, 4] = menu_diagonal
    ui_layout[1, 5] = Label(fig, "Lower Recipe")
    ui_layout[1, 6] = menu_lower

    ui_layout[1, 7] = Label(fig, "Current Idx")
    ui_layout[1, 8] = slider_curr_idx

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

