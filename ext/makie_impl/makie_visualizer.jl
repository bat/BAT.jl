
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


function BATMakieVisualizer(
    recipes::NamedTuple=(upper=QuantileHist2D(), diagonal=Hist1D(), lower=Hist2D()),
    vsel::Vector{Int64}=[1, 2, 3], # Figure out default values. Pass samples to determine number of parameters?
    N_max::Int64=5,
    n_batch::Int64=10
)
    graph = ComputeGraph()
    gridspec = S.GridLayout()

    # Set up the UI observables that the user will interact with
    # TODO: Figure out where to define the update logic. Requires the populated compute graph, which will be manipulated on UI control changes
    recipe_dropdown = nothing
    vsel_selector = nothing
    n_batch_slider = nothing

    ui_controls = (
        recipe_dropdown=recipe_dropdown,
        vsel_selector=vsel_selector,
        n_batch_slider=n_batch_slider
    )

    vis = BATMakieVisualizer(
        recipes,
        vsel,
        N_max,
        n_batch,
        graph,
        gridspec,
        ui_controls
    )

    return vis
end


function Makie.convert_arguments(
    ::Type{<:AbstractPlot},
    vis::BATMakieVisualizer
)

    return nothing
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

function init_visualizer!!(
    vis::BATMakieVisualizer,
    smpls::DensitySampleVector,
    sampling::SA
) where {SA<:AbstractSamplingAlgorithm}

    (; recipes, vsel, N_max, n_batch, graph, gridspec) = vis

    vs = varshape(smpls)

    # TODO: Think about whether or not the more expressive vsel symbol approach is desired, or integers are sufficient
    # idxs = vsel isa AbstractVector{<:Int64} ? vsel : reduce(vcat, asindex.(Ref(vs), vsel))
    idxs = filter(<=(totalndof(vs)), vsel)

    graph = _populate_compute_graph!(
        graph,
        smpls,
        idxs,
        recipes,
        N_max,
        n_batch
    )

    return vis
end


using Makie: ComputeGraph, add_input!, register_computation!


function _populate_compute_graph!(
    graph::ComputeGraph,
    smpls::DensitySampleVector,
    idxs::Vector{Integer},
    recipes::NamedTuple,
    n::Integer,
    n_batch_init
)

    batch = smpls[end-n_batch_init:end]

    add_input!(graph, :samples, smpls)
    map!(
        smpls -> length(smpls),
        graph,
        :samples,
        :current_idx
    )
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
    add_input!(graph, :upper_recipe, recipe_symbol(recipes.upper))
    add_input!(graph, :diagonal_recipe, recipe_symbol(recipes.diagonal))
    add_input!(graph, :lower_recipe, recipe_symbol(recipes.lower))

    possible_recipes_1D = recipe_symbol.(BAT_MAKIE_RECIPES_1D)
    possible_recipes_2D = recipe_symbol.(BAT_MAKIE_RECIPES_2D)

    recipe_islives_1D = islive_symbol.(BAT_MAKIE_RECIPES_1D)
    recipe_islives_2D = islive_symbol.(BAT_MAKIE_RECIPES_2D)

    for i in eachindex(possible_recipes_1D)
        map!(
            recipe -> typeof(recipe) == typeof(possible_recipes_1D[i]),
            graph,
            :diagonal_recipe,
            recipe_islives_1D[i]
        )
    end

    for i in eachindex(possible_recipes_2D)
        map!(
            (upper, lower) -> typeof(upper) == typeof(possible_recipes_2D[i]) || typeof(lower) == typeof(possible_recipes_2D[i]),
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
            marg_sym_2D = marg_symbol((i, j))
            map!(
                (smpls, vsel_map, current_idx) -> view(smpls.v.data, [vsel_map[i, j]...], 1:current_idx),
                graph,
                [:samples, :vsel_map, :current_idx],
                marg_sym_2D
            )

            primitive_symbols_2D = [primitive_symbol(recipe, (i, j)) for recipe in BAT_MAKIE_RECIPES_2D]

            for k in eachindex(primitive_symbols_2D)
                register_computation!(graph,
                    [marg_sym_2D, :weights, recipe_islives_2D[k], :live_map],
                    [primitive_symbols_2D[k]]
                ) do inputs, changed, cached
                    coords, weights, recipe_islive, live_map = inputs
                    recipe = BAT_MAKIE_RECIPES_2D[k]
                    islive = recipe_islive && live_map[i, j]

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

function build_gridspec(
    graph::ComputeGraph,
    idxs::Vector{Int64},
    recipes::NamedTuple,
    n::Int64
)

    plotspecs = Matrix{Observable}(undef, n, n)

    for i in 1:n

        for j in 1:n

            active_recipes = 1
            plotspecs = compose_plotspecs()

        end
    end

    return nothing
end


# Implement methods for each plotting recipe



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

