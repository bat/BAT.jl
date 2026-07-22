# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Exercises every recipe's full "select this from the dropdown" code path once
# at package-precompile time -- compute_plotting_primitives/compose_plotspecs
# for every 1D/2D recipe (both the incremental and dead-cell paths), the
# ComputePipeline TypedEdge construction for each graph node, and Makie's
# SpecApi PlotSpec-reconciliation dispatch for each distinct plot type -- so
# that Julia-level method compilation doesn't have to happen the first time a
# user actually picks a given recipe interactively (previously only the
# lowest-level Makie plotting calls were warmed via warmup_makie_shaders,
# which doesn't touch any of this).
#
# Verified empirically (before writing this) that none of it needs a concrete
# backend (CairoMakie/GLMakie) active: Figure/Axis/SpecApi construction and
# Observable-driven PlotSpec reconciliation all work fine without one -- only
# actual rasterization (colorbuffer/display) does, and that's deliberately
# never called here. filter=true's fallback path is intentionally not
# exercised: it's a construction-time config field with no interactive UI
# toggle, so it's not part of the "dropdown latency" this targets.
function _makie_precompile_workload()
    objective = NamedTupleDist(a=Normal(0, 1), b=MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]))
    # nchains=1 breaks Gelman-Rubin convergence checking (BrooksGelmanConvergence
    # needs >=2 chains to compute between-chain variance) -- burn-in then never
    # "converges" and aborts after exhausting its cycle budget. 2 is the minimum
    # that actually works; nsteps is otherwise kept small since this only needs
    # to produce enough samples to exercise the recipes below, not to converge
    # to anything meaningful.
    samplingalg = TransformedMCMC(nsteps=200, nchains=2)
    context = BATContext()
    res = bat_sample(objective, samplingalg, context)
    samples = res.result

    recipes = (upper=QuantileHist2D, diagonal=Hist1D, lower=Hist2D)
    triagonal_config = (
        weights=nothing, nsigma=1.0, nbins=(20, 20), closed=:left, normalization=:pdf,
        levels=[0.3934693402873665, 0.8646647167633873], filter=false, colormap=:inferno,
        alpha=1.0, rev=false, threshold=nothing, markersize=2.0
    )
    diagonal_config = (
        weights=nothing, nsigma=1.0, nbins=20, closed=:left, normalization=:pdf,
        levels=[0.3934693402873665, 0.8646647167633873], filter=false, colormap=:inferno,
        alpha=1.0, y_ebars=0.0, filled_pdf=true, npoints_pdf=30, rev=false
    )
    N_max = 2 # smallest grid that still has a diagonal, an upper, and a lower cell

    graph = _init_compute_graph(recipes, triagonal_config, diagonal_config, N_max)

    unshaped_samples = unshaped.(samples)
    samples_graph = graph[:samples][]
    push!(samples_graph, [unshaped_samples])
    update!(graph, samples=samples_graph)

    current_idxs_graph = graph[:current_idxs][]
    push!(current_idxs_graph, [length(samples)])
    update!(graph, current_idxs=current_idxs_graph)

    n_dof = totalndof(varshape(samples))
    domain_lo, domain_hi = _domain_from_samples(unshaped_samples.v.data, n_dof)
    update!(graph, domain_lo=domain_lo, domain_hi=domain_hi)
    update!(graph, idxs=_clamp_vsel([1, 2], n_dof, N_max))

    gridlayout = _init_gridlayout(graph, N_max)
    fig = _build_fig(graph, gridlayout, nothing) # already wires plot(fig[1,1], gridlayout)

    for recipe_1d in BAT_MAKIE_RECIPES_1D
        update!(graph, diagonal_recipe=typeof(recipe_1d))
        gridlayout[]
    end
    for recipe_2d in BAT_MAKIE_RECIPES_2D
        update!(graph, upper_recipe=typeof(recipe_2d))
        gridlayout[]
        update!(graph, lower_recipe=typeof(recipe_2d))
        gridlayout[]
    end
    update!(graph, show_stats_diag=true, show_stats_upper=true, show_stats_lower=true)
    gridlayout[]

    return nothing
end
