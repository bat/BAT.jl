# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Precompile workload: drives the REAL user entry points through the
# _maybe_display suppression gate, so coverage can't drift. The live leg warms
# the ElasticMatrix sample flavor (vs static Matrix-backed); GLMakie's own
# render-stack compilation cannot be covered here (needs GLMakie loaded).

# Drives every interactive widget the way real input would: menus, toggles, vsel checkboxes, collapse button, resize, step slider.
function _drive_figure_widgets!(fig)
    for m in filter(x -> x isa Menu, fig.content)
        for k in eachindex(m.options[])
            m.i_selected[] = k
        end
    end

    for t in filter(x -> x isa Toggle, fig.content)
        t.active[] = true
        t.active[] = false
    end

    # Shrink/restore x4 stabilizes the idxs-change grid rebuild; overflow click hits the N_max-exceeded rejection branch.
    checkboxes = filter(x -> x isa Checkbox, fig.content)
    if !isempty(checkboxes)
        shrink_idx = findfirst(cb -> cb.checked[], checkboxes)
        if !isnothing(shrink_idx)
            for _ in 1:4
                checkboxes[shrink_idx].checked[] = false
                checkboxes[shrink_idx].checked[] = true
            end
        end
        overflow_idx = findfirst(cb -> !cb.checked[], checkboxes)
        isnothing(overflow_idx) || (checkboxes[overflow_idx].checked[] = true)
    end

    # Expand (restore-to-visible branch, unreached by construction's initial collapsed state), then collapse back.
    button = filter(x -> x isa Button, fig.content)[1]
    notify(button.clicks)
    notify(button.clicks)

    # Warms the relayout solve against a genuine geometry change.
    resize!(fig, 500, 650)

    # A genuine window change warms the per-walker step-window path.
    for isl in filter(x -> x isa IntervalSlider, fig.content)
        r = isl.range[]
        isl.interval[] = (max(first(r), fld(last(r), 4)), max(first(r), fld(last(r), 2)))
    end
    return nothing
end

function _makie_precompile_workload()
    _SUPPRESS_DISPLAY[] = true
    try
        # 4 total dof against default N_max=3 keeps the cap-exceeded branch reachable (specializations are N_max-sensitive).
        objective = NamedTupleDist(a=Normal(0, 1), b=MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]), c=Normal(0, 1))

        # ---- Live leg: a real visualizer-attached run -------------------
        # nchains=2 is the minimum passing Gelman-Rubin; max_buffered is far above the sample count so the worker never blocks on backpressure.
        vis = BATVisualizer(BATMakieVisualization(max_buffered=10^6))
        res = bat_sample(objective, TransformedMCMC(nsteps=200, nchains=2), BATContext(visualizer=vis))
        samples = res.result
        live_fig = _SUPPRESSED_FIG[]
        live_graph = vis.content.graph

        # Widget sweep against the LIVE (ElasticMatrix-backed) data flavor.
        _drive_figure_widgets!(live_fig)

        # Non-empty support changes the recipes' config NamedTuple type and takes the KDE reflection branch.
        update!(live_graph, support_lo=fill(-2.0, 4), support_hi=fill(2.0, 4))
        for recipe_1d in (KDE1D, QuantileKDE1D)
            update!(live_graph, diagonal_recipe=recipe_1d)
        end
        for recipe_2d in (KDE2D, QuantileKDE2D)
            update!(live_graph, upper_recipe=recipe_2d)
        end
        update!(live_graph, support_lo=Float64[], support_hi=Float64[])

        # ---- Static legs: the real entry points -------------------------
        bat_makie_plot(samples)
        static_fig = _SUPPRESSED_FIG[]
        # Same widget sweep against the STATIC (Matrix-backed) flavor.
        _drive_figure_widgets!(static_fig)

        # Dark theme and the support kwarg in one build (dark changes attribute values, not compiled types); only KDE recipes are support-sensitive.
        bat_makie_plot(samples; dark=true, support=[(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)])
        dark_fig = _SUPPRESSED_FIG[]
        # Menus identified by their option lists, NOT by layout position: the
        # never-displayed figure has no meaningful computedbboxes to sort by.
        # Driving every menu that offers a given KDE recipe covers both the
        # upper (plain) and lower (transposed) compose variants.
        for m in filter(x -> x isa Menu, dark_fig.content)
            for r in (KDE1D, QuantileKDE1D, KDE2D, QuantileKDE2D)
                k = findfirst(o -> o[2] == r, m.options[])
                isnothing(k) || (m.i_selected[] = k)
            end
        end

        # The embed path -- returns a bare GridLayoutSpec, no figure to drive.
        Makie.convert_arguments(Makie.Plot{Makie.plot}, samples)
    finally
        # Reset the display gate; a Figure left in the stash ref would be
        # serialized into the package image.
        _SUPPRESS_DISPLAY[] = false
        _SUPPRESSED_FIG[] = nothing
    end
    return nothing
end
