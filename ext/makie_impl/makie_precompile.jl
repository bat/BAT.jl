# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Precompile workload: drives the real user entry points. Obtain the completed
# live Figure through its accessor; static calls return theirs with display=false.
# The live leg warms the ElasticMatrix sample flavor (vs static Matrix-backed);
# GLMakie's render-stack compilation needs GLMakie loaded and is not covered here.

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

    # Draft changes alone do not exercise window computation. Require the
    # expected controls and drive Apply/All on both live and static data.
    slider = only(filter(x -> x isa IntervalSlider, fig.content))
    r = slider.range[]
    slider.interval[] = (max(first(r), fld(last(r), 4)), max(first(r), fld(last(r), 2)))
    apply = only(filter(x -> x isa Button && x.label[] == "Apply", fig.content))
    all_button = only(filter(x -> x isa Button && x.label[] == "All", fig.content))
    notify(apply.clicks)
    notify(all_button.clicks)
    return nothing
end

function _makie_precompile_workload()
    suppress_display = _SUPPRESS_DISPLAY[]
    _SUPPRESS_DISPLAY[] = true
    vis = nothing
    static_fig = nothing
    dark_fig = nothing
    try
        # 4 total dof against default N_max=3 keeps the cap-exceeded branch reachable (specializations are N_max-sensitive).
        objective = NamedTupleDist(a=Uniform(-2, 2),
            b=product_distribution([Uniform(-2, 2), Uniform(-2, 2)]), c=Uniform(-2, 2))

        # ---- Live leg: a real visualizer-attached run -------------------
        # nchains=2 is the minimum passing Gelman-Rubin; max_buffered is far above the sample count so the worker never blocks on backpressure.
        vis = BATVisualizer(BATMakieVisualization(max_buffered=10^6))
        res = bat_sample(objective, TransformedMCMC(nsteps=200, nchains=2), BATContext(visualizer=vis))
        samples = res.result
        live_fig = BAT.visualizer_figure(vis)
        isnothing(live_fig) && error("Makie precompile workload: completed live Figure is unavailable")
        # The actual original target must supply support through the live
        # EvaluatedMeasure wrapper, including initialization and burn-in rows.
        graph = _live_graph(vis)
        graph[:support_lo][] == fill(-2.0, 4) && graph[:support_hi][] == fill(2.0, 4) ||
            error("Makie precompile workload: original target support was not forwarded")

        # Widget sweep against the LIVE (ElasticMatrix-backed) data flavor.
        _drive_figure_widgets!(live_fig)

        # Real bounded observations exercise the reflection branch. Explicit
        # support may not clamp unbounded samples into an artificial interval.
        for recipe_1d in (KDE1D, QuantileKDE1D)
            _live_graph_update!(vis; diagonal_recipe=recipe_1d)
        end
        for recipe_2d in (KDE2D, QuantileKDE2D)
            _live_graph_update!(vis; upper_recipe=recipe_2d)
        end
        _live_graph_update!(vis; support_lo=Float64[], support_hi=Float64[])
        _live_graph_update!(vis; support_lo=fill(-2.0, 4), support_hi=fill(2.0, 4))
        # Live callbacks isolate presentation failures from sampling. Such a
        # failure must not silently count as successful precompile coverage.
        BAT.visualizer_figure(vis) === live_fig ||
            error("Makie precompile workload: live Figure became unavailable during updates")

        # ---- Static legs: the real entry points -------------------------
        static_fig = bat_makie_plot(samples; display=false)
        # Same widget sweep against the STATIC (Matrix-backed) flavor.
        _drive_figure_widgets!(static_fig)

        # Dark theme and the support kwarg in one build (dark changes attribute values, not compiled types); only KDE recipes are support-sensitive.
        dark_fig = bat_makie_plot(samples; display=false, dark=true,
            support=[(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)])
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
        try
            # Each successfully returned static figure belongs to this workload.
            # Cleanup/diagnostic failures must not skip the other figure or live
            # disposal, nor replace the original workload exception.
            for fig in (static_fig, dark_fig)
                isnothing(fig) && continue
                try
                    empty!(fig)
                catch err
                    BAT._visualizer_cleanup_diagnostic(:precompile_static_figure,
                        (err, catch_backtrace()))
                end
            end
            if !isnothing(vis)
                # No live task, timer or retained view may enter the package image.
                BAT._safe_request_visualizer_stop!(vis)
                try
                    BAT.dispose_visualizer!(vis)
                catch err
                    _record_live_cleanup_error!(vis, err, catch_backtrace())
                end
                BAT._safe_finish_visualizer!(vis; success=false)
            end
        finally
            _SUPPRESS_DISPLAY[] = suppress_display
        end
    end
    return nothing
end
