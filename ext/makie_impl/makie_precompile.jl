# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# Exercise static Matrix-backed and live ElasticMatrix-backed samples through the public entry
# points. Native GL compilation requires GLMakie and is not covered here.

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

    # Exercise repeated selection changes and rejection at the grid-size limit.
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

    # Controls start collapsed; exercise expansion and collapse.
    button = filter(x -> x isa Button, fig.content)[1]
    notify(button.clicks)
    notify(button.clicks)

    resize!(fig, 500, 650)

    # Apply draft changes to exercise window computation, then restore All.
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
        # Four parameters exceed the default three-variable grid.
        objective = NamedTupleDist(a = Uniform(-2, 2),
            b = product_distribution([Uniform(-2, 2), Uniform(-2, 2)]), c = Uniform(-2, 2))

        # Use two chains for Gelman-Rubin and a large buffer to avoid backpressure during
        # compilation.
        vis = BATVisualizer(BATMakieVisualization(max_buffered = 10^6))
        res = bat_sample(objective, TransformedMCMC(nsteps = 200, nchains = 2), BATContext(visualizer = vis))
        samples = res.result
        live_fig = BAT.visualizer_figure(vis)
        isnothing(live_fig) && error("Makie precompile workload: completed live Figure is unavailable")
        # Support must survive the EvaluatedMeasure wrapper, including initialization and
        # burn-in.
        graph = _live_graph(vis)
        graph[:support_lo][] == fill(-2.0, 4) && graph[:support_hi][] == fill(2.0, 4) ||
            error("Makie precompile workload: original target support was not forwarded")

        _drive_figure_widgets!(live_fig)

        # The bounded observations admit reflection without clamping samples to artificial
        # support.
        for recipe_1d in (KDE1D, QuantileKDE1D)
            _live_graph_update!(vis; diagonal_recipe = recipe_1d)
        end
        for recipe_2d in (KDE2D, QuantileKDE2D)
            _live_graph_update!(vis; upper_recipe = recipe_2d)
        end
        _live_graph_update!(vis; support_lo = Float64[], support_hi = Float64[])
        _live_graph_update!(vis; support_lo = fill(-2.0, 4), support_hi = fill(2.0, 4))
        # Live callback failures are contained, so check that the updates succeeded.
        BAT.visualizer_figure(vis) === live_fig ||
            error("Makie precompile workload: live Figure became unavailable during updates")

        static_fig = bat_makie_plot(samples; display = false)
        _drive_figure_widgets!(static_fig)

        # Exercise the dark theme and explicit KDE support.
        dark_fig = bat_makie_plot(samples; display = false, dark = true,
            support = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)])
        # The undisplayed figure has no resolved layout. Select menus by their options to cover
        # both ordinary and transposed KDE composition.
        for m in filter(x -> x isa Menu, dark_fig.content)
            for r in (KDE1D, QuantileKDE1D, KDE2D, QuantileKDE2D)
                k = findfirst(o -> o[2] == r, m.options[])
                isnothing(k) || (m.i_selected[] = k)
            end
        end

        # The ordinary Makie path returns a GridLayoutSpec.
        Makie.convert_arguments(Makie.Plot{Makie.plot}, samples)
    finally
        try
            # Clean up each owned figure independently, preserving the original workload
            # exception.
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
                BAT._safe_finish_visualizer!(vis; success = false)
            end
        finally
            _SUPPRESS_DISPLAY[] = suppress_display
        end
    end
    return nothing
end
