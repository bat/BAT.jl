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
# Also exercises every *widget's own callback* -- not just the graph-level
# computation a widget eventually triggers via `update!`, which is a
# different, separately-compiled code path. `on(widget.attribute) do ... end`
# only *registers* a closure; Julia doesn't compile its body until the
# Observable actually notifies, which never happened here before even though
# the downstream graph computation was warmed via direct `update!` calls.
# Confirmed via direct investigation (before writing this) that the previous
# version of this workload left several real, user-facing interactions cold:
# the vsel picker was never constructed at all (`picker_info=nothing`), vsel
# was only ever exercised fully-active (nothing exercised deselecting a
# variable), the controls panel never actually expanded (only its initial
# collapsed state fired), no real window resize ever happened, and every
# Menu/Toggle/Slider/Button's own `on(...)` closure was never invoked. All of
# that is closed below.
#
# Verified empirically (before writing this) that none of it needs a concrete
# backend (CairoMakie/GLMakie) active: Figure/Axis/SpecApi construction and
# Observable-driven PlotSpec reconciliation all work fine without one -- only
# actual rasterization (colorbuffer/display) does, and that's deliberately
# never called here; this also holds for `resize!` (a pure Observable/layout-
# size mutation, no rendering involved) and for setting/`notify`-ing widget
# attributes directly, none of which touch the backend. filter=true's
# fallback path is intentionally not exercised: it's a construction-time
# config field with no interactive UI toggle, so it's not part of the
# "dropdown latency" this targets.
function _makie_precompile_workload()
    # 4 total dof (1 from `a`, 2 from `b`, 1 from `c`) against N_max=3 below --
    # matching BATMakieVisualization()'s own real default (N_max=3, see
    # src/extdefs/makie_defs.jl) is deliberate, not incidental: an earlier
    # version of this workload used N_max=2 ("smallest grid that still has a
    # diagonal, an upper, and a lower cell", for a faster precompile) and
    # n_dof=3 -- and while that DID warm the shrink-to-partial-selection
    # (phantom-row) code path, it only warmed it *for N_max=2 specifically*.
    # Confirmed directly (via @time on a real interactive session after
    # precompiling) that a real BATMakieVisualization()'s default N_max=3
    # still paid a full fresh-compilation cost on its first checkbox click
    # despite that -- i.e. this code path's compiled specialization is
    # sensitive to N_max, and precompiling only the minimal grid size doesn't
    # transfer to the size real users actually get by default. Still keeping
    # n_dof > N_max (4 > 3) preserves the picker's N_max-exceeded cap-check
    # branch too (see the interaction block below).
    objective = NamedTupleDist(a=Normal(0, 1), b=MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]), c=Normal(0, 1))
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
    # The real default configs, shrunk for precompile speed (fewer bins,
    # fewer levels, fewer PDF curve points) via merge -- the magic level
    # values this used to hardcode were just cdf.(Chi(2), 1:2).
    triagonal_config = merge(_default_makie_triagonal_config(),
        (nbins=(20, 20), levels=cdf.(Chi(2), 1:2)))
    diagonal_config = merge(_default_makie_diagonal_config(),
        (nbins=20, levels=cdf.(Chi(2), 1:2), npoints_pdf=30))
    N_max = 3 # matches BATMakieVisualization()'s own real default -- see comment above

    graph, n_dof = _setup_static_graph(samples, recipes, [1, 2, 3], N_max, triagonal_config, diagonal_config)

    gridlayout = _init_gridlayout(graph, N_max)

    # Real picker_info (mirroring the static bat_makie_plot path's own
    # construction, ext/makie_impl/makie_plot_samples.jl) rather than
    # `nothing` -- n_dof=4 > N_max=3 here, a genuine "more variables than
    # grid slots" scenario, so the picker's own N_max-exceeded cap-check
    # branch (see the interaction block below) gets exercised on real data,
    # not just a degenerate always-fits case.
    picker_info = (
        N=n_dof,
        N_max=N_max,
        initial_vsel=[1, 2, 3],
        apply_vsel! = new_vsel -> _apply_vsel_to_graph!(graph, n_dof, N_max, new_vsel),
    )
    # has_chain_info=true: real MCMC samples from bat_sample above always
    # carry chain identity, matching what the real static/live callers pass
    # for genuine MCMC data (see _samples_have_chain_ids).
    built = _build_fig(graph, gridlayout, picker_info; has_chain_info=true, has_trace_info=true) # already wires plot(fig[1,1], gridlayout)
    fig = built.fig

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
    # ChainScatter2D isn't in BAT_MAKIE_RECIPES_2D (see _init_compute_graph's
    # comment on why it's registered separately), so it needs its own
    # explicit precompile cycle here rather than being covered by the loop
    # above.
    update!(graph, upper_recipe=ChainScatter2D)
    gridlayout[]
    update!(graph, lower_recipe=ChainScatter2D)
    gridlayout[]
    update!(graph, show_stats_diag=true, show_stats_upper=true, show_stats_lower=true)
    gridlayout[]
    # Trace2D is likewise not in BAT_MAKIE_RECIPES_2D (an always-live overlay,
    # not a selectable main recipe -- see its own determine_recipe_status
    # override) -- exercise its toggle the same way show_stats_* is above.
    update!(graph, show_trace_upper=true, show_trace_lower=true)
    gridlayout[]

    # Boundary-corrected KDE path: non-empty support bounds both change the
    # recipes' config NamedTuple type (the merge adds a :support field --
    # a different compiled specialization per recipe method) and take the
    # reflection branch in _weighted_kde1d/_weighted_kde2d -- both cold
    # without this. The bounds needn't match the objective's actual
    # (unbounded) prior; this only warms code, not visuals. Reset to the
    # empty = no-correction default afterwards so the widget-warming below
    # runs under the common uncorrected config type again.
    update!(graph, support_lo=fill(-2.0, n_dof), support_hi=fill(2.0, n_dof))
    for recipe_1d in (KDE1D, QuantileKDE1D)
        update!(graph, diagonal_recipe=recipe_1d)
        gridlayout[]
    end
    for recipe_2d in (KDE2D, QuantileKDE2D)
        update!(graph, upper_recipe=recipe_2d)
        gridlayout[]
    end
    update!(graph, support_lo=Float64[], support_hi=Float64[])
    gridlayout[]

    # ---- Widget-callback warming --------------------------------------
    # Everything above mutates the graph *directly*, which warms the graph-
    # level computation but not the widgets' own `on(...)` closures -- those
    # are separate compiled methods that only run once actually invoked (see
    # this function's header comment). Below, drive each real widget through
    # its own reactive attribute so its closure body compiles here instead of
    # on a real user's first interaction.

    menus = filter(x -> x isa Menu, fig.content)
    for m in menus
        # The closure body doesn't depend on *which* option was chosen, only
        # that it runs -- notify() re-fires it with the current selection,
        # no need to actually change it.
        notify(m.selection)
    end

    toggles = filter(x -> x isa Toggle, fig.content)
    for t in toggles
        t.active[] = true
        gridlayout[]
        t.active[] = false
        gridlayout[]
    end

    # Vsel picker checkboxes: exercise both branches of the interactive
    # handler -- a valid shrink (n_active < N_max, the phantom-row/inactive-
    # cell code path a partial selection hits) and the N_max-exceeded
    # cap-check rejection (active is back at the N_max cap after the restore
    # below, so the next checkbox checked is guaranteed to overflow it).
    # Found via findfirst rather than a fixed index since the checkbox <->
    # (i,j) mapping isn't exposed outside _build_vsel_picker! -- exactly
    # which pair gets toggled doesn't matter, only that both code paths run.
    #
    # The shrink/restore cycle runs a few times, not once: confirmed directly
    # (via @time on a real interactive session) that this specific
    # transition -- an idxs change triggering _init_gridlayout's full grid
    # rebuild -- takes several repetitions to fully stabilize (observed
    # ~5s -> ~1.6s -> ~1.5s -> near-instant across successive toggles of the
    # *same* checkbox), not just one. A single precompile exercise measurably
    # helps but doesn't fully converge; a few repetitions here front-loads
    # more of that convergence into precompile time instead of a real user's
    # first several clicks.
    checkboxes = filter(x -> x isa Checkbox, fig.content)
    if !isempty(checkboxes)
        shrink_idx = findfirst(cb -> cb.checked[], checkboxes)
        if !isnothing(shrink_idx)
            for _ in 1:4
                checkboxes[shrink_idx].checked[] = false
                gridlayout[]
                checkboxes[shrink_idx].checked[] = true # restore to the N_max cap
                gridlayout[]
            end
        end
        overflow_idx = findfirst(cb -> !cb.checked[], checkboxes)
        if !isnothing(overflow_idx)
            checkboxes[overflow_idx].checked[] = true # rejected: already at N_max
            gridlayout[]
        end
    end

    # Collapse/expand button: exercises _set_block_visible!'s restore-to-
    # visible branch (never reached by controls_visible's own construction-
    # time update=true firing, which only ever sees the initial collapsed
    # value), and therefore rescale_picker!() too -- otherwise cold until a
    # real user's first ☰ click.
    button = filter(x -> x isa Button, fig.content)[1]
    notify(button.clicks)

    # A real resize -- exercises GridLayoutBase's own Auto()/Aspect(1,1)
    # relayout solve (row 1's corner grid absorbing whatever the anchor/
    # controls rows don't need) against a genuine geometry change.
    resize!(fig, 500, 650)

    # Step-range slider (exists whenever samples are registered, which they
    # are here) -- drive it through a genuine window change to warm the
    # per-walker step-window path.
    sliders = filter(x -> x isa IntervalSlider, fig.content)
    if !isempty(sliders)
        ms = graph[:max_step][]
        sliders[1].interval[] = (max(1, ms ÷ 4), max(1, ms ÷ 2))
    end

    # One cheap pass under the dark theme: both real entry points wrap
    # construction in with_theme(dark ? bat_theme_dark() : bat_theme()), but
    # everything above ran under the ambient (light) theme only -- a user's
    # first dark session previously paid the dark-specific attribute
    # application compilation live, on top of the unthemed-checkbox issue
    # this workload could never have caught.
    with_theme(bat_theme_dark()) do
        _build_fig(graph, gridlayout, picker_info; has_chain_info=true, has_trace_info=true)
        gridlayout[]
    end

    return nothing
end
