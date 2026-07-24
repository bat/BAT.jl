# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# A color that visually separates from `bg` in whichever direction actually
# increases contrast, without hardcoding one shade per theme: darkening a
# fixed amount reads fine starting from a light color but is imperceptible
# starting from an already-near-black one (0.10 darkened by 0.08 is still
# 0.02, visually identical) -- so this shifts *away* from whichever extreme
# `bg` is already closer to (darker if bg is light, lighter if bg is dark).
# Used twice in a row (see bat_theme()/bat_theme_dark()) to build a 3-step
# ladder -- page background, then the UI panel a step further, then the
# widgets on top of the panel a further step still -- each step using the
# *previous* step's own color as its `bg`, so the two gaps (bg-to-panel,
# panel-to-widget) are always identical regardless of amount.
function _panel_bg_color(bg, amount::Real=0.08)
    rgb = Colors.RGB(Makie.to_color(bg))
    luminance = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b
    delta = luminance > 0.5 ? -amount : amount
    return Colors.RGB(clamp(rgb.r + delta, 0, 1), clamp(rgb.g + delta, 0, 1), clamp(rgb.b + delta, 0, 1))
end

# Palette sample positions for the quantile-level color gradient shared by
# QuantileHist1D/2D and QuantileKDE1D/2D. range(0.05, 0.7, length=n) throws
# ArgumentError("endpoints differ") for n==1 (a single point can't have two
# distinct endpoints) -- unguarded in all four recipes, so a config with
# exactly one level surviving `filter(x -> 0<x<1, levels)` (a deliberate
# single-level levels=[0.68], or QuantileKDE2D's own unconditional extra
# threshold making even an empty levels come out at length 1) crashed the
# whole recompute. Confirmed via direct reproduction. n==0 is unaffected
# (range(...,0) is a valid empty range); n>=2 is unaffected too -- only n==1
# is special-cased here, to a single fixed midpoint palette value.
function _quantile_palette_positions(n::Integer)
    n <= 1 && return fill(0.375, n)
    return collect(range(0.05, 0.7, length=n))
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
