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

# Fixed, hand-picked colors for all four quantile-level recipes' credible-
# region bands (QuantileHist1D/2D in makie_hist.jl, QuantileKDE1D/2D in
# makie_kde.jl) -- replaces a previous cgrad(config.colormap)-sampled
# continuous gradient, per explicit user request. Shared here (rather than
# defined once per file) because all four recipes' own "i" loop-index
# convention already agrees on what index 1 vs index 3 means (confirmed
# directly, not assumed): ascending index -> loosest/widest credible region
# first, narrowest/tightest last, for every one of the four -- so plugging
# this same lookup into all four preserves a single consistent visual
# convention across the whole quantile-recipe family without needing to
# special-case or reverse the order anywhere.
#
# Only supports up to length(_QUANTILE_LEVEL_COLORS) distinct levels -- the
# default config (3 levels after filtering) fits exactly; an explicit, clear
# error rather than a BoundsError if a caller ever configures more.
const _QUANTILE_LEVEL_COLORS = [
    RGB(1.0, 0.0, 0.0),
    RGB(1.0, 1.0, 0.0),
    RGB(0.462, 0.933, 0.0),
]

function _quantile_level_color(i::Integer)
    i <= length(_QUANTILE_LEVEL_COLORS) || throw(ArgumentError(
        "The quantile-level recipes only support up to " *
        "$(length(_QUANTILE_LEVEL_COLORS)) credible levels with the " *
        "current fixed color list (_QUANTILE_LEVEL_COLORS) -- got level " *
        "index $i. Reduce the number of configured `levels`."
    ))
    return _QUANTILE_LEVEL_COLORS[i]
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
