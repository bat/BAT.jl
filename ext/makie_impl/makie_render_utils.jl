# This file is a part of BAT.jl, licensed under the MIT License (MIT).

_rel_luminance(rgb::Colors.RGB) = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b

# Shifts `bg` away from whichever extreme it is already closer to (darker if light, lighter if dark); used twice to build the background -> panel -> widget ladder.
function _panel_bg_color(bg, amount::Real = 0.08)
    rgb = Colors.RGB(Makie.to_color(bg))
    delta = _rel_luminance(rgb) > 0.5 ? -amount : amount
    return Colors.RGB(clamp(rgb.r + delta, 0, 1), clamp(rgb.g + delta, 0, 1), clamp(rgb.b + delta, 0, 1))
end

# Warning/error status text color: deep red on light backgrounds, a lighter warm red on dark.
function _status_text_color(bg)
    rgb = Colors.RGB(Makie.to_color(bg))
    return _rel_luminance(rgb) > 0.5 ? Colors.RGB(0.72, 0.05, 0.05) : Colors.RGB(1.0, 0.45, 0.35)
end

# Fixed colors for all four Quantile* recipes' credible-region bands (widest region first, tightest last); supports up to length(_QUANTILE_LEVEL_COLORS) levels.
const _QUANTILE_LEVEL_COLORS = [
    RGBAf(1.0, 0.0, 0.0, 1.0),
    RGBAf(1.0, 1.0, 0.0, 1.0),
    RGBAf(0.462, 0.933, 0.0, 1.0),
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

# Carrier for the vsel picker's construction inputs. apply_vsel! is a deliberately type-erased Function, not a NamedTuple field: a NamedTuple would embed
# each call site's closure type (per-site MethodInstances); one concrete type shares one compiled chain, at one dynamic dispatch per click.
struct PickerInfo
    N::Int
    N_max::Int
    initial_vsel::Vector{Int}
    apply_vsel!::Function
end

# Display gate: the precompile workload runs the entry points headless through
# this, stashing the figure to drive its widgets; the workload resets both refs.
const _SUPPRESS_DISPLAY = Ref(false)
const _SUPPRESSED_FIG = Ref{Any}(nothing)

function _maybe_display(fig)
    if _SUPPRESS_DISPLAY[]
        _SUPPRESSED_FIG[] = fig
    else
        display(fig)
    end
    return nothing
end

# Latches once per session on the first successful warmup; a missing backend skips gracefully without latching, so a later call after loading still warms.
const _SHADERS_WARMED = Ref(false)

function warmup_makie_shaders()
    _SHADERS_WARMED[] && return nothing
    ismissing(Makie.current_backend()) && return nothing
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
    _SHADERS_WARMED[] = true
    return nothing
end
