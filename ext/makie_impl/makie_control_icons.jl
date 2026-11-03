# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function _rounded_control_triangle(vertices; radius = 3.0f0)
    points = Point2f[]
    for i in eachindex(vertices)
        p = vertices[i]
        previous = vertices[mod1(i - 1, length(vertices))]
        following = vertices[mod1(i + 1, length(vertices))]
        a = p + radius * normalize(previous - p)
        b = p + radius * normalize(following - p)
        for t in range(0.0f0, 1.0f0; length = 7)
            push!(points, (1 - t)^2 * a + 2t * (1 - t) * p + t^2 * b)
        end
    end
    return points
end

function _control_icon_points(kind::Symbol)
    if kind === :upper
        return _rounded_control_triangle(Point2f[(4, 20), (20, 20), (20, 4)])
    elseif kind === :lower
        return _rounded_control_triangle(Point2f[(4, 20), (4, 4), (20, 4)])
    elseif kind === :diagonal
        return Point2f[(5, 19), (19, 5)]
    elseif kind === :menu
        return Point2f[(5, 18), (19, 18), (NaN, NaN), (5, 12), (19, 12),
            (NaN, NaN), (5, 6), (19, 6)]
    elseif kind === :check
        return Point2f[(5, 12), (10, 7), (19, 18)]
    elseif kind === :close
        return Point2f[(6, 6), (18, 18), (NaN, NaN), (6, 18), (18, 6)]
    elseif kind === :reset
        arc = Point2f[Point2f(12, 12) + 8 * Point2f(cos(t), sin(t))
            for t in range(0.8pi, -0.85pi; length = 40)]
        return vcat(arc, Point2f[(NaN, NaN), (4, 21), (4, 15), (10, 15)])
    end
    throw(ArgumentError("Unknown control icon: $kind"))
end

function _control_icon!(block, kind; filled = false, size = 20.0)
    icon = kind isa Observable ? kind : Observable(kind)
    points = lift(block.blockscene, block.layoutobservables.computedbbox, icon) do bounds, value
        center = Point2f(bounds.origin + bounds.widths / 2)
        [center + (size / 24) * (p - Point2f(12, 12)) for p in _control_icon_points(value)]
    end
    color = block isa Button ? block.labelcolor : Makie.theme(block.blockscene, :textcolor)
    visible = block isa Button ? block.blockscene.visible : block.visible
    plot = if filled
        poly!(block.blockscene, points; color, visible, strokewidth = 0, inspectable = false)
    else
        lines!(block.blockscene, points; color, visible, linewidth = 1.8, linecap = :round,
            joinstyle = :round, inspectable = false)
    end
    # Native Buttons draw their background in a child scene. Match their text's
    # explicit foreground depth so glyphs cannot disappear behind that background.
    translate!(plot, 0, 0, 2)
    return icon
end

function _control_tooltip!(fig, block, text; owner = fig.scene)
    position = lift(block.blockscene, block.layoutobservables.computedbbox) do bounds
        Point2f(bounds.origin + Vec2f(bounds.widths[1] / 2, bounds.widths[2]))
    end
    tip = tooltip!(fig.scene, position; text, visible = false, fontsize = 13,
        placement = :above, inspectable = false,
        backgroundcolor = _panel_bg_color(fig.scene.backgroundcolor[]),
        outline_color = _panel_bg_color(fig.scene.backgroundcolor[], 0.35))
    _view_on(owner, events(fig.scene).mouseposition) do _
        alignment = position[][1] > Makie.viewport(fig.scene)[].widths[1] / 2 ? 1.0 : 0.0
        tip.align[] == alignment || (tip.align[] = alignment)
        visible = _control_hit(block, fig.scene)
        tip.visible[] == visible || (tip.visible[] = visible)
    end
    _view_on(owner, block.blockscene.visible) do visible
        visible || (tip.visible[] = false)
    end
    if hasproperty(block, :visible)
        _view_on(owner, block.visible) do visible
            visible || (tip.visible[] = false)
        end
    end
    return tip
end

function _control_icon_button(fig, position, kind::Symbol, description = nothing;
    owner = fig.scene, size = 24.0)
    button = Button(position; label = "", width = size, height = size,
        cornerradius = 4, padding = (0, 0, 0, 0))
    icon = _control_icon!(button, kind; size = 17.0)
    tooltip = isnothing(description) ? nothing : _control_tooltip!(fig, button, description; owner)
    return (button = button, icon = icon, tooltip = tooltip)
end
