# This file is a part of BAT.jl, licensed under the MIT License (MIT).


mutable struct _ControlNoticeClock
    duration::Float64
    remaining::Float64
    updated::UInt64
    paused::Bool
    active::Bool
end

_ControlNoticeClock(duration::Real) = _ControlNoticeClock(duration, 0.0, 0, false, false)


# Cairo snapshots may not emit ticks; also expire notices before display.
struct _ControlPresentation
    update!::Function
end
Makie.update_state_before_display!(presentation::_ControlPresentation) = presentation.update!()


function _start_notice!(clock::_ControlNoticeClock, now::UInt64 = time_ns())
    clock.remaining = clock.duration
    clock.updated = now
    clock.paused = false
    clock.active = true
    return nothing
end


function _advance_notice!(clock::_ControlNoticeClock, paused::Bool, now::UInt64 = time_ns())
    clock.active || return nothing
    if !clock.paused
        clock.remaining = max(0.0, clock.remaining - Float64(now - clock.updated) / 1.0e9)
    end
    clock.updated = now
    clock.paused = paused
    clock.active = clock.remaining > 0
    return nothing
end


# Position notices outside control cells, in the space reserved above the range strip.
function _control_notice(fig, anchor; owner = fig.scene, duration::Real = 3.0)
    clock = _ControlNoticeClock(duration)
    tile_height = Ref(88.0)
    return_focus = Ref(WeakRef(nothing))
    bounds = Observable(Rect2f(0, 0, 340, 88))
    background = Box(
        fig; bbox = bounds, color = _panel_bg_color(fig.scene.backgroundcolor[], 0.04),
        cornerradius = 8, strokewidth = 1,
        strokecolor = _panel_bg_color(fig.scene.backgroundcolor[], 0.3)
    )
    title_bounds = lift(fig.scene, bounds) do b
        Rect2f(b.origin + Vec2f(12, b.widths[2] - 36), Vec2f(280, 28))
    end
    body_bounds = lift(fig.scene, bounds) do b
        Rect2f(b.origin + Vec2f(12, 12), Vec2f(316, b.widths[2] - 52))
    end
    close_bounds = lift(fig.scene, bounds) do b
        Rect2f(b.origin + Vec2f(306, b.widths[2] - 30), Vec2f(24, 24))
    end
    title = Label(fig, ""; bbox = title_bounds, fontsize = 15, halign = :left,
        justification = :left, tellwidth = false, word_wrap = true)
    body = Label(fig, ""; bbox = body_bounds, fontsize = 14, halign = :left, valign = :top,
        justification = :left, tellwidth = false, word_wrap = true)
    close = _control_icon_button(fig, fig, :close, "Dismiss message"; owner)
    close.button.layoutobservables.suggestedbbox[] = close_bounds[]
    _view_on(owner, close_bounds) do b
        close.button.layoutobservables.suggestedbbox[] = b
    end
    bar_bounds = Observable(Rect2f(0, 0, 0, 3))
    bar = Box(fig; bbox = bar_bounds, color = _status_text_color(fig.scene.backgroundcolor[]),
        cornerradius = 1.5, strokewidth = 0)
    blocks = (background, title, body, close.button, bar)
    kind = Ref(:info)
    shown = Ref(true)

    function visible!(value)
        shown[] == value && return nothing
        shown[] = value
        for block in blocks
            _set_block_visible!(block, value)
        end
        return nothing
    end

    function dismiss!(; restore_focus = false)
        clock.active = false
        visible!(false)
        target = return_focus[].value
        if restore_focus && !isnothing(target) && target.blockscene.visible[]
            _control_focus!(target)
        end
        return nothing
    end

    function paint!()
        if !clock.active
            visible!(false)
            return nothing
        end
        viewport = Makie.viewport(fig.scene)[]
        a = anchor.layoutobservables.computedbbox[]
        x = max(8, min(a.origin[1] + a.widths[1] - 340, viewport.widths[1] - 348))
        y = max(8, min(a.origin[2] + a.widths[2] + 8, viewport.widths[2] - tile_height[] - 8))
        next_bounds = Rect2f(x, y, 340, tile_height[])
        bounds[] == next_bounds || (bounds[] = next_bounds)
        # The left edge advances rightward while the right edge remains anchored.
        remaining_width = 340 * clock.remaining / clock.duration
        bar_bounds[] = Rect2f(x + 340 - remaining_width, y, remaining_width, 3)
        return nothing
    end

    function show!(error; severity = :error, force = false, focus = nothing)
        clock.active && kind[] === :error && severity === :info && !force && return nothing
        kind[] = severity
        title.text[] = "Marginals · " * error.title
        body.text[] = error.message
        tile_height[] = clamp(ceil(something(body.layoutobservables.autosize[][2], 0)) + 56, 88, 200)
        if isnothing(focus)
            focused = findfirst(b -> b isa Textbox && b.focused[], fig.content)
            focus = isnothing(focused) ? nothing : fig.content[focused]
        end
        return_focus[] = WeakRef(focus)
        bar.color[] = severity === :error ? _status_text_color(fig.scene.backgroundcolor[]) :
            _panel_bg_color(fig.scene.backgroundcolor[], 0.45)
        _start_notice!(clock)
        paint!()
        visible!(true)
        return nothing
    end

    function advance!()
        !clock.active && !shown[] && return nothing
        paused = clock.active && Point2f(events(fig.scene).mouseposition[]) in bounds[]
        _advance_notice!(clock, paused)
        paint!()
        return nothing
    end

    _view_on(owner, close.button.clicks) do _
        dismiss!(; restore_focus = true)
    end
    # Own complete press/release pairs, including when a close press hides us.
    # Outside-origin drags must reach native dragstop even when released here.
    pressed = Set{Mouse.Button}()
    _view_on(owner, events(fig.scene).mousebutton; priority = 85) do event
        if event.action == Mouse.release
            consumed = event.button in pressed
            delete!(pressed, event.button)
            return Consume(consumed)
        end
        inside = clock.active && Point2f(events(fig.scene).mouseposition[]) in bounds[]
        inside || return Consume(false)
        push!(pressed, event.button)
        if event.button == Mouse.left && event.action == Mouse.press && _control_hit(close.button, fig.scene)
            dismiss!(; restore_focus = true)
        end
        return Consume(true)
    end
    _view_on(owner, events(fig.scene).scroll; priority = 85) do _
        return Consume(clock.active && Point2f(events(fig.scene).mouseposition[]) in bounds[])
    end
    _view_on(owner, events(fig.scene).mouseposition) do _
        advance!()
    end
    _view_on(owner, events(fig.scene).tick) do _
        advance!()
    end
    _view_on(owner, Makie.viewport(fig.scene)) do _
        advance!()
    end
    _view_on(owner, events(fig.scene).keyboardbutton; priority = 90) do event
        if clock.active && event.action == Keyboard.press && event.key == Keyboard.escape
            dismiss!(; restore_focus = true)
            return Consume(true)
        end
        return Consume(false)
    end
    refresh! = owner isa Scene ? advance! : () -> _with_live_view(advance!, owner)
    push!(fig.content, _ControlPresentation(refresh!))
    # Draw above the surrounding controls and plots.
    for block in blocks
        translate!(block.blockscene, 0, 0, 1000)
    end
    visible!(false)
    return (
        show! = show!, dismiss! = dismiss!, advance! = advance!, clock = clock,
        bounds = bounds, close = close.button, blocks = blocks
    )
end
