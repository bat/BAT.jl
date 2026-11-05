# This file is a part of BAT.jl, licensed under the MIT License (MIT).


mutable struct _RangeGesture
    phase::Symbol
    native_start::Tuple{Int64,Int64}
    exact_start::Tuple{Int64,Int64}
    submitted::Union{Nothing,Tuple{Int64,Int64}}
end


# Native widgets own dragging and editing; the view owner applies committed ranges.
function _range_controls!(
    fig, graph, slider, value_control, live_drag;
    owner = fig.scene, live_vis = nothing
)
    value_entry = value_control.box
    input = events(fig.scene)
    draft = Observable(something(graph[:window_steps][], (Int64(1), graph[:max_step][])))
    syncing = Ref(false)
    edited = Ref(false)
    text_updates = Ref(0)
    gesture = Ref{Union{Nothing,_RangeGesture}}(nothing)
    last_apply = Ref(zero(UInt64))
    closed = Ref(false)
    seen_open = Ref(input.window_open[])

    function paint!(interval; text = true)
        syncing[] = true
        try
            draft[] = interval
            set_close_to!(slider, interval...)
            text && value_control.replace!("$(interval[1]):$(interval[2])"; stored = true)
        finally
            syncing[] = false
        end
        return nothing
    end

    function reconcile!()
        closed[] && return nothing
        limit = graph[:max_step][]
        if !isnothing(gesture[])
            # Keep the slider scale fixed until the gesture ends.
            limit <= 1 && (gesture[].phase = :cancelling)
            return nothing
        end
        if limit > 1 && last(slider.range[]) != limit
            syncing[] = true
            try
                slider.range[] = SampleRangeGrid(limit)
                set_close_to!(slider, draft[]...)
            finally
                syncing[] = false
            end
        end
        if !edited[] && limit > 0
            interval = something(graph[:window_steps][], (Int64(1), limit))
            paint!(interval; text = !value_entry.focused[] && text_updates[] == 0)
        end
        return nothing
    end

    function submit!(interval)
        closed[] && return nothing
        last_apply[] = time_ns()
        # Do not deduplicate live requests against either a local last value or
        # the displayed graph: an earlier deferred intent may need replacing.
        _apply_sample_window!(graph, interval; live_vis)
        isnothing(live_vis) && reconcile!()
        return nothing
    end

    function commit_text!(; explicit = true)
        closed[] && return nothing
        !explicit && !edited[] && return reconcile!()
        interval = _parse_sample_window(_control_text(value_entry), graph[:max_step][])
        if isnothing(interval)
            value_entry.bordercolor[] = _status_text_color(fig.scene.backgroundcolor[])
            return nothing
        end
        value_entry.bordercolor[] = _panel_bg_color(fig.scene.backgroundcolor[], 0.3)

        edited[] = false
        paint!(interval)
        submit!(interval)
        return nothing
    end

    function cancel_text!()
        closed[] && return nothing

        edited[] = false
        value_control.blur!()
        value_entry.bordercolor[] = _panel_bg_color(fig.scene.backgroundcolor[], 0.3)
        reconcile!()
        return nothing
    end

    function preview(active)
        native = Int64.(slider.interval[])
        return native == active.native_start ? active.exact_start : native
    end

    function apply_drag!()
        active = gesture[]
        (closed[] || !live_drag[] || isnothing(active) || active.phase !== :dragging) && return nothing
        interval = preview(active)
        interval == active.exact_start && isnothing(active.submitted) && return nothing
        interval == active.submitted && return nothing
        time_ns() - last_apply[] < _LIVE_DRAG_INTERVAL_NS && return nothing
        active.submitted = interval
        submit!(interval)
        return nothing
    end

    function finish!()
        active = gesture[]
        (closed[] || isnothing(active)) && return nothing
        # Tick/input boundaries follow native dispatch. GLFW supplies releases
        # after focus loss; restoration must wait for native dragstop as well.
        Mouse.left in input.mousebuttonstate && active.phase in (:dragging, :cancelling) &&
            return nothing
        cancelled = active.phase in (:cancelling, :cancelled)
        interval = preview(active)
        changed = Int64.(slider.interval[]) != active.native_start
        # Claim completion before any reentrant callback.
        gesture[] = nothing

        if !cancelled && (changed || !isnothing(active.submitted))
            edited[] = false
            paint!(interval)
            submit!(interval)
        else
            reconcile!()
        end
        return nothing
    end

    _view_on(owner, slider.interval) do interval
        active = gesture[]
        (closed[] || syncing[] || isnothing(active) || !(active.phase in (:dragging, :released))) &&
            return
        next = preview(active)
        if Int64.(interval) != active.native_start || next != draft[]
            edited[] = false
            value_entry.bordercolor[] = _panel_bg_color(fig.scene.backgroundcolor[], 0.3)
            paint!(next)
        end
        apply_drag!()
    end
    _view_on(owner, value_entry.displayed_string) do _
        (closed[] || syncing[]) && return

        edited[] = true
        interval = _parse_sample_window(_control_text(value_entry), graph[:max_step][])
        if isnothing(interval)
            live_drag[] && (value_entry.bordercolor[] = _status_text_color(fig.scene.backgroundcolor[]))
        else
            value_entry.bordercolor[] = _panel_bg_color(fig.scene.backgroundcolor[], 0.3)
            paint!(interval; text = false)
            if live_drag[]
                # Preserve native text/cursor while each valid edit is applied.
                edited[] = false
                text_updates[] += 1
                try
                    submit!(interval)
                finally
                    text_updates[] -= 1
                end
            end
        end
    end

    # Finish prior gestures and record release before textbox blur (100) and
    # notice handling (85), without consuming native input.
    _view_on(owner, input.mousebutton; priority = 101) do event
        closed[] && return Consume(false)
        if event.action == Mouse.press
            # The button-state tracker has already added this new press.
            active = gesture[]
            if !isnothing(active) && active.phase in (:released, :cancelled)
                finish!()
            end
        elseif event.button == Mouse.left && event.action == Mouse.release
            active = gesture[]
            if !isnothing(active)
                active.phase === :dragging && (active.phase = :released)
                active.phase === :cancelling && (active.phase = :cancelled)
            end
        end
        return Consume(false)
    end
    _view_on(owner, input.mousebutton; priority = 2) do event
        if !closed[] && event.button == Mouse.left && event.action == Mouse.press &&
                length(input.mousebuttonstate) == 1 && graph[:max_step][] > 1 &&
                _control_hit(slider, fig.scene)
            gesture[] = _RangeGesture(:dragging, Int64.(slider.interval[]), draft[], nothing)
        end
        return Consume(false)
    end
    for event in (input.keyboardbutton, input.unicode_input)
        _view_on(owner, event; priority = 101) do _
            finish!()
            return Consume(false)
        end
    end
    _view_on(owner, input.hasfocus) do focused
        active = gesture[]
        if !focused && !isnothing(active) && active.phase === :dragging
            active.phase = :cancelling
        end
    end
    _view_on(owner, input.window_open) do opened
        if opened
            seen_open[] = true
            closed[] = false
            reconcile!()
        elseif seen_open[]
            closed[] = true
            gesture[] = nothing
            edited[] = false
            value_control.blur!()
            value_entry.bordercolor[] = _panel_bg_color(fig.scene.backgroundcolor[], 0.3)
        end
    end
    _view_on(owner, input.tick) do _
        finish!()
        apply_drag!()
    end
    _view_on(owner, live_drag) do _
        apply_drag!()
    end
    if isnothing(live_vis)
        _view_on(owner, graph[:window_steps]) do _
            reconcile!()
        end
    else
        _view_on(owner, _live_view(live_vis).settled) do _
            reconcile!()
        end
    end
    return (; draft, syncing, reconcile!, commit_text!, cancel_text!)
end
