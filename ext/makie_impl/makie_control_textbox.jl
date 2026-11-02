# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# Keep Makie-version-specific editing details here. BAT owns admission and submit /
# cancel semantics; native Textbox retains text rendering, cursor movement and selection.
_legacy_control_textbox(box::Textbox) = hasfield(typeof(box), :cursorindex)
_control_text(box::Textbox) = _legacy_control_textbox(box) && box.displayed_string[] == " " ?
    "" : String(box.displayed_string[])

function _control_insert_candidate(box::Textbox, insertion::String)
    text = _control_text(box)
    if _legacy_control_textbox(box)
        chars = collect(text)
        cursor = clamp(box.cursorindex[], 0, length(chars))
        return (join(chars[1:cursor]) * insertion * join(chars[cursor+1:end]), cursor + length(insertion))
    else
        # EditableText's native edit operation preserves selections and multiple cursors.
        return Makie.apply_edits(text, box.editor.cursors[], c ->
            Makie.EditOp(min(c.anchor, c.head):max(c.anchor, c.head), insertion))
    end
end

function _control_cursor!(box::Textbox, cursor)
    if _legacy_control_textbox(box)
        box.cursorindex[] = cursor
    else
        box.editor.cursors[] = cursor
    end
    return nothing
end

function _control_focus!(box::Textbox)
    # Old Makie.focus! starts an unowned animation task. Setting focused first also
    # makes its subsequent native mouse handler a no-op for task creation.
    box.focused[] = true
    _legacy_control_textbox(box) && (box.cursorcolor[] = box.textcolor[])
    return nothing
end

function _control_blur!(box::Textbox)
    box.focused[] = false
    _legacy_control_textbox(box) && (box.cursorcolor[] = :transparent)
    return nothing
end

function _control_hit(box, scene)
    box.blockscene.visible[] || return false
    hasproperty(box, :visible) && !box.visible[] && return false
    bounds = box.layoutobservables.computedbbox[]
    all(bounds.widths .> 0) || return false
    return Point2f(events(scene).mouseposition[]) in bounds
end

function _control_textbox(fig, position; owner = fig.scene, limit::Integer,
    text::AbstractString = "", on_submit = () -> nothing, on_cancel = () -> nothing,
    on_blur = on_submit, skip_blur = () -> false, on_reject = () -> nothing,
    on_focus = () -> nothing,
    kwargs...)
    ncodeunits(text) <= limit || throw(ArgumentError("Initial control text exceeds its input limit"))
    box = Textbox(position; stored_string = String(text), placeholder = "",
        reset_on_defocus = false, defocus_on_submit = false, kwargs...)
    function replace!(value; stored = false)
        if ncodeunits(value) > limit
            on_reject()
            return false
        end
        box.displayed_string[] = String(value)
        stored && (box.stored_string[] = String(value))
        return true
    end
    function insert!(value)
        # Reject before layout or expansion; normalize multiline clipboard lists.
        ncodeunits(value) <= limit || (on_reject(); return false)
        insertion = replace(String(value), '\r' => ' ', '\n' => ' ')
        candidate, cursor = _control_insert_candidate(box, insertion)
        ncodeunits(candidate) <= limit || (on_reject(); return false)
        replace!(candidate)
        _control_cursor!(box, cursor)
        return true
    end
    _view_on(owner, events(fig.scene).unicode_input; priority = 100) do char
        box.focused[] || return Consume(false)
        # Admission before native Unicode handlers (priority 60).
        candidate, _ = _control_insert_candidate(box, string(char))
        if ncodeunits(candidate) > limit
            on_reject()
            return Consume(true)
        end
        return Consume(false)
    end
    _view_on(owner, events(fig.scene).keyboardbutton; priority = 100) do event
        box.focused[] || return Consume(false)
        event.action == Keyboard.release && return Consume(false)
        if event.key == Keyboard.v && ispressed(fig.scene,
                Keyboard.left_control | Keyboard.right_control | Keyboard.left_super | Keyboard.right_super)
            value = try
                Makie.clipboard()
            catch
                on_reject()
                return Consume(true)
            end
            insert!(value)
            return Consume(true)
        elseif event.key in (Keyboard.enter, Keyboard.kp_enter)
            on_submit()
            return Consume(true)
        elseif event.key == Keyboard.escape
            on_cancel()
            return Consume(true)
        elseif _legacy_control_textbox(box) &&
                ((event.key == Keyboard.backspace && box.cursorindex[] == 0) ||
                 (event.key == Keyboard.delete && box.cursorindex[] >= length(_control_text(box))))
            return Consume(true)
        end
        return Consume(false)
    end
    _view_on(owner, events(fig.scene).mousebutton; priority = 100) do event
        event.button == Mouse.left && event.action == Mouse.press || return Consume(false)
        if _control_hit(box, fig.scene)
            was_focused = box.focused[]
            _control_focus!(box)
            was_focused || on_focus()
        elseif box.focused[]
            _control_blur!(box)
            skip_blur() || on_blur()
        end
        return Consume(false)
    end
    return (box = box, replace! = replace!, insert! = insert!,
        focus! = () -> _control_focus!(box), blur! = () -> _control_blur!(box))
end
