# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function _build_marginal_controls!(
    fig, layout, graph, info::PickerInfo, column,
    notice, collapse_button; live_vis = nothing
)
    owner = isnothing(live_vis) ? fig.scene : live_vis
    state = _MarginalSelection(info.N, info.N_max, info.initial_vsel)
    nslots = length(state.headers)
    body_size = _picker_body_size(info.N)
    gap = 4.0
    cell = (body_size - nslots * gap) / (nslots + 1)
    syncing = Ref(false)
    visible = Ref(false)
    mode = Ref(:matrix)
    header_drafts = Dict{Int,String}()
    header_errors = Dict{Int,Any}()
    editor_error = Ref{Any}(nothing)

    heading = layout[1, column] = GridLayout(; halign = :left)
    title = Label(heading[1, 1], "Marginals"; halign = :left)
    reset = _control_icon_button(fig, heading[1, 2], :reset; owner)
    switch = _control_icon_button(fig, heading[1, 3], :menu; owner)

    matrix = layout[2:4, column] = GridLayout(; width = body_size, height = body_size)
    editor_layout = layout[2:4, column] = GridLayout(4, 1; width = body_size, height = body_size)
    editor_heading = editor_layout[1, 1] = GridLayout(; halign = :left)
    editor_title = Label(editor_heading[1, 1:2], "Display dimensions:"; halign = :left,
        fontsize = 16, tellwidth = false, word_wrap = true, justification = :left)
    preview = Label(editor_layout[3, 1], ""; fontsize = 12, tellwidth = false,
        word_wrap = true, halign = :left, justification = :left)
    help = Ref{Union{Nothing,Button}}(nothing)
    help_requests = Observable(0)
    cancel = _control_icon_button(fig, editor_heading[1, 3], :close, "Cancel dimension edits"; owner)
    for row in (heading, editor_heading)
        colsize!(row, 1, Fixed(92))
        colsize!(row, 2, Fixed(24))
        colsize!(row, 3, Fixed(24))
        colgap!(row, 6)
    end
    skip_blur() = (notice.clock.active && Point2f(events(fig.scene).mouseposition[]) in notice.bounds[]) ||
        any(b -> _control_hit(b, fig.scene),
            (reset.button, switch.button, cancel.button, collapse_button, notice.close)) ||
        (!isnothing(help[]) && _control_hit(help[], fig.scene))
    reject_input() = notice.show!(_marginal_error("Input is too long", "Enter a shorter dimension list."))
    editor = _control_textbox(
        fig, editor_layout[2, 1]; owner,
        limit = _marginal_text_limit(info.N, info.N_max),
        on_submit = () -> commit_editor!(), on_cancel = () -> cancel_editor!(),
        on_blur = () -> nothing, skip_blur, on_reject = reject_input,
        on_focus = () -> isnothing(editor_error[]) || notice.show!(editor_error[]),
        validator = text -> isnothing(_parse_marginal_dimensions(text, state.N, state.capacity).error),
        width = nothing, height = 32, tellwidth = false, fontsize = 15
    )
    editor_blocks = Any[editor_title, editor.box, preview, cancel.button]
    rowgap!(editor_layout, 8)
    rowsize!(editor_layout, 1, Fixed(40))
    rowsize!(editor_layout, 2, Fixed(32))
    rowsize!(editor_layout, 3, Auto(false))
    rowsize!(editor_layout, 4, Fixed(28))
    colsize!(editor_layout, 1, Fixed(body_size))

    headers = Dict{Tuple{Int,Symbol},Any}()
    checks = Dict{Tuple{Int,Int},Checkbox}()
    matrix_blocks = Any[]
    for slot in 1:nslots, axis in (:row, :column)
        position = axis === :row ? matrix[slot+1, 1] : matrix[1, slot+1]
        header = _control_textbox(
            fig, position; owner, text = string(state.headers[slot]),
            limit = ndigits(info.N) + 2,
            on_submit = () -> commit_header!(slot), on_cancel = () -> cancel_header!(slot),
            skip_blur, on_reject = reject_input,
            on_focus = () -> !haskey(header_errors, slot) || notice.show!(header_errors[slot]),
            validator = text -> isnothing(_marginal_header_candidate(state, slot, text).error),
            width = cell, height = min(cell, 30), fontsize = 14,
            textpadding = (3, 3, 3, 3), cornerradius = 4
        )
        headers[(slot, axis)] = header
        push!(matrix_blocks, header.box)
        _control_tooltip!(fig, header.box, "Dimension at matrix position $slot; linked row and column"; owner)
        _view_on(owner, header.box.displayed_string) do _
            syncing[] && return
            state.revision += 1
            header_drafts[slot] = _control_text(header.box)
            candidate = _marginal_header_candidate(state, slot, header_drafts[slot])
            if haskey(header_errors, slot) && isnothing(candidate.error)
                delete!(header_errors, slot)
                notice.dismiss!()
            end
            sync!()
        end
    end
    for i in 1:nslots, j in 1:nslots
        if i >= j
            box = Checkbox(matrix[i+1, j+1]; size = min(24, cell - 8), roundness = 0.25)
            checks[(i, j)] = box
            push!(matrix_blocks, box)
            _view_on(owner, box.checked) do checked
                syncing[] && return
                if !isempty(header_errors)
                    sync!()
                    notice.show!(_marginal_error("Finish the header edit",
                        "Correct or cancel the highlighted dimension before changing the selection."))
                    return
                end
                selected = _vsel_after_toggle(
                    Set(state.selection), state.headers[i], state.headers[j], checked
                )
                if length(selected) > state.capacity
                    sync!()
                    notice.show!(_marginal_capacity_error(state.capacity))
                else
                    submit!(sort!(collect(selected)))
                end
            end
        else
            box = Box(
                matrix[i+1, j+1]; width = min(24, cell - 8), height = min(24, cell - 8),
                cornerradius = 4, strokewidth = 0,
                color = _panel_bg_color(_panel_bg_color(fig.scene.backgroundcolor[]))
            )
            push!(matrix_blocks, box)
        end
    end
    rowgap!(matrix, gap)
    colgap!(matrix, gap)
    for k in 1:nslots+1
        rowsize!(matrix, k, Fixed(cell))
        colsize!(matrix, k, Fixed(cell))
    end

    function sync!()
        syncing[] && return nothing
        syncing[] = true
        try
            for ((slot, _), header) in headers
                text = get(header_drafts, slot, string(state.headers[slot]))
                committed = !haskey(header_drafts, slot)
                if _control_text(header.box) != text || (committed && header.box.stored_string[] != text)
                    header.replace!(text; stored = committed)
                end
                color = haskey(header_errors, slot) ?
                    _status_text_color(fig.scene.backgroundcolor[]) :
                    _panel_bg_color(fig.scene.backgroundcolor[], 0.3)
                header.box.bordercolor[] == color || (header.box.bordercolor[] = color)
            end
            selected = Set(state.selection)
            for ((i, j), check) in checks
                checked = _checkbox_should_be_checked(selected, state.headers[i], state.headers[j])
                check.checked[] == checked || (check.checked[] = checked)
            end
        finally
            syncing[] = false
        end
        return nothing
    end

    function show_help!(shown)
        # Native Button hit-testing ignores scene visibility and accepts even a
        # zero-height bounding box. Remove this overlapping control while hidden.
        if shown && isnothing(help[])
            button = Button(editor_layout[4, 1]; label = "Input formats", fontsize = 13,
                height = 26, tellwidth = false)
            help[] = button
            # The button owns this bridge to the persistent action subscription;
            # delete! releases it when leaving or folding the editor.
            on(button.blockscene, button.clicks) do _
                help_requests[] += 1
            end
        elseif !shown && !isnothing(help[])
            delete!(help[])
            help[] = nothing
        end
        return nothing
    end

    function show_mode!()
        editing = mode[] === :input
        show_help!(visible[] && editing)
        switch.icon[] = editing ? :check : :menu
        for block in matrix_blocks
            _set_block_visible!(block, visible[] && !editing)
        end
        for block in editor_blocks
            _set_block_visible!(block, visible[] && editing)
        end
        for header in values(headers)
            visible[] && !editing || header.blur!()
        end
        visible[] && editing || editor.blur!()
        return nothing
    end

    function settle!()
        # Read the committed selection after the owner drains deferred updates.
        state.selection == graph[:idxs][] || (state.selection = copy(graph[:idxs][]))
        sync!()
        return nothing
    end

    function submit!(selected, mapped_headers = state.headers; finish = () -> nothing)
        revision = _accept_marginal_intent!(state, selected, mapped_headers)
        admitted = info.apply_vsel!(copy(state.selection))
        if admitted === false
            revision == state.revision && settle!()
            return false
        end
        if isnothing(live_vis)
            settle!()
        end
        if revision == state.revision
            notice.dismiss!()
            finish()
            sync!()
        end
        return true
    end

    function commit_header!(slot)
        haskey(header_drafts, slot) || return true
        candidate = _marginal_header_candidate(state, slot, header_drafts[slot])
        if !isnothing(candidate.error)
            header_errors[slot] = candidate.error
            sync!()
            notice.show!(candidate.error; focus = headers[(slot, :row)].box)
            return false
        end
        return submit!(candidate.selection, candidate.headers; finish = () -> begin
            delete!(header_drafts, slot)
            delete!(header_errors, slot)
        end)
    end

    function cancel_header!(slot)
        state.revision += 1
        delete!(header_drafts, slot)
        delete!(header_errors, slot)
        notice.dismiss!()
        sync!()
        return nothing
    end

    function update_preview!()
        result = _parse_marginal_dimensions(_control_text(editor.box), state.N, state.capacity)
        if isnothing(result.error)
            editor_error[] = nothing
            preview.text[] = isempty(result.dimensions) ? "No dimensions selected." :
                "$(length(result.dimensions)) / $(state.capacity) dimensions · all pairwise marginals"
        else
            preview.text[] = "Up to $(state.capacity) dimensions. Confirm with ✓."
        end
        editor.box.bordercolor[] = isnothing(editor_error[]) ?
            _panel_bg_color(fig.scene.backgroundcolor[], 0.3) : _status_text_color(fig.scene.backgroundcolor[])
        return result
    end

    function open_editor!()
        if mode[] === :input
            editor.focus!()
            return nothing
        end
        state.revision += 1
        mode[] = :input
        editor_error[] = nothing
        editor.replace!(join(state.selection, ", "); stored = true)
        notice.dismiss!()
        show_mode!()
        editor.focus!()
        update_preview!()
        return nothing
    end

    function cancel_editor!(; intent = true)
        intent && (state.revision += 1)
        mode[] = :matrix
        editor_error[] = nothing
        editor.replace!(""; stored = true)
        notice.dismiss!()
        show_mode!()
        return nothing
    end

    function commit_editor!()
        result = update_preview!()
        if !isnothing(result.error)
            editor_error[] = result.error
            update_preview!()
            notice.show!(result.error)
            editor.focus!()
            return false
        end
        accepted = submit!(
            result.dimensions, _marginal_editor_headers(state, result.dimensions);
            finish = () -> begin
                empty!(header_drafts)
                empty!(header_errors)
                cancel_editor!(; intent = false)
            end
        )
        if accepted && result.duplicates
            notice.show!(_marginal_error("Repeated dimensions merged",
                "Each dimension is selected once; all pairwise marginals are displayed."); severity = :info)
        end
        return accepted
    end

    function reset!()
        submit!(state.initial_selection, state.initial_headers; finish = () -> begin
            empty!(header_drafts)
            empty!(header_errors)
            foreach(h -> h.blur!(), values(headers))
            cancel_editor!(; intent = false)
        end)
        return nothing
    end

    _view_on(owner, reset.button.clicks) do _
        reset!()
    end
    _view_on(owner, switch.button.clicks) do _
        mode[] === :matrix ? open_editor!() : commit_editor!()
    end
    _view_on(owner, cancel.button.clicks) do _
        cancel_editor!()
    end
    _view_on(owner, events(fig.scene).keyboardbutton; priority = 95) do event
        if visible[] && mode[] === :input && event.action == Keyboard.press && event.key == Keyboard.escape
            cancel_editor!()
            return Consume(true)
        end
        return Consume(false)
    end
    _view_on(owner, editor.box.displayed_string) do _
        state.revision += 1
        update_preview!()
    end
    _view_on(owner, help_requests) do _
        notice.show!(_marginal_error("Input formats",
            "Lists: 2, 7, 8 or [2, 7, 8]. Ranges: 2:4. Pairs: (2,8); (7,8). " *
            "Pairs select their dimensions and all combinations. Empty input clears the selection.");
            severity = :info, force = true)
    end
    if !isnothing(live_vis)
        _view_on(owner, _live_view(live_vis).settled) do _
            settle!()
        end
    end

    function set_visible!(value)
        visible[] = value
        for block in (title, reset.button, switch.button)
            _set_block_visible!(block, value)
        end
        value || notice.dismiss!()
        show_mode!()
        return nothing
    end

    sync!()
    set_visible!(false)
    return (;
        state, headers, checks, editor, header_drafts, header_errors, editor_error,
        mode, reset = reset.button, switch = switch.button, cancel = cancel.button,
        preview, submit!, reset!, commit_header!, commit_editor!, cancel_editor!,
        open_editor!, set_visible!, settle!, matrix_blocks, editor_blocks
    )
end
