# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Drive production text/gesture routes, without depending on a removed button
# or mutating IntervalSlider's output attribute/private mouse state.
function range_key!(fig, key = Makie.Keyboard.enter)
    input = Makie.events(fig)
    input.keyboardbutton[] = Makie.KeyEvent(key, Makie.Keyboard.press)
    input.keyboardbutton[] = Makie.KeyEvent(key, Makie.Keyboard.release)
end

function range_text!(fig, interval; submit = false)
    controls = Base.get_extension(BAT, :BATMakieExt)._figure_controls(fig).range[]
    controls.value_control.focus!()
    controls.value_control.replace!("$(interval[1]):$(interval[2])")
    submit && range_key!(fig)
    return nothing
end

function range_tick!(fig; advance = 0.3)
    input = Makie.events(fig)
    tick = input.tick[]
    input.tick[] = Makie.Tick(Makie.RegularRenderTick, tick.count + 1, tick.time + advance, advance)
end

function range_point(slider, fraction)
    box = slider.layoutobservables.computedbbox[]
    padding = box.widths[2] / 2
    return box.origin + Makie.Vec2f(padding + fraction * (box.widths[1] - 2padding), padding)
end

range_pointer!(fig, position) = (Makie.events(fig).mouseposition[] = Tuple(Float64.(position)))

function range_press!(fig, slider, fraction; advance = true)
    advance && range_tick!(fig)
    range_pointer!(fig, (-10, -10))
    point = range_point(slider, fraction)
    range_pointer!(fig, point + Makie.Vec2f(0, 1))
    range_pointer!(fig, point)
    Makie.events(fig).mousebutton[] = Makie.MouseButtonEvent(Makie.Mouse.left, Makie.Mouse.press)
end

function range_release!(fig; tick = true)
    Makie.events(fig).mousebutton[] = Makie.MouseButtonEvent(Makie.Mouse.left, Makie.Mouse.release)
    tick && range_tick!(fig)
end

function range_move!(fig, slider, fraction)
    range_pointer!(fig, range_point(slider, fraction))
end
