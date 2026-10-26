# This file is a part of BAT.jl, licensed under the MIT License (MIT).


_rel_luminance(rgb::Colors.RGB) = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b

# Contrast each panel/widget background with its parent, darkening light colors and lightening
# dark ones.
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


# Shared credible-region palette, ordered from widest to tightest region.
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


# Type-erase the picker callback to share compiled widget code across call sites, at the cost of
# one dynamic dispatch per click.
struct PickerInfo
    N::Int
    N_max::Int
    initial_vsel::Vector{Int}
    apply_vsel!::Function
end


# The precompile workload suppresses display without retaining any Figure here.
const _SUPPRESS_DISPLAY = Ref(false)

function _start_observed_gl_renderloop!(fig, screen, backend, on_failure)
    isnothing(on_failure) && return _start_gl_renderloop!(screen, backend)
    # A stop request does not finish its task. Starting now could overlap a
    # predecessor which can still close this screen or clear its task handle.
    task = screen.rendertask
    !isnothing(task) && !istaskdone(task) && throw(ArgumentError(
        "Live GL display requires a renderer started by BAT"))
    config = screen.config
    original = config.renderloop
    scene = WeakRef(fig.scene)
    entered = Ref{Union{Nothing,Task}}(nothing)
    guarded = function(native_screen)
        entered[] = current_task()
        try
            result = original(native_screen)
            expected = scene.value
            if !isnothing(expected) && native_screen.scene === expected &&
                    isopen(native_screen) && !native_screen.stop_renderloop[]
                error("Live GL renderloop returned while its window remained open")
            end
            return result
        catch err
            bt = catch_backtrace()
            expected = scene.value
            if !isnothing(expected) && native_screen.scene === expected
                try
                    on_failure(err, bt)
                catch reporting_error
                    BAT._visualizer_cleanup_diagnostic(:renderer_observer,
                        (reporting_error, catch_backtrace()))
                end
            end
            rethrow()
        end
    end
    try
        config.renderloop = guarded
        _start_gl_renderloop!(screen, backend)
        if isopen(screen) && screen.scene === fig.scene
            task = screen.rendertask
            task === entered[] && backend.renderloop_running(screen) ||
                error("Live GL display requires an active observed renderer")
        end
    finally
        # The upstream startup contract guarantees invocation entry before
        # returning. The running frame owns the wrapper; config never does
        # beyond enrollment, including closed-screen or startup-error paths.
        config.renderloop === guarded && (config.renderloop = original)
    end
    return nothing
end


function _maybe_display(fig; on_failure = nothing, on_render_failure = nothing)
    _SUPPRESS_DISPLAY[] && return nothing
    backend = Makie.current_backend()
    if _is_gl_backend(backend)
        _display_gl_figure(fig, backend; on_failure = on_failure, on_render_failure = on_render_failure)
    else
        display(fig)
    end
    return nothing
end


_is_gl_backend(backend) = backend isa Module && nameof(backend) === :GLMakie

function _check_gl_thread()
    Threads.threadid() == 1 ||
        throw(ArgumentError("Display or warm up GLMakie figures on Julia's main thread"))
    return nothing
end


# GLMakie.start_renderloop! yields before storing its task handle. Run only
# that enrollment in a short, owned task, and join it even when our caller is
# interrupted. In particular, do not move scene/display callbacks into it.
function _enroll_gl_renderloop!(screen, backend)
    Base.disable_sigint(() -> backend.start_renderloop!(screen))
    return nothing
end


function _run_gl_startup(screen, backend)
    task = nothing
    enrolled = false
    failure = Ref{Union{Nothing,CapturedException}}(nothing)
    Base.disable_sigint() do
        try
            rng = Random.default_rng()
            saved = copy(rng)
            try
                # Each one-shot startup has an explicit private origin. This
                # is not a unique seed or a stream for numerical sampling.
                copy!(rng, Random.Xoshiro(0x424154676c303031))
                task = @task begin
                    try
                        _enroll_gl_renderloop!(screen, backend)
                    catch err
                        failure[] = CapturedException(err, catch_backtrace())
                    end
                end
                schedule(task)
                enrolled = true
            finally
                copy!(rng, saved)
            end
            Base.reenable_sigint(() -> wait(task))
            isnothing(failure[]) || throw(failure[].ex)
        finally
            if enrolled
                while !istaskdone(task)
                    try
                        Base.reenable_sigint(() -> wait(task))
                    catch err
                        (err isa InterruptException ||
                            (err isa TaskFailedException && err.task === task && istaskdone(task))) ||
                            rethrow()
                    end
                end
            end
        end
    end
    return nothing
end


function _start_gl_renderloop!(screen, backend)
    isopen(screen) || return nothing
    if backend.renderloop_running(screen)
        # The running-screen path only resumes a paused loop; it creates no task.
        backend.start_renderloop!(screen)
    else
        _run_gl_startup(screen, backend)
    end
    # GLMakie can return after a render task has already failed. Do not certify
    # that as successful startup; retain the failed task and its original cause.
    task = screen.rendertask
    isnothing(task) || !istaskfailed(task) || fetch(task)
    return nothing
end


# GLMakie's stop routine catches an interrupted wait and clears rendertask.
# Retain our own handle until terminal, including repeated cleanup interrupts.
# Call outside the live view guard: renderer callbacks may need that guard.
function _close_gl_screen!(screen, backend)
    task = screen.rendertask
    Base.disable_sigint() do
        try
            while true
                try
                    Base.reenable_sigint(() ->
                        backend.stop_renderloop!(screen; close_after_renderloop = false))
                    break
                catch err
                    err isa InterruptException || rethrow()
                end
            end
        finally
            try
                if !isnothing(task) && task !== current_task()
                    while !istaskdone(task)
                        try
                            Base.reenable_sigint(() -> wait(task))
                        catch err
                            (err isa InterruptException ||
                                (err isa TaskFailedException && err.task === task && istaskdone(task))) ||
                                rethrow()
                        end
                    end
                end
            finally
                while true
                    try
                        Base.reenable_sigint(() -> close(screen))
                        break
                    catch err
                        err isa InterruptException || rethrow()
                    end
                end
            end
        end
    end
    return nothing
end


# Explicit screens avoid taking over an unrelated singleton window. Keep a
# local owner through partial scene attachment, before screen.scene is set.
# A constructor that throws before returning a handle is a backend boundary.
function _with_gl_screen(f, fig, backend; temporary::Bool = false, on_failure = nothing)
    _check_gl_thread()
    screen = temporary ? nothing : Makie.getscreen(fig.scene, backend)
    created = isnothing(screen) || !isopen(screen)
    if created
        screen = temporary ?
            backend.Screen(; start_renderloop = false, visible = false) :
            backend.Screen(; start_renderloop = false)
    end
    success = false
    try
        result = f(screen)
        success = true
        return result
    catch err
        # A live owner records startup failure before rollback emits a window
        # close event. Otherwise that event could claim the first stop outcome.
        isnothing(on_failure) || on_failure(err, catch_backtrace())
        rethrow()
    finally
        if temporary || (created && !success)
            try
                _close_gl_screen!(screen, backend)
            catch err
                BAT._visualizer_cleanup_diagnostic(:renderer_startup, (err, catch_backtrace()))
            end
        end
    end
end


function _display_gl_figure(fig, backend; on_failure = nothing, on_render_failure = nothing)
    inline = Makie.ALWAYS_INLINE_PLOTS[]
    show_inline = (inline === true || inline === Makie.automatic) &&
        Makie.can_show_inline(backend)
    if inline === true && !show_inline
        @warn "No inline Makie display is available; opening a GLMakie window"
    end
    _with_gl_screen(fig, backend; on_failure = on_failure) do screen
        # Keep scene attachment and synchronous display callbacks on the initiating task.
        predecessor = screen.rendertask
        if !isnothing(on_render_failure) && !isnothing(predecessor) && !istaskdone(predecessor)
            throw(ArgumentError("Live GL display requires a renderer started by BAT"))
        end
        display(screen, fig; update = !show_inline)
        _start_observed_gl_renderloop!(fig, screen, backend, on_render_failure)
        renderer = screen.rendertask
        if show_inline && (isnothing(on_render_failure) || isopen(screen))
            Core.invoke(display, Tuple{Any}, fig)
            Makie.wait_for_display(Makie.getscreen(fig.scene))
            if !isnothing(on_render_failure) && isopen(screen) && screen.scene === fig.scene
                screen.rendertask === renderer ||
                    error("Live GL inline display replaced its observed renderer")
            end
        end
    end
    return nothing
end


function _render_warmup_figure(fig, backend)
    if _is_gl_backend(backend)
        _with_gl_screen(fig, backend; temporary = true) do screen
            display(screen, fig)
            # colorbuffer(fig) can start a render loop when applying screen
            # configuration. Rendering the explicit screen needs no such task.
            Makie.colorbuffer(screen)
        end
    else
        Makie.colorbuffer(fig; backend = backend)
    end
    return nothing
end


# Successful warmup per renderer module, reset at runtime by __init__. This is
# not a cache of individual screens/GL contexts or their shader resources.
const _WARMED_BACKENDS = Set{Module}()

function warmup_makie_shaders()
    backend = Makie.current_backend()
    ismissing(backend) && return nothing
    backend in _WARMED_BACKENDS && return nothing
    _is_gl_backend(backend) && _check_gl_thread()
    @info "Warming up Makie renderer" backend
    fig = Figure()
    try
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
        hexbin!(ax, [0.0], [0.0]; bins = 2)
        poly!(ax, Point2f[(0, 0), (1, 0), (0, 1)])

        _render_warmup_figure(fig, backend)
        push!(_WARMED_BACKENDS, backend)
    finally
        # This temporary figure never becomes the live session's view.
        try
            _close_owned_live_screens!(fig)
        catch err
            BAT._visualizer_cleanup_diagnostic(:shader_warmup, (err, catch_backtrace()))
        end
        try
            empty!(fig)
        catch err
            BAT._visualizer_cleanup_diagnostic(:shader_warmup, (err, catch_backtrace()))
        end
    end
    return nothing
end
