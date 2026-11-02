# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakiePicker

using BAT, CairoMakie, Distributions, Makie, Test

const EXT = Base.get_extension(BAT, :BATMakieExt)
const RECIPES = (upper = Hist2D, diagonal = Hist1D, lower = Hist2D)

button(fig, label) = only(filter(x -> x isa Button && x.label[] == label, fig.content))
checkboxes(fig) = filter(x -> x isa Checkbox, fig.content)
axes(fig) = filter(x -> x isa Axis, fig.content)
diagonal(checks, i) = checks[i * (i + 1) ÷ 2]
cell(fig, i, j) = only(filter(x -> x isa Axis, contents(fig[1, 1][i, j])))

function clear_selection!(fig)
    picker = EXT._figure_controls(fig).picker
    picker.open_editor!()
    picker.editor.replace!("")
    @test picker.commit_editor!()
end

@testset "Clearing static variable selections" begin
    for (n, selected, capacity) in ((1, [1], 1), (11, [11], 1),
        (12, [2, 11, 12], 3), (12, Int[], 2))
        @testset "n=$n selected=$selected capacity=$capacity" begin
            samples = DensitySampleVector(v = [Float64.(1:n), Float64.(2:n+1)], logd = zeros(2))
            before = deepcopy(samples)
            fig = bat_makie_plot(samples, RECIPES, selected, capacity; display = false, message_duration = 4.5)
            try
                @test EXT._figure_controls(fig).notice.clock.duration == 4.5
                checks = checkboxes(fig)
                original_axes = axes(fig)
                clear = EXT._figure_controls(fig).picker.reset
                collapse = button(fig, "☰")
                @test length(checks) == EXT._marginal_slot_count(n) * (EXT._marginal_slot_count(n) + 1) ÷ 2
                @test !clear.blockscene.visible[]
                @test clear.width[] == clear.height[] == 0
                Makie.notify(collapse.clicks)
                @test clear.blockscene.visible[]
                @test clear.width[] != 0 && clear.height[] != 0
                @test !isempty(Makie.colorbuffer(fig))
                bounds = clear.layoutobservables.computedbbox[]
                viewport = Makie.viewport(fig.scene)[]
                @test all(bounds.widths .> 0)
                @test all(bounds.origin .>= viewport.origin)
                @test all(bounds.origin .+ bounds.widths .<= viewport.origin .+ viewport.widths)

                if length(selected) == capacity && 1 ∉ selected
                    diagonal(checks, 1).checked[] = true
                    @test !diagonal(checks, 1).checked[]
                    @test EXT._figure_controls(fig).notice.clock.active
                end

                for _ in 1:2
                    clear_selection!(fig)
                    @test all(cb -> !cb.checked[], checks)
                    @test all(axis -> !axis.xlabelvisible[] && !axis.ylabelvisible[], original_axes)
                    @test !EXT._figure_controls(fig).notice.clock.active
                end
                diagonal(checks, 1).checked[] = true
                @test diagonal(checks, 1).checked[]
                @test String(cell(fig, 1, 1).xlabel[]) == "\$v_1\$"
                @test cell(fig, 1, 1).xlabelvisible[]
                if n > 1
                    checks[2].checked[] = true # The pair (2, 1).
                    @test checks[2].checked[] == (capacity >= 2)
                    @test diagonal(checks, 2).checked[] == (capacity >= 2)
                    @test diagonal(checks, 1).checked[]
                end
                clear_selection!(fig)
                @test all(cb -> !cb.checked[], checks)
                @test all(axes(fig) .=== original_axes)
                @test samples == before
                Makie.notify(collapse.clicks)
                @test !clear.blockscene.visible[]
                @test clear.width[] == clear.height[] == 0

                empty!(fig)
                @test isempty(Makie.Observables.listeners(clear.clicks))
                Makie.notify(clear.clicks)
                diagonal(checks, 1).checked[] = true
                @test isempty(fig.content)
            finally
                empty!(fig)
            end
        end
    end
end

mutable struct Capture <: AbstractDisplay
    figures::Vector{Figure}
end
Base.displayable(::Capture, ::MIME"image/png") = true
Base.display(c::Capture, fig::Figure) = (push!(c.figures, fig); nothing)

function waitfor(predicate)
    ready = timedwait(predicate, 120.0; pollint = 0.01) == :ok
    @test ready
    ready || error("Timed out waiting for picker fixture publication")
end

@testset "Clearing live selections preserves observation and ownership" begin
    defaults = BATMakieVisualization()
    config = BATMakieVisualization(RECIPES, [11], 1, 1, 4, false,
        defaults.batch_growth_rate, 0.01, defaults.dark,
        defaults.triagonal_config, defaults.diagonal_config; observer_timeout = 120.0, message_duration = 4.5)
    vis = BATVisualizer(config)
    target = BAT.BATDistMeasure(MvNormal(zeros(11), ones(11)))
    algorithm = TransformedMCMC(nchains = 1, nwalkers = 1,
        adaptive_transform = BAT.NoAdaptiveTransform(), transform_tuning = NoMCMCTransformTuning())
    context = BATContext(rng = BAT.Philox4x((41, 73)), visualizer = vis)
    states = [BAT.MCMCState(algorithm, target, Int32(1), [zeros(11)], context)]
    outputs = BAT._empty_chain_outputs.(states)
    capture = Capture(Figure[])
    pushdisplay(capture)
    try
        BAT.init_visualizer!(vis; rng = BAT.get_rng(context), mcmc_states = states,
            outputs = outputs, target = target, f_pretransform = identity)
        fig = only(capture.figures)
        @test EXT._figure_controls(fig).notice.clock.duration == 4.5
        graph = EXT._live_graph(vis)
        clear = EXT._figure_controls(fig).picker.reset
        checks = checkboxes(fig)
        original_axis = only(axes(fig))

        for expected in (8, 16)
            producer = @async BAT.mcmc_iterate!!(outputs, states;
                max_nsteps = 8, update_visualizer = true)
            waitfor(() -> istaskdone(producer))
            states = fetch(producer)
            waitfor(() -> sum(graph[:flat_weights][]) == expected)
            if expected == 8
                @test graph[:idxs][] == [11]
                @test String(original_axis.xlabel[]) == "\$v_11\$"
                rng = copy(BAT.get_rng(states[1].chain_state.context))
                clear_selection!(fig)
                @test isempty(graph[:idxs][])
                @test all(cb -> !cb.checked[], checks)
                @test !original_axis.xlabelvisible[]
                @test sum(graph[:flat_weights][]) == 8
                @test BAT.get_rng(states[1].chain_state.context) == rng
            else
                @test isempty(graph[:idxs][])
                diagonal(checks, 1).checked[] = true
                @test graph[:idxs][] == [1]
                @test String(original_axis.xlabel[]) == "\$v_1\$"
                @test original_axis.xlabelvisible[]
                @test sum(graph[:flat_weights][]) == 16
            end
        end

        BAT.finish_visualizer!(vis; success = true)
        @test BAT.visualizer_status(vis).data_complete
        @test BAT.visualizer_figure(vis) === fig
        clear_selection!(fig)
        @test isempty(graph[:idxs][])
        diagonal(checks, 1).checked[] = true
        @test graph[:idxs][] == [1]
        @test only(axes(fig)) === original_axis
        @test sum(only(only(outputs)).weight) == 16

        # A reset nested in an older selection must win after the owner drains updates.
        picker = EXT._figure_controls(fig).picker
        armed = Ref(true)
        reentrant = on(graph[:idxs]) do _
            if armed[]
                armed[] = false
                picker.reset!()
            end
        end
        try
            picker.submit!([2])
            @test graph[:idxs][] == [11]
            @test picker.state.selection == [11]
            @test picker.state.headers == picker.state.initial_headers
            @test picker.mode[] === :matrix
        finally
            off(reentrant)
        end
        # Selection actions nested in a non-selection graph update also settle.
        nested = on(graph[:show_stats_diag]) do enabled
            if enabled
                picker.submit!([2])
                picker.submit!([11])
            end
        end
        try
            EXT._live_graph_update!(vis; show_stats_diag = true)
            @test picker.state.selection == graph[:idxs][] == [11]
            @test diagonal(checks, 1).checked[] == false
            @test occursin("outside picker", picker.summary.label[])
        finally
            off(nested)
        end

        # A draft entered during a synchronous graph notification is newer than
        # the submission whose completion is still unwinding.
        picker.submit!([1])
        header = picker.headers[(1, :row)]
        header.replace!("9")
        armed[] = true
        newer_draft = on(graph[:idxs]) do _
            if armed[]
                armed[] = false
                header.replace!("8")
            end
        end
        try
            @test picker.commit_header!(1)
            @test graph[:idxs][] == [9]
            @test picker.header_drafts[1] == "8"
            @test header.box.displayed_string[] == "8"
        finally
            off(newer_draft)
        end
        picker.reset!()

        BAT.dispose_visualizer!(vis)
        waitfor(() -> !BAT.visualizer_status(vis).cleanup_pending)
        @test BAT.visualizer_status(vis).history_released
        @test isempty(Makie.Observables.listeners(clear.clicks))
        indices = copy(graph[:idxs][])
        Makie.notify(clear.clicks)
        @test graph[:idxs][] == indices
        @test isempty(fig.content)
    finally
        popdisplay(capture)
        BAT.request_visualizer_stop!(vis; reason = :disabled)
        BAT.finish_visualizer!(vis; success = false)
        BAT.dispose_visualizer!(vis)
    end
end

end
