# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestMakieHistogramColors

using BAT, Makie, CairoMakie, Test

const EXT = Base.get_extension(BAT, :BATMakieExt)
const CONFIG = merge(BAT._default_makie_triagonal_config(), (colormap=:viridis,))
const DIAG = BAT._default_makie_diagonal_config()
const NUMERIC = (upper=Hist2D, diagonal=Hist1D, lower=QuantileHist2D)
const REGIONS = (upper=QuantileHist2D, diagonal=Hist1D, lower=QuantileHist2D)
const CORNERS = [0.0 1.0 0.0 1.0; 0.0 0.0 1.0 1.0]

samples(scale=1.0; weights=ones(4)) = DensitySampleVector(
    [collect(v) for v in eachcol(scale .* CORNERS)], zeros(4); weight=copy(weights))
primitive(values) = (xedges=Float64.(0:length(values)), yedges=[0.0, 1.0],
    weights=reshape(Float64.(values), :, 1))

function admission_error(f)
    err = try
        f()
        nothing
    catch e
        e
    end
    @test err !== nothing
    if !isnothing(err)
        message = sprint(showerror, err)
        @test occursin("Hist2D visible heights", message)
        @test occursin("Float32 color values", message)
        @test !occursin("cmin == cmax", message)
    end
    return err
end

function numeric_heatmap(fig)
    return only([p for ax in fig.content if ax isa Axis for p in ax.scene.plots
        if p isa Makie.Heatmap && eltype(p[3][]) <: Real])
end

@testset "Hist2D native color admission" begin
    @testset "Independent range fixtures and both orientations" begin
        tiny, wide = floatmin(Float32), 2f0^125
        # Binary boundaries distinguish safe padding, a normal denominator,
        # and reciprocal headroom. Expected admission does not call the guard.
        cases = [
            ("empty", Float64[], true), ("blank", [NaN, NaN], true),
            ("ordinary", [1.0, 2.0, NaN], true), ("constant", [2.0, 2.0], true),
            ("uniform subnormal", fill(Float64(nextfloat(0f0)), 2), true),
            ("accepted coalescing", [6666.666666, 6666.666667], true),
            ("asymmetric padding", [Float64(2f0^23)], true),
            ("padding odd tie", [Float64(2f0^23 + 1f0)], true),
            ("padding even tie", [Float64(2f0^23 + 2f0)], false),
            ("uniform large", fill(2.5e13, 2), false),
            ("narrowed uniform", [2.5e13, nextfloat(2.5e13)], false),
            ("subnormal span", Float64.([tiny, nextfloat(tiny)]), false),
            ("below minimum span", Float64.([tiny, prevfloat(2tiny)]), false),
            ("minimum span", Float64.([tiny, 2tiny]), true),
            ("above minimum span", Float64.([tiny, nextfloat(2tiny)]), true),
            ("below reciprocal limit", Float64.([1f0, prevfloat(wide)]), true),
            ("reciprocal limit", Float64.([1f0, wide]), true),
            ("above reciprocal limit", Float64.([1f0, nextfloat(wide)]), false),
            ("subnormal reciprocal", [1.0, Float64(floatmax(Float32))], false),
            ("overflow", [1e50], false), ("underflow", [1e-50], false),
            ("zero", [0.0], false), ("negative", [-1.0], false),
            ("infinity", [Inf], false),
        ]
        for (label, values, allowed) in cases, transposed in (false, true)
            @testset "$label transposed=$transposed" begin
                raw = primitive(values)
                before = deepcopy(raw)
                if !allowed
                    admission_error(() -> EXT.compose_plotspecs(raw, Hist2D(), CONFIG; transposed))
                elseif isempty(values)
                    @test isempty(EXT.compose_plotspecs(raw, Hist2D(), CONFIG; transposed))
                else
                    spec = only(EXT.compose_plotspecs(raw, Hist2D(), CONFIG; transposed))
                    fig = Figure()
                    try
                        plot = heatmap!(Axis(fig[1,1]), spec.args...; spec.kwargs...)
                        expected = transposed ? permutedims(raw.weights) : raw.weights
                        @test isequal(spec.args[3], expected)
                        @test isequal(plot[3][], Float32.(expected))
                        @test plot[1][] == (transposed ? raw.yedges : raw.xedges)
                        @test plot[2][] == (transposed ? raw.xedges : raw.yedges)
                        if any(!isnan, values)
                            lo, hi = plot.scaled_colorrange[]
                            @test lo isa Float32 && hi isa Float32 && lo < hi
                            span, reciprocal = hi - lo, inv(hi - lo)
                            @test isfinite(span) && span >= tiny
                            @test isfinite(reciprocal) && reciprocal >= 2tiny
                            visible = filter(!isnan, vec(plot.scaled_color[]))
                            @test all(isfinite, (visible .- lo) ./ span)
                            @test all(isfinite, (visible .- lo) .* reciprocal)
                            if label == "asymmetric padding"
                                @test (lo, hi) == (2f0^23 - 0.5f0, 2f0^23)
                            end
                        end
                        @test !isempty(Makie.colorbuffer(fig))
                    finally
                        empty!(fig)
                    end
                end
                @test isequal(raw, before)
            end
        end
    end

    @testset "Blank and mixed masks render transparently" begin
        counts = Int[]
        for values in ([NaN, NaN], [2.0, 2.0], [NaN, 2.0])
            raw = primitive(values)
            spec = only(EXT.compose_plotspecs(raw, Hist2D(), CONFIG))
            fig = Figure(size=(320,160), figure_padding=0)
            try
                axis = Axis(fig[1,1]; limits=(0.0, 2.0, 0.0, 1.0))
                hidedecorations!(axis)
                hidespines!(axis)
                plot = heatmap!(axis, spec.args...; spec.kwargs...)
                pixels = Makie.colorbuffer(fig)
                push!(counts, count(pixels) do color
                    rgb = Makie.RGBf(color)
                    max(rgb.r, rgb.g, rgb.b) - min(rgb.r, rgb.g, rgb.b) > 0.1
                end)
                @test isequal(isnan.(plot[3][]), isnan.(raw.weights))
            finally
                empty!(fig)
            end
        end
        @test counts[1] == 0
        @test counts[2] > 1000
        @test 0.45 < counts[3] / counts[2] < 0.55
    end

    @testset "Native styles and reversal on admitted data" begin
        for rev in (false, true)
            Makie.with_theme(Heatmap=(colorscale=log10, colorrange=(1.0, 10.0), alpha=0.4)) do
                spec = only(EXT.compose_plotspecs(primitive([2.0, 8.0]), Hist2D(), merge(CONFIG, (; rev))))
                @test !haskey(spec.kwargs, :colorscale) && !haskey(spec.kwargs, :colorrange)
                fig = Figure()
                try
                    plot = heatmap!(Axis(fig[1,1]), spec.args...; spec.kwargs...)
                    @test plot.colorscale[] === log10
                    @test plot.colorrange[] == Makie.Vec2f(1, 10)
                    @test plot.scaled_colorrange[] == Makie.Vec2f(0, 1)
                    palette = Makie.to_colormap(:viridis)
                    @test Makie.to_colormap(plot.colormap[]) == (rev ? reverse(palette) : palette)
                    @test plot.alpha[] == 0.4
                    @test !isempty(Makie.colorbuffer(fig))
                finally
                    empty!(fig)
                end
            end
        end
        Makie.with_theme(Heatmap=(colorrange=(1e13, 3e13),)) do
            admission_error(() -> EXT.compose_plotspecs(primitive([2.5e13]), Hist2D(), CONFIG))
        end
        Makie.with_theme(Heatmap=(colorscale=log10, colorrange=(1.0, 1e4), alpha=0.4)) do
            fig = Figure()
            try
                Makie.plot(fig[1,1], samples(); recipes=NUMERIC, vsel=[1,2], N_max=2)
                plot = numeric_heatmap(fig)
                @test plot.colorscale[] === log10
                @test plot.colorrange[] == Makie.Vec2f(1, 1e4)
                @test plot.alpha[] == 0.4
                @test !isempty(Makie.colorbuffer(fig))
            finally
                empty!(fig)
            end
        end
    end

    @testset "Public rejection, numeric and region controls, zero mass" begin
        for scale in (1.0, 1e-5), regions in (false, true), ordinary in (false, true), zero_mass in (false, true)
            sample = samples(scale; weights=zero_mass ? zeros(4) : ones(4))
            before = deepcopy(sample)
            recipes = regions ? REGIONS : NUMERIC
            fig = ordinary ? Figure() : nothing
            build() = begin
                if ordinary
                    Makie.plot(fig[1,1], sample; recipes, vsel=[1,2], N_max=2)
                else
                    fig = bat_makie_plot(sample, recipes, [1,2], 2; display=false)
                end
                @test !isempty(Makie.colorbuffer(fig))
            end
            try
                if scale == 1e-5 && !regions && !zero_mass
                    admission_error(build)
                else
                    build()
                end
            finally
                isnothing(fig) || empty!(fig)
            end
            @test sample == before
        end
    end

    @testset "Ordinary rejected update disposes and recreates" begin
        input = Observable(samples())
        fig = Figure()
        wrapper = nothing
        try
            wrapper = Makie.plot(fig[1,1], input; recipes=NUMERIC, vsel=[1,2], N_max=2)
            @test !isempty(Makie.colorbuffer(fig))
            admission_error() do
                input[] = samples(1e-5)
                Makie.colorbuffer(fig)
            end
        finally
            empty!(fig)
            # The ordinary grid returns a separate caller-owned native handle.
            isnothing(wrapper) || Makie.free(wrapper)
        end
        @test isempty(fig.content)
        @test isempty(Makie.Observables.listeners(input))
        fresh = Figure()
        try
            Makie.plot(fresh[1,1], samples(); recipes=NUMERIC, vsel=[1,2], N_max=2)
            @test !isempty(Makie.colorbuffer(fresh))
        finally
            empty!(fresh)
        end
    end

    @testset "Construction rollback and public recipe control" begin
        other = Figure()
        lines!(Axis(other[1,1]), [0.0, 1.0], [1.0, 0.0])
        before = copy(Makie.colorbuffer(other))
        font = Makie.to_value(Makie.theme(:fontsize))
        background = Makie.to_value(Makie.theme(:backgroundcolor))
        try
            admission_error(() -> bat_makie_plot(samples(1e-5), NUMERIC, [1,2], 2; display=false))
            failed = Makie.current_figure()
            @test failed !== other && isempty(failed.content)
            @test isempty(failed.scene.deregister_callbacks)
            @test Makie.colorbuffer(other) == before
            @test Makie.to_value(Makie.theme(:fontsize)) == font
            @test Makie.to_value(Makie.theme(:backgroundcolor)) == background
            empty!(failed)
            fig = bat_makie_plot(samples(1e-5), REGIONS, [1,2], 2; display=false)
            menu = first(filter(x -> x isa Menu && any(o -> last(o) === Hist2D, x.options[]), fig.content))
            try
                @test !isempty(Makie.colorbuffer(fig))
                admission_error(() -> (menu.selection[] = Hist2D))
            finally
                empty!(fig)
            end
            @test isempty(fig.content)
            @test isempty(Makie.Observables.listeners(menu.selection))
            fresh = bat_makie_plot(samples(1e-5), REGIONS, [1,2], 2; display=false)
            try
                @test !isempty(Makie.colorbuffer(fresh))
            finally
                empty!(fresh)
            end
        finally
            empty!(other)
        end
    end

    @testset "Bound grid masks and live rejection ownership" begin
        sample = samples(1e-5)
        before = deepcopy(sample)
        pair = merge(CONFIG, (normalization=:probability,))
        graph, _ = EXT._setup_static_graph(sample, NUMERIC, [1,2], 2, pair, DIAG)
        fig = Figure()
        other = Figure()
        lines!(Axis(other[1,1]), [0.0, 1.0], [1.0, 0.0])
        vis = BATVisualizer(BATMakieVisualization())
        EXT._claim_live_visualizer!(vis)
        try
            EXT._bind_interactive_grid!(fig, EXT._init_gridlayout!(graph, 2))
            for data in (samples(1e-5; weights=zeros(4)), sample, sample[1:0], sample)
                Makie.update!(graph; samples=[[data]], current_idxs=[[length(data)]])
                @test !isempty(Makie.colorbuffer(fig))
            end
            # Real view/listener ownership; no sampler-equivalence claim.
            @test EXT._with_live_view(vis) do
                EXT._attach_live_view!(vis, fig)
            end
            EXT._start_live_listener!(vis, () -> nothing, () -> nothing)
            @test !EXT._with_live_view(vis) do
                Makie.update!(graph; triagonal_config=merge(pair, (normalization=:pdf,)))
            end
            BAT.finish_visualizer!(vis; success=true)
            status = BAT.visualizer_status(vis)
            @test status.outcome === :failed
            @test !status.data_complete && !status.presentation_valid
            @test status.quiesced && status.view_disposed && !status.cleanup_pending
            @test status.error.phase === :view
            @test isempty(fig.content)
            called = Ref(false)
            @test !EXT._with_live_view(vis) do
                called[] = true
            end
            @test !called[]
            @test !isempty(Makie.colorbuffer(other))
            @test sample == before
        finally
            BAT.request_visualizer_stop!(vis)
            BAT.finish_visualizer!(vis; success=false)
            empty!(fig)
            empty!(other)
        end
    end
end
end # module
